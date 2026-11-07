"""Isaac Lab boundary: real snapshots before auto-reset and external state writes.

Private manager buffers are intentionally localized here and checked at startup.
The adapter is experimental until its real-simulator parity gate passes.
"""

from __future__ import annotations
import hashlib
import os
from pathlib import Path
import torch
from .physics import Kinematics, ActionMap, amp_features, apply, inverse
from .observation import OracleObservation
from .oracle_env import box_dimensions, box_mass
from .task import HISTORY_FIELDS
from .events import physical_write_mask, macro_step_target


MEMORY_FIELDS = ("_bed_hold_steps", "_tp_count", "_box_regrace", "_v38_cradle_ticks")
EXPECTED_TERMS = (
    "command",
    "motion_anchor_pos_b",
    "motion_anchor_ori_b",
    "base_lin_vel",
    "base_ang_vel",
    "joint_pos",
    "joint_vel",
    "actions",
    "balloon_pos_b",
    "balloon_lin_vel_b",
    "box_ori",
    "box_ang",
    "box_dimensions",
    "box_mass",
)


class IsaacAdapter:
    def __init__(
        self,
        env,
        urdf,
        monitor_contacts=False,
        learn_external_events=False,
        parity_every=1,
        cache_context=False,
        native_collection_inputs=False,
    ):
        if parity_every < 1:
            raise ValueError("parity_every must be positive")
        self.parity_every = parity_every
        self.native_collection_inputs = native_collection_inputs
        self._before_observation = self._before_amp = None
        self.property_cache = None
        self.monitor_contacts = monitor_contacts
        self.learn_external_events = learn_external_events
        self.env = env.unwrapped
        e = self.env
        self.device = e.device
        self.robot, self.box = e.scene["robot"], e.scene["balloon"]
        if cache_context:
            from .context_cache import ContextPropertyCache

            self.property_cache = ContextPropertyCache(
                {"robot": self.robot, "box": self.box}, device=self.device
            )
        self.command = e.command_manager.get_term("motion")
        self.term = e.action_manager.get_term("joint_pos")
        self.events = torch.zeros(e.num_envs, dtype=torch.bool, device=self.device)
        self.pre_reset = None
        self.real_steps = 0
        self._contact_seen = torch.zeros(
            e.num_envs, dtype=torch.bool, device=self.device
        )
        self._before = None
        self.event_scheduler = None
        self.event_plan = None
        self.reward_names = list(e.reward_manager.active_terms)
        self.joint_names = list(self.robot.data.joint_names)
        if len(self.joint_names) != 29 or e.action_manager.total_action_dim != 29:
            raise ValueError("Only full 29-joint G1 action/state contract is supported")
        if list(e.observation_manager.active_terms["policy"]) != list(EXPECTED_TERMS):
            raise ValueError(
                f"Unexpected Oracle observation layout: {e.observation_manager.active_terms['policy']}"
            )
        if abs(e.step_dt - 0.02) > 1e-8 or e.cfg.decimation != 4:
            raise ValueError(
                "First version requires dt=.02 and four .005s physics steps"
            )
        if (
            os.environ.get("BOX_V38_TS_FROM_PERC") != "0"
            or os.environ.get("BOX_DP_EST") != "0"
        ):
            raise ValueError(
                "Oracle requires detector OFF and launch-driven arm trigger"
            )
        if os.environ.get("BOX_V38_ARM_TWOSTAGE") != "1":
            raise ValueError("Expected the two-stage arm controller")
        ids = self.term._joint_ids
        if not isinstance(ids, slice) and list(ids) != list(range(29)):
            raise ValueError("Action order differs from simulator joint order")
        # Populate lazy two-stage/reference caches without advancing the simulator.
        self.term.process_actions(torch.zeros((e.num_envs, 29), device=self.device))
        if not hasattr(self.term, "_v38_ts_cache"):
            raise ValueError("Two-stage controller cache was not initialized")
        self.term._v38_ts_cache[2].zero_()
        _ = self.command.joint_pos
        cols, _, _ = self.term._v38_ts_cache
        from reflex.envs import teacher_terms as tt

        self.action_map = ActionMap(
            cols.tolist(),
            ramp=tt.V38_TS_RAMP,
            delta_min=tt.V38_TS_DELTA_MIN,
            delta_gate=tt.V38_TS_DELTA_GATE,
            base_blend=tt.V38_TS_BASE_BLEND,
        ).to(self.device)
        fk = Kinematics(urdf, self.joint_names).to(self.device)
        c = self.command
        if not hasattr(c, "_v38_arm_cache"):
            raise ValueError("Reference arm patch expected")
        idx, down = c._v38_arm_cache
        self.bridge = OracleObservation(
            fk,
            c.motion.joint_pos,
            c.motion.joint_vel,
            c.motion.body_pos_w[:, c.motion_anchor_body_index],
            c.motion.body_quat_w[:, c.motion_anchor_body_index],
            self.robot.data.default_joint_pos[0],
            self.robot.data.default_joint_vel[0],
            c.cfg.anchor_body_name,
            idx,
            down,
        ).to(self.device)
        self.context_schema = []
        context = self.context()
        context_offsets = {}
        cursor = 0
        for key, width in self.context_schema:
            context_offsets[key] = cursor
            cursor += width
        com_indices = (
            context_offsets["robot.coms"] + 7 * self.robot.body_names.index("pelvis"),
            context_offsets["box.coms"],
        )
        self.contract = {
            "state_layout": "g1-71_world_angvel+box13_wxyz_v1",
            "joint_names": self.joint_names,
            "observation_terms": list(EXPECTED_TERMS),
            "context_schema": self.context_schema,
            "memory": list(MEMORY_FIELDS) + ["episode_seconds", "since_launch_seconds"],
            "context_dim": context.shape[-1],
            "com_indices": com_indices,
            "dt": e.step_dt,
            "decimation": e.cfg.decimation,
            "urdf_sha256": hashlib.sha256(Path(urdf).read_bytes()).hexdigest(),
            "motion_sha256": hashlib.sha256(
                Path(c.cfg.motion_file).read_bytes()
            ).hexdigest(),
            "flags": {
                k: v
                for k, v in sorted(os.environ.items())
                if k.startswith(("BOX_", "AMP_W_", "M5_"))
            },
            "task_schema_version": 2,
            "task_history_fields": list(HISTORY_FIELDS),
            "body_names": list(self.robot.body_names),
            "tracked_names": list(c.cfg.body_names),
            "max_episode_steps": e.max_episode_length,
            "termination_names": list(e.termination_manager.active_terms),
        }
        self.task = None
        self.task_parity_steps = 0
        self.task_parity_max = torch.zeros(len(self.reward_names), device=self.device)
        # Capture exactly the post-physics state used for real rewards, before
        # Isaac automatically resets terminated environments or injects a throw.
        original_reward = e.reward_manager.compute

        def reward_compute(dt):
            from .experience import pair_contact, clean2_intercept_candidate

            contact, known = pair_contact(
                e.scene.sensors.get("wm_box_contacts")
                if self.monitor_contacts
                else None,
                e.num_envs,
                self.device,
            )
            phase = torch.where(contact, 1, torch.where(self._contact_seen, 2, 0))
            phase = torch.where(known, phase, -1)
            if self.monitor_contacts:
                self._contact_seen |= contact
            loose = torch.zeros(e.num_envs, dtype=torch.bool, device=self.device)
            loose_known = torch.zeros_like(loose)
            if "catch_success_bonus" in self.reward_names:
                spec = e.reward_manager.get_term_cfg("catch_success_bonus")
                if spec.func.__name__ == "catch_success_bonus":
                    loose = spec.func(e, **spec.params).bool()
                    loose_known.fill_(True)
            sensor = (
                e.scene.sensors.get("wm_box_contacts")
                if self.monitor_contacts
                else None
            )
            pair_peak = net_peak = torch.zeros(e.num_envs, device=self.device)
            if sensor is not None:
                data = sensor.data
                pair_peak = data.force_matrix_w_history.norm(dim=-1).flatten(1).amax(1)
                net_peak = data.net_forces_w_history.norm(dim=-1).flatten(1).amax(1)
            previous_history = self.task_history()
            result = original_reward(dt)
            self.pre_reset = {
                "state": self.physical(),
                "memory": self.memory(),
                "amp": self.real_amp(),
                "reward_terms": e.reward_manager._step_reward.clone() * dt,
                "terminated": e.termination_manager.terminated.clone(),
                "timeout": e.termination_manager.time_outs.clone(),
                "task_history_after_reward": self.task_history(),
                "intercept_candidate": clean2_intercept_candidate(
                    self.box.data.root_pos_w,
                    self.box.data.root_lin_vel_w,
                    self.robot.data.body_pos_w[
                        :, self.robot.body_names.index("torso_link")
                    ],
                    getattr(
                        e,
                        "_balloon_radius_per_env",
                        torch.full((e.num_envs,), 0.1016, device=self.device),
                    ),
                ),
                "loose_catch": loose,
                "loose_catch_known": loose_known,
                "pair_force_peak": pair_peak,
                "net_force_peak": net_peak,
                "box_contact": contact,
                "contact_known": known,
                "contact_phase": phase,
                "termination_reasons": torch.stack(
                    [
                        e.termination_manager.get_term(n).clone()
                        for n in e.termination_manager.active_terms
                    ],
                    -1,
                )
                if hasattr(e.termination_manager, "get_term")
                else torch.zeros(
                    e.num_envs,
                    len(e.termination_manager.active_terms),
                    dtype=torch.bool,
                    device=self.device,
                ),
                "termination_reasons_known": torch.full(
                    (e.num_envs,),
                    hasattr(e.termination_manager, "get_term"),
                    dtype=torch.bool,
                    device=self.device,
                ),
            }
            if self.parity_due() and not torch.allclose(
                self.pre_reset["reward_terms"].sum(-1), result, atol=1e-5, rtol=1e-5
            ):
                raise RuntimeError("Isaac reward term weighting/dt contract changed")
            if self.task is not None and self._before is not None:
                self.pre_reset["termination_features"] = (
                    self.native_termination_features()
                )
            if self.task is not None and self._before is not None and self.parity_due():
                b = self._before
                analytic = self.task.evaluate_step(
                    b,
                    b["state"],
                    self.pre_reset["state"],
                    self.term.raw_actions,
                    b["previous_action"],
                    b["phase"],
                    previous_history,
                    b["episode_steps"],
                    b["next_external_event_seconds"],
                    0,
                )
                torch.testing.assert_close(
                    self.pre_reset["termination_features"],
                    analytic.termination_features,
                    atol=2e-4,
                    rtol=1e-5,
                )
                err = (analytic.rewards - self.pre_reset["reward_terms"]).abs().amax(0)
                self.task_parity_max = torch.maximum(self.task_parity_max, err.detach())
                exact = self.task_exact_mask
                if bool((err[exact] > 2e-4).any()):
                    report = {
                        name: float(error)
                        for name, error in zip(self.reward_names, err)
                        if float(error) > 2e-4
                    }
                    raise RuntimeError(f"Analytic task reward parity failed: {report}")
                if not torch.equal(analytic.terminated, self.pre_reset["terminated"]):
                    raise RuntimeError("Analytic termination parity failed")
                # Interval-event counters are updated later. These reward-history
                # counters must already match the pre-reset reward snapshot.
                cols = [
                    HISTORY_FIELDS.index(n)
                    for n in ("_bed_hold_steps", "_tp_count", "_chest_v5_persist")
                ]
                if not torch.equal(
                    analytic.history[:, cols],
                    self.pre_reset["task_history_after_reward"][:, cols],
                ):
                    raise RuntimeError("Analytic reward-history parity failed")
                self.task_parity_steps += self.env.num_envs
            return result

        e.reward_manager.compute = reward_compute
        original_process = self.term.process_actions

        def process(raw):
            original_process(raw)
            self.real_target = self.term.processed_actions.clone()
            self.real_blend = self.term._v38_ts_cache[2].clone()[:, None]

        self.term.process_actions = process
        # Compare buffers around actual writes: hover implementation writes ALL
        # envs, including unchanged launched boxes, so marking all IDs is wrong.
        for asset in (self.robot, self.box):
            for method in (
                "write_root_pose_to_sim",
                "write_root_velocity_to_sim",
                "write_joint_state_to_sim",
            ):
                if hasattr(asset, method):
                    self._watch_write(asset, method)

    def _watch_write(self, asset, name):
        original = getattr(asset, name)

        def write(*args, **kwargs):
            before = self.physical()
            if self.event_plan is not None:
                from .events_v24.writes import record_native

                record_native(
                    self.event_plan,
                    before,
                    self.env.scene.env_origins,
                    asset is self.robot,
                    name,
                    args,
                    kwargs,
                    getattr(asset, "_reflex_event_write_mask", None),
                )
            result = original(*args, **kwargs)
            after = self.physical()
            self.events |= physical_write_mask(before, after)
            return result

        setattr(asset, name, write)

    def attach_task(self, task):
        self.task = task
        self.task_exact_mask = torch.tensor(
            [
                r["kind"] != "ground_height_proxy_no_force_or_self_contact"
                for r in task.specification()["rewards"]
            ],
            device=self.device,
        )

    def parity_due(self):
        # Always check the startup window, then sample explicitly.
        return self.real_steps <= 8 or self.real_steps % self.parity_every == 0

    def native_termination_features(self):
        cmd = self.command
        gravity = self.robot.data.GRAVITY_VEC_W
        return torch.cat(
            (
                cmd.anchor_pos_w - cmd.robot_anchor_pos_w,
                apply(inverse(cmd.anchor_quat_w), gravity)
                - apply(inverse(cmd.robot_anchor_quat_w), gravity),
                (cmd.body_pos_relative_w - cmd.robot_body_pos_w).flatten(-2),
            ),
            -1,
        ).clone()

    def physical(self):
        d, b = self.robot.data, self.box.data
        origins = self.env.scene.env_origins
        return torch.cat(
            (
                d.root_pos_w - origins,
                d.root_quat_w,
                d.joint_pos,
                d.root_lin_vel_w,
                d.root_ang_vel_w,
                d.joint_vel,
                b.root_pos_w - origins,
                b.root_quat_w,
                b.root_lin_vel_w,
                b.root_ang_vel_w,
            ),
            -1,
        ).clone()

    def memory(self):
        e = self.env
        zero = torch.zeros(e.num_envs, device=self.device)
        vals = [
            getattr(e, k, zero).float().reshape(e.num_envs, 1) * e.step_dt
            for k in MEMORY_FIELDS
        ]
        vals.extend(
            (
                e.episode_length_buf[:, None].float() * e.step_dt,
                (e.episode_length_buf - getattr(e, "_box_launch_step", zero))[
                    :, None
                ].float()
                * e.step_dt,
            )
        )
        return torch.cat(vals, -1).clone()

    def task_history(self):
        zero = torch.zeros(self.env.num_envs, device=self.device)
        return torch.stack(
            [getattr(self.env, name, zero).float() for name in HISTORY_FIELDS], -1
        ).clone()

    def next_event_seconds(self):
        manager = self.env.event_manager
        result = torch.full((self.env.num_envs,), float("inf"), device=self.device)
        configs = manager._mode_term_cfgs.get("interval", [])
        for cfg, timer in zip(configs, manager._interval_term_time_left):
            if cfg.interval_range_s[1] > 0:
                result = torch.minimum(result, timer.expand_as(result))
        return result.clone()

    def context(self):
        n = self.env.num_envs
        values = [box_dimensions(self.env)]
        schema = [("box.dimensions", 3)]
        for name, asset in (("box", self.box), ("robot", self.robot)):
            view = asset.root_physx_view
            for prop in ("masses", "inertias", "coms", "material_properties"):
                value = (
                    (
                        self.property_cache.get(name, asset, prop)
                        if self.property_cache is not None
                        else getattr(view, "get_" + prop)()
                    )
                    .to(self.device)
                    .float()
                    .reshape(n, -1)
                )
                values.append(value)
                schema.append((name + "." + prop, value.shape[-1]))
        # Actual effective actuator properties, including randomization.
        for prop in (
            "joint_stiffness",
            "joint_damping",
            "joint_armature",
            "joint_friction_coeff",
            "joint_effort_limits",
        ):
            if not hasattr(self.robot.data, prop):
                raise ValueError(
                    f"Missing actual actuator property {prop}; update adapter explicitly"
                )
            value = getattr(self.robot.data, prop).float().reshape(n, -1)
            values.append(value)
            schema.append(("robot." + prop, value.shape[-1]))
        self.context_schema = schema
        return torch.cat(values, -1).clone()

    def real_amp(self):
        # Same legacy AMP convention, but resolved by body names.
        from .physics import apply, heading, inverse, mul
        from reflex.amp.context import _rot6d

        d = self.robot.data
        q = d.root_quat_w
        hi = inverse(heading(q))
        p = d.root_pos_w
        keys = [
            apply(hi, d.body_pos_w[:, self.robot.body_names.index(name)] - p)
            for name in (
                "left_wrist_yaw_link",
                "right_wrist_yaw_link",
                "left_ankle_roll_link",
                "right_ankle_roll_link",
            )
        ]
        return torch.cat(
            (
                d.joint_pos,
                d.joint_vel,
                p[:, 2:3] - self.env.scene.env_origins[:, 2:3],
                _rot6d(mul(hi, q)),
                apply(hi, d.root_lin_vel_w),
                apply(hi, d.root_ang_vel_w),
                *keys,
            ),
            -1,
        ).clone()

    def snapshot(self):
        e = self.env
        n = e.num_envs
        expand = lambda v: torch.as_tensor(v, device=self.device).expand(n, 29).clone()
        cols, down, blend = self.term._v38_ts_cache
        # Process-only clipping is separate from the gimbal's per-substep clamp.
        limits = torch.full((n, 29, 2), float("inf"), device=self.device)
        limits[..., 0] *= -1
        if os.environ.get("BOX_V38_TGT_CLIP", "0") == "1":
            lo, hi = self.term._v38_lim
            limits = torch.stack((lo.expand(n, 29), hi.expand(n, 29)), -1)
        from .experience import pair_contact

        contact, known = pair_contact(
            e.scene.sensors.get("wm_box_contacts") if self.monitor_contacts else None,
            n,
            self.device,
        )
        # Pair history precedes interval teleports/reset. Do not attribute stale
        # forces to a new throw snapshot before its first physics transition.
        known &= ~self.events
        if self.real_steps == 0:
            known.fill_(False)
        start_contact_phase = torch.where(
            known, torch.where(contact, 1, torch.where(self._contact_seen, 2, 0)), -1
        )
        state = self.physical()
        dist = (state[:, 71:74] - state[:, :3]).norm(dim=-1)
        stage = torch.where(
            dist > 1.0,
            0,
            torch.where(
                dist > 0.5, 1, torch.where(state[:, 78:81].norm(dim=-1) > 2, 2, 3)
            ),
        )
        result = {
            "state": state,
            "context": self.context(),
            "memory": self.memory(),
            "previous_action": e.action_manager.action.clone(),
            "phase": self.command.time_steps.clone(),
            "yaw": self.command.yaw_offsets.clone(),
            "launched": e._box_launched.clone(),
            "dimensions": box_dimensions(e).clone(),
            "mass": (
                self.property_cache.get("box", self.box, "masses")
                .to(self.device)
                .reshape(n, 1)
                if self.property_cache is not None
                else box_mass(e)
            ).clone(),
            "default_q": self.robot.data.default_joint_pos.clone(),
            "default_dq": self.robot.data.default_joint_vel.clone(),
            "blend": blend.clone()[:, None],
            "scale": expand(self.term._scale),
            "offset": expand(self.term._offset),
            "down": down.expand(n, len(cols)).clone(),
            "up": self.term._up_arm.expand(n, len(cols)).clone(),
            "limits": limits,
            "stage": stage,
            "start_contact_phase": start_contact_phase,
            "reference_length": torch.full(
                (n,),
                len(self.command.motion.joint_pos),
                device=self.device,
                dtype=torch.long,
            ),
        }
        limits = getattr(
            self.robot.data, "joint_pos_limits", self.robot.data.soft_joint_pos_limits
        )
        result.update(
            {
                "task_history": self.task_history(),
                "reference_relative_p": (
                    self.command.body_pos_relative_w - e.scene.env_origins[:, None, :]
                ).clone(),
                "reference_relative_q": self.command.body_quat_relative_w.clone(),
                "episode_steps": e.episode_length_buf.clone(),
                "common_step": torch.full_like(
                    e.episode_length_buf, e.common_step_counter
                ),
                "launch_step": e._box_launch_step.clone(),
                "box_radius": getattr(
                    e,
                    "_balloon_radius_per_env",
                    torch.full((n,), 0.1016, device=self.device),
                ).clone(),
                "hard_joint_limits": limits.clone(),
                "soft_joint_limits": self.robot.data.soft_joint_pos_limits.clone(),
                "next_external_event_seconds": self.next_event_seconds(),
                "start_valid": torch.isfinite(state).all(-1)
                & torch.isfinite(result["context"]).all(-1),
            }
        )
        if self.event_scheduler is not None:
            result.update(self.event_scheduler.snapshot(self.env))
        self._before = result
        if self.native_collection_inputs:
            self._before_observation = self.env.observation_manager.compute_group(
                "policy"
            ).clone()
            self._before_amp = self.real_amp()
            if self.parity_due():
                self.parity(result)
        if getattr(self, "wm_context_spec", None) is not None:
            self.wm_context_guard.check(result["context"])
            result["wm_context"] = self.wm_context_spec.select(result["context"])
        return result

    def parity(self, b, atol=2e-4, critic=False):
        """Compare pure Torch reconstruction with live manager observations / links."""
        reconstructed = self.bridge(
            b["state"],
            b["previous_action"],
            b["phase"],
            b["yaw"],
            b["launched"],
            b["dimensions"],
            b["mass"],
            b["default_q"],
            b["default_dq"],
        )
        actual = self.env.observation_manager.compute_group("policy")
        amp = amp_features(b["state"], self.bridge.fk)
        errors = {
            "observation_max": float((reconstructed - actual).abs().max()),
            "amp_max": float((amp - self.real_amp()).abs().max()),
        }
        if critic:
            from .observation import PrivilegedObservation

            observed = self.env.observation_manager.compute_group("critic")
            reconstructed = PrivilegedObservation(
                self.bridge, self.command.cfg.body_names
            ).to(self.device)(
                b["state"],
                b["previous_action"],
                b["phase"],
                b["yaw"],
                b["launched"],
                b["dimensions"],
                b["mass"],
                b["default_q"],
                b["default_dq"],
            )
            if observed.shape != reconstructed.shape or observed.shape[-1] != 292:
                raise RuntimeError(
                    f"Privileged critic layout mismatch: {observed.shape} vs {reconstructed.shape}"
                )
            errors["critic_observation_max"] = float(
                (observed - reconstructed).abs().max()
            )
        if max(errors.values()) > atol:
            raise RuntimeError(f"FK / observation parity failed: {errors}")
        return errors

    def step(self, raw, expected_target, expected_blend):
        b = self._before
        if b is None:
            raise RuntimeError("snapshot() must precede step()")
        self.events.zero_()
        self.real_steps += 1
        self.pre_reset = None
        if self.event_scheduler is not None:
            from .events_v24.writes import WritePlan

            self.event_plan = WritePlan(b["state"])
        getattr(self, "step_environment", self.env.step)(raw)
        p = self.pre_reset
        if p is None:
            raise RuntimeError("Reward snapshot hook was not called")
        if self.parity_due() and not torch.allclose(
            self.real_target, expected_target, atol=2e-5, rtol=1e-5
        ):
            raise RuntimeError(
                f"Action target parity failed: {(self.real_target - expected_target).abs().max()}"
            )
        if self.parity_due() and not torch.allclose(
            self.real_blend, expected_blend, atol=1e-6
        ):
            raise RuntimeError("Two-stage controller state parity failed")
        done = p["terminated"] | p["timeout"]
        # Counters updated by non-physical interval events belong to next memory.
        memory = torch.where(done[:, None], p["memory"], self.memory())
        launch_event = self.env._box_launched.bool() & (
            ~b["launched"].bool() | (self.env._box_launch_step != b["launch_step"])
        )
        reference_jump = self.command.time_steps != b["phase"] + 1
        boundary = (
            self.events | done | launch_event | ~b["launched"].bool() | reference_jump
        )
        tr = {
            **b,
            "action": raw.clone(),
            "target": self.real_target,
            "next_blend": self.real_blend,
            "next_state": p["state"],
            "next_memory": memory,
            "reward_terms": p["reward_terms"],
            "terminated": p["terminated"],
            "timeout": p["timeout"],
            "boundary": boundary,
            "valid": b["start_valid"].clone(),  # all finite physical transitions
            "amp": self.real_amp_from_state_before(b),
            "next_amp": p["amp"],
        }
        if self.learn_external_events:
            tr["physical_next_state"] = p["state"]
            tr["next_state"], macro_mode_target, macro_reset = macro_step_target(
                p["state"],
                self.physical(),
                done,
                reference_jump,
                b["launched"],
                self.env._box_launched,
            )
            tr["next_memory"] = torch.where(macro_reset[:, None], p["memory"], memory)
            tr["external_boundary"] = (
                self.events.clone() | launch_event | ~b["launched"].bool()
            )
            tr["boundary"] = macro_reset
        if self.event_scheduler is not None:
            reset = done | reference_jump
            tr["event_targets"] = torch.where(
                reset[:, None], 0, self.event_plan.targets
            )
            tr["event_mask"] = self.event_plan.mask & ~reset[:, None]
            next_event = self.event_scheduler.snapshot(self.env)["event_state"]
            tr["next_event_state"] = torch.where(
                reset[:, None], b["event_state"], next_event
            )
            self.event_plan = None
        from reflex.envs.base.chest_v5_terms import STRICT_PERSIST_FRAMES

        tr.update(
            {
                k: p[k]
                for k in (
                    "intercept_candidate",
                    "loose_catch",
                    "loose_catch_known",
                    "box_contact",
                    "contact_known",
                    "contact_phase",
                    "termination_reasons",
                    "termination_reasons_known",
                )
            }
        )
        tr["strict_success"] = (
            p["task_history_after_reward"][:, -1] >= STRICT_PERSIST_FRAMES
        )
        tr["launch_event"] = launch_event
        tr["after_launched"] = self.env._box_launched.bool().clone()
        if self.learn_external_events:
            tr["after_launched"] = macro_mode_target
        self._contact_seen &= ~(done | tr["launch_event"])
        if "termination_features" in p:
            tr["termination_features"] = p["termination_features"]
        return tr

    def real_amp_from_state_before(self, b):
        # FK is accepted only after parity() has passed; no hard-coded body IDs.
        return (
            self._before_amp
            if self.native_collection_inputs
            else amp_features(b["state"], self.bridge.fk)
        )

    def manifest(self):
        e = self.env
        result = {
            "contract": self.contract,
            "rewards": [
                {
                    "name": name,
                    "weight": e.reward_manager.get_term_cfg(name).weight,
                    "function": e.reward_manager.get_term_cfg(name).func.__module__
                    + "."
                    + e.reward_manager.get_term_cfg(name).func.__name__,
                    "imagination": "learned_per_term",
                    "target": "original weighted reward contribution including control dt",
                }
                for name in self.reward_names
            ],
            "unavailable_rewards": [],
            "analytic_rewards": [],
            "warning": "All task reward terms are learned surrogates. GPU gradient/ranking validation required.",
        }
        if self.task is not None:
            result["task"] = self.task.specification()
            result["rewards"] = self.task.specification()["rewards"]
            result["warning"] = (
                "Original Torch formulas; configured force-only proxy is explicitly labeled. Neural WM learns state only."
            )
            result["task_parity_checked_transitions"] = self.task_parity_steps
            result["task_parity_max"] = {
                n: float(v) for n, v in zip(self.reward_names, self.task_parity_max)
            }
        return result
