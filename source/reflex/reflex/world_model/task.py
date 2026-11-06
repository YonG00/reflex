"""Task semantics outside the neural dynamics model.

Original Torch reward/termination functions run against a state-backed facade.
No PhysX simulator is called. Hard predicates/history remain deterministic;
smooth original reward paths retain autograd. Any force-only fallback is named
in the manifest instead of claiming sensor parity.
"""

from __future__ import annotations
from dataclasses import dataclass
from types import SimpleNamespace as NS
import re
import torch
from torch import nn
from .physics import apply, inverse, mul, heading

HISTORY_FIELDS = (
    "_bed_hold_steps",
    "_tp_count",
    "_box_regrace",
    "_v38_cradle_ticks",
    "_chest_v5_persist",
)


def primitive(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, slice):
        return {"slice": [value.start, value.stop, value.step]}
    if isinstance(value, (tuple, list)):
        return [primitive(v) for v in value]
    if isinstance(value, dict):
        return {k: primitive(v) for k, v in value.items()}
    if torch.is_tensor(value):
        return {"tensor": value.detach().cpu().tolist()}
    if hasattr(value, "name"):
        return {
            "entity": {
                k: primitive(getattr(value, k))
                for k in (
                    "name",
                    "body_names",
                    "body_ids",
                    "joint_names",
                    "joint_ids",
                    "preserve_order",
                )
                if hasattr(value, k)
            }
        }
    raise TypeError(f"Unsupported task specification parameter: {type(value)}")


@dataclass
class Term:
    name: str
    function: object
    params: dict
    weight: float = 1.0
    timeout: bool = False


class Asset:
    def __init__(self, data, body_names, joint_names):
        self.data = data
        self.body_names = body_names
        self.joint_names = joint_names
        self.device = data.root_pos_w.device

    def find_bodies(self, patterns, preserve_order=False):
        patterns = [patterns] if isinstance(patterns, str) else patterns
        ids = [
            i
            for i, n in enumerate(self.body_names)
            if any(re.fullmatch(p, n) for p in patterns)
        ]
        return ids, [self.body_names[i] for i in ids]

    def find_joints(self, patterns, preserve_order=False):
        patterns = [patterns] if isinstance(patterns, str) else patterns
        ids = [
            i
            for i, n in enumerate(self.joint_names)
            if any(re.fullmatch(p, n) for p in patterns)
        ]
        return ids, [self.joint_names[i] for i in ids]


class Scene(dict):
    pass


@dataclass
class TaskResult:
    rewards: torch.Tensor
    history: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    timeout: torch.Tensor
    phase: torch.Tensor
    episode_steps: torch.Tensor
    event_seconds: torch.Tensor
    termination_features: torch.Tensor


class AnalyticTask(nn.Module):
    """Reference task evaluator with explicit runtime dependencies and schema."""

    schema_version = 2

    def __init__(
        self,
        fk,
        body_names,
        joint_names,
        tracked_names,
        anchor_name,
        reference,
        rewards,
        terminations,
        context_schema,
        dt=0.02,
        max_steps=1500,
        rethrow_enabled=False,
        rethrow_grace=30,
        perturb_enabled=False,
        cradled=None,
    ):
        super().__init__()
        self.fk = fk
        self.body_names, self.joint_names, self.tracked_names = (
            body_names,
            joint_names,
            tracked_names,
        )
        self.anchor_name = anchor_name
        self.rewards, self.terminations = rewards, terminations
        self.dt, self.max_steps = dt, max_steps
        self.rethrow_enabled, self.rethrow_grace = rethrow_enabled, rethrow_grace
        self.perturb_enabled, self.cradled = perturb_enabled, cradled
        self._static_cache = {}
        for key, value in reference.items():
            self.register_buffer("ref_" + key, value.detach().clone())
        self.context_slices = {}
        offset = 0
        for key, size in context_schema:
            self.context_slices[key] = slice(offset, offset + size)
            offset += size

    @classmethod
    def from_environment(cls, adapter):
        e = adapter.env
        c = adapter.command
        rewards = [
            Term(
                n,
                e.reward_manager.get_term_cfg(n).func,
                e.reward_manager.get_term_cfg(n).params,
                e.reward_manager.get_term_cfg(n).weight,
            )
            for n in e.reward_manager.active_terms
        ]
        terms = [
            Term(
                n,
                e.termination_manager.get_term_cfg(n).func,
                e.termination_manager.get_term_cfg(n).params,
                timeout=e.termination_manager.get_term_cfg(n).time_out,
            )
            for n in e.termination_manager.active_terms
        ]
        from reflex.envs import teacher_terms as tt

        obj = cls(
            adapter.bridge.fk,
            list(adapter.robot.body_names),
            adapter.joint_names,
            list(c.cfg.body_names),
            c.cfg.anchor_body_name,
            {
                "q": c.motion.joint_pos,
                "dq": c.motion.joint_vel,
                "p": c.motion.body_pos_w,
                "quat": c.motion.body_quat_w,
                "v": c.motion.body_lin_vel_w,
                "w": c.motion.body_ang_vel_w,
                "arm_columns": adapter.bridge.reference_arm_columns,
                "arm_down": adapter.bridge.reference_down,
            },
            rewards,
            terms,
            adapter.context_schema,
            e.step_dt,
            e.max_episode_length,
            tt.RETHROW_ENABLED,
            tt.RETHROW_GRACE_STEPS,
            bool(getattr(e.cfg.events, "v38_cradle_perturb", None)),
            tt._cradled,
        )
        obj.motion_fps = float(c.motion.fps)
        return obj

    def specification(self):
        def desc(t):
            proxy = t.function.__name__ == "undesired_contacts" and t.weight != 0
            return {
                "name": t.name,
                "function": t.function.__module__ + "." + t.function.__name__,
                "params": primitive(t.params),
                "weight": t.weight,
                "timeout": t.timeout,
                "kind": "configured_zero"
                if t.weight == 0
                else "ground_height_proxy_no_force_or_self_contact"
                if proxy
                else "original_torch_formula",
            }

        return {
            "schema": self.schema_version,
            "history": list(HISTORY_FIELDS),
            "dt": self.dt,
            "body_names": self.body_names,
            "tracked_names": self.tracked_names,
            "rewards": [desc(t) for t in self.rewards],
            "terminations": [desc(t) for t in self.terminations],
            "event_handling": "truncate at known scheduled event, rethrow, possible cradle kick or reference end",
            "bootstrap": "timeouts/model truncations bootstrap last pre-event state; external jump is not simulated",
        }

    def _com(self, b):
        values = b["context"][..., self.context_slices["robot.coms"]]
        return values.reshape(*values.shape[:-1], len(self.body_names), 7)[..., :3]

    @property
    def termination_feature_dim(self):
        return 6 + 3 * len(self.tracked_names)

    def _body_data(self, state, b, fk_result=None):
        com = self._com(b)
        root_i = self.body_names.index("pelvis")
        p, q, v, w = self.fk.velocities(state, com[..., root_i, :], fk_result)
        bp = torch.stack([p[n] for n in self.body_names], -2)
        bq = torch.stack([q[n] for n in self.body_names], -2)
        bw = torch.stack([w[n] for n in self.body_names], -2)
        bv = torch.stack([v[n] for n in self.body_names], -2) + torch.cross(
            bw, apply(bq, com), dim=-1
        )
        g = torch.tensor([0.0, 0.0, -1.0], device=state.device).expand_as(
            state[..., :3]
        )
        return NS(
            root_pos_w=state[..., :3],
            root_quat_w=state[..., 3:7],
            root_lin_vel_w=state[..., 36:39],
            root_ang_vel_w=state[..., 39:42],
            root_lin_vel_b=apply(inverse(state[..., 3:7]), state[..., 36:39]),
            root_ang_vel_b=apply(inverse(state[..., 3:7]), state[..., 39:42]),
            body_pos_w=bp,
            body_quat_w=bq,
            body_lin_vel_w=bv,
            body_ang_vel_w=bw,
            joint_pos=state[..., 7:36],
            joint_vel=state[..., 42:71],
            joint_names=self.joint_names,
            body_names=self.body_names,
            default_joint_pos=b["default_q"],
            default_joint_vel=b["default_dq"],
            joint_pos_limits=b["hard_joint_limits"],
            joint_limits=b["hard_joint_limits"],
            soft_joint_pos_limits=b["soft_joint_limits"],
            GRAVITY_VEC_W=g,
            projected_gravity_b=apply(inverse(state[..., 3:7]), g),
        )

    def _view(
        self,
        b,
        before,
        after,
        action,
        previous,
        phase,
        history,
        episode_steps,
        step,
        fk_before=None,
        fk_after=None,
        initial_step=None,
    ):
        n = len(after)
        device = after.device
        data = self._body_data(after, b, fk_after)
        old_p, old_q = self.fk(before) if fk_before is None else fk_before
        robot = Asset(data, self.body_names, self.joint_names)
        box = Asset(
            NS(
                root_pos_w=after[:, 71:74],
                root_quat_w=after[:, 74:78],
                root_lin_vel_w=after[:, 78:81],
                root_ang_vel_w=after[:, 81:84],
            ),
            ["box"],
            [],
        )
        scene = Scene(robot=robot, balloon=box)
        scene.env_origins = torch.zeros(n, 3, device=device)
        scene.sensors = {}
        zero = torch.zeros_like(b["yaw"])
        yawq = torch.stack(
            (torch.cos(b["yaw"] / 2), zero, zero, torch.sin(b["yaw"] / 2)), -1
        )
        yq = yawq[:, None, :].expand(n, len(self.tracked_names), 4)
        ref_p = apply(yq, self.ref_p[phase])
        ref_q = mul(yq, self.ref_quat[phase])
        anchor_idx = self.tracked_names.index(self.anchor_name)
        robot_anchor = self.body_names.index(self.anchor_name)
        goal_p, goal_q = ref_p[:, anchor_idx], ref_q[:, anchor_idx]
        # MotionCommand caches the relative body target at the PRE-step anchor.
        alignment = heading(mul(old_q[self.anchor_name], inverse(goal_q)))
        aligned_p = old_p[self.anchor_name].clone()
        aligned_p[:, 2] = goal_p[:, 2]
        aq = alignment[:, None, :].expand_as(ref_q)
        relative_p = aligned_p[:, None, :] + apply(aq, ref_p - goal_p[:, None, :])
        relative_q = mul(aq, ref_q)
        first = step == 0 if initial_step is None else initial_step
        if first and "reference_relative_p" in b:
            # Manager caches may precede reset writes or interval events. Preserve
            # the actual initial cache, then derive subsequent cache updates.
            relative_p = b["reference_relative_p"]
            relative_q = b["reference_relative_q"]
        qref = self.ref_q[phase].clone()
        idx = self.ref_arm_columns
        qref[:, idx] = torch.where(
            b["launched"][:, None].bool(), qref[:, idx], self.ref_arm_down
        )
        tracked = [self.body_names.index(n) for n in self.tracked_names]
        cmd = NS(
            cfg=NS(body_names=self.tracked_names),
            joint_pos=qref,
            joint_vel=self.ref_dq[phase],
            anchor_pos_w=goal_p,
            anchor_quat_w=goal_q,
            body_pos_w=ref_p,
            body_quat_w=ref_q,
            body_pos_relative_w=relative_p,
            body_quat_relative_w=relative_q,
            body_lin_vel_w=apply(yq, self.ref_v[phase]),
            body_ang_vel_w=apply(yq, self.ref_w[phase]),
            robot_body_pos_w=data.body_pos_w[:, tracked],
            robot_body_quat_w=data.body_quat_w[:, tracked],
            robot_body_lin_vel_w=data.body_lin_vel_w[:, tracked],
            robot_body_ang_vel_w=data.body_ang_vel_w[:, tracked],
            robot_anchor_pos_w=data.body_pos_w[:, robot_anchor],
            robot_anchor_quat_w=data.body_quat_w[:, robot_anchor],
            motion=NS(
                joint_pos=self.ref_q,
                joint_vel=self.ref_dq,
                fps=getattr(self, "motion_fps", 50.0),
            ),
        )
        term = NS(
            raw_actions=action,
            _offset=b["offset"],
            _scale=b["scale"],
            _joint_ids=slice(None),
        )
        view = NS(
            scene=scene,
            num_envs=n,
            device=device,
            step_dt=self.dt,
            physics_dt=self.dt / 4,
            common_step_counter=step,
            episode_length_buf=episode_steps,
            max_episode_length=self.max_steps,
            command_manager=NS(get_term=lambda name: cmd, _terms={"motion": cmd}),
            action_manager=NS(
                action=action, prev_action=previous, get_term=lambda name: term
            ),
            termination_manager=NS(
                terminated=torch.zeros(n, dtype=torch.bool, device=device),
                time_outs=episode_steps >= self.max_steps,
            ),
            _box_launched=b["launched"].bool(),
            _box_half_extents=b["dimensions"] / 2,
            _balloon_radius_per_env=b["box_radius"],
            _box_launch_step=b["launch_step"].long(),
        )
        for i, name in enumerate(HISTORY_FIELDS):
            setattr(view, name, history[:, i].round().long())
        for name, value in self._static_cache.items():
            setattr(view, name, value)
        return view

    def evaluate_step(
        self,
        b,
        before,
        after,
        action,
        previous,
        phase,
        history,
        episode_steps,
        event_seconds,
        step=0,
        fk_before=None,
        fk_after=None,
        initial_step=None,
    ):
        required = (
            "task_history",
            "hard_joint_limits",
            "soft_joint_limits",
            "box_radius",
            "launch_step",
            "next_external_event_seconds",
        )
        if any(k not in b for k in required):
            raise ValueError(
                "Task v2 context missing; v1 replay is valid for dynamics fitting but not full imagination resume"
            )
        episode_next = episode_steps + 1
        view = self._view(
            b,
            before,
            after,
            action,
            previous,
            phase,
            history,
            episode_next,
            step,
            fk_before,
            fk_after,
            initial_step,
        )
        cmd = view.command_manager.get_term("motion")
        gravity = view.scene["robot"].data.GRAVITY_VEC_W
        term_features = torch.cat(
            (
                cmd.anchor_pos_w - cmd.robot_anchor_pos_w,
                apply(inverse(cmd.anchor_quat_w), gravity)
                - apply(inverse(cmd.robot_anchor_quat_w), gravity),
                (cmd.body_pos_relative_w - cmd.robot_body_pos_w).flatten(-2),
            ),
            -1,
        )
        term = torch.zeros(len(after), dtype=torch.bool, device=after.device)
        timeout = episode_next >= self.max_steps
        for spec in self.terminations:
            value = spec.function(view, **spec.params).bool()
            if spec.timeout:
                timeout |= value
            else:
                term |= value
        view.termination_manager.terminated = term
        view.termination_manager.time_outs = timeout
        parts = []
        for spec in self.rewards:
            if spec.weight == 0:
                value = torch.zeros(len(after), device=after.device)
            elif spec.function.__name__ == "undesired_contacts":
                # Explicit fallback only for an active sensor-only penalty.
                ids = spec.params["sensor_cfg"].body_ids
                value = torch.sigmoid(
                    (0.03 - view.scene["robot"].data.body_pos_w[:, ids, 2]) / 0.01
                ).sum(-1)
            else:
                try:
                    value = spec.function(view, **spec.params)
                except (AttributeError, KeyError) as error:
                    raise RuntimeError(
                        f"Unimplemented analytic dependency in reward {spec.name}: {error}"
                    ) from error
            parts.append(value * spec.weight * self.dt)
        for name in ("_box_posture_ids", "_chest_v5_body_ids", "_v38_hang_cache"):
            if hasattr(view, name):
                self._static_cache[name] = getattr(view, name)
        # These two counters are interval-event history, after reward evaluation.
        z = after[:, 73]
        speed = after[:, 78:81].norm(dim=-1)
        bad = b["launched"].bool() & (
            (z < 0.2)
            | (z > 30)
            | (speed > 30)
            | ~torch.isfinite(z)
            | ~torch.isfinite(speed)
        )
        view._box_regrace = torch.where(bad, view._box_regrace + 1, 0)
        cr = (
            self.cradled(view, view.scene["balloon"])
            if self.cradled is not None
            else torch.zeros_like(term)
        )
        view._v38_cradle_ticks = torch.where(cr, view._v38_cradle_ticks + 1, 0)
        hist = torch.stack([getattr(view, name).float() for name in HISTORY_FIELDS], -1)
        phase_next = phase + 1
        event_next = event_seconds - self.dt
        trunc = timeout | (phase_next >= len(self.ref_q)) | (event_next <= 0)
        if self.rethrow_enabled:
            common = b["common_step"] + step + 1
            trunc |= (view._box_regrace >= self.rethrow_grace) & (common % 5 == 0)
        if self.perturb_enabled:
            trunc |= view._v38_cradle_ticks >= 75
        return TaskResult(
            torch.stack(parts, -1),
            hist,
            term,
            trunc,
            timeout,
            phase_next.clamp_max(len(self.ref_q) - 1),
            episode_next,
            event_next,
            term_features,
        )
