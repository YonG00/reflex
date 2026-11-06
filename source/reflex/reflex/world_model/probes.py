"""Bounded real-simulator counterfactual ranking probe.

Restores exposed state/control/event tensors between branches. PhysX contact
warm-start caches cannot be serialized through Isaac Lab: the report states
this limitation and includes a repeated-action determinism check.
"""

import copy
import random
import numpy as np
import torch
from .physics import amp_features


def tensor_tree(value):
    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, (list, tuple)) and all(torch.is_tensor(v) for v in value):
        return type(value)(v.clone() for v in value)
    if (
        isinstance(value, dict)
        and value
        and all(torch.is_tensor(v) for v in value.values())
    ):
        return {k: v.clone() for k, v in value.items()}
    return None


@torch.no_grad()
def rank_actions(
    adapter, trainer, horizon=4, min_pairs=16, min_agreement=0.6, repeat_tolerance=0.02
):
    if trainer.cfg.condition_external_events and horizon != 1:
        raise ValueError("Conditioned V24 ranking currently supports physical H1 only")
    e = adapter.env
    sensors = list(e.scene.sensors.values())
    objects = (
        e,
        e.action_manager,
        e.event_manager,
        e.command_manager,
        adapter.command,
        adapter.term,
        *sensors,
        *(sensor._data for sensor in sensors if hasattr(sensor, "_data")),
    )
    saved = []
    for obj in objects:
        saved.append(
            {k: t for k, v in vars(obj).items() if (t := tensor_tree(v)) is not None}
        )
    blend = adapter.term._v38_ts_cache[2].clone()
    initial = adapter.snapshot()
    state = initial["state"].clone()
    rng = (
        torch.get_rng_state(),
        torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
        random.getstate(),
        np.random.get_state(),
    )
    scalar = (e.common_step_counter, getattr(e, "_sim_step_counter", None))
    origins = e.scene.env_origins

    def restore():
        for obj, values in zip(objects, saved):
            for k, v in values.items():
                setattr(obj, k, copy.deepcopy(v))
        adapter.term._v38_ts_cache[2].copy_(blend)
        e.common_step_counter = scalar[0]
        if scalar[1] is not None:
            e._sim_step_counter = scalar[1]
        for asset, p, q, v, w in (
            (
                adapter.robot,
                state[:, :3],
                state[:, 3:7],
                state[:, 36:39],
                state[:, 39:42],
            ),
            (
                adapter.box,
                state[:, 71:74],
                state[:, 74:78],
                state[:, 78:81],
                state[:, 81:84],
            ),
        ):
            asset.write_root_pose_to_sim(torch.cat((p + origins, q), -1))
            asset.write_root_velocity_to_sim(torch.cat((v, w), -1))
        adapter.robot.write_joint_state_to_sim(state[:, 7:36], state[:, 42:71])
        e.sim.forward()
        e.scene.update(e.physics_dt)
        torch.set_rng_state(rng[0])
        random.setstate(rng[2])
        np.random.set_state(rng[3])
        if rng[1] is not None:
            torch.cuda.set_rng_state_all(rng[1])
        adapter.events.zero_()

    raw = trainer.act(initial)
    perturb = 0.15 * torch.randn_like(raw)
    candidates = (raw + perturb, raw - perturb)
    real = []
    predicted = []
    masks = []
    real_height, predicted_height = [], []
    reset_idx = e._reset_idx
    # Terminal branches are excluded. Avoid their auto-resets changing random
    # masses/materials or curriculum while evaluating alternate actions.
    e._reset_idx = lambda ids: None
    try:
        for candidate in (candidates[0], candidates[0], candidates[1]):
            restore()
            b = adapter.snapshot()
            s, m = b["state"], b["memory"]
            previous = b["previous_action"]
            history = b.get("task_history")
            episode = b.get("episode_steps")
            event = b.get("next_external_event_seconds")
            phase = b["phase"]
            mb = b["blend"]
            rr = torch.zeros(e.num_envs, device=e.device)
            mr = rr.clone()
            valid = b["phase"] + horizon < b["reference_length"]
            if not trainer.cfg.condition_external_events:
                valid &= b["launched"].bool()
            for t in range(horizon):
                rb = adapter.snapshot()
                target, nb = trainer.targets(rb, candidate)
                tr = adapter.step(candidate, target, nb)
                # Repeated and rejected branches still consume real physics.
                trainer.real_transitions += e.num_envs
                trainer.probe_transitions += e.num_envs
                style = trainer.prior.reward(tr["amp"], tr["next_amp"])
                rr += trainer.cfg.gamma**t * (
                    trainer.prior._w_task * tr["reward_terms"].sum(-1)
                    + trainer.prior._w_style * trainer.prior.ramp * style
                )
                valid &= (
                    tr["valid"] & ~tr["boundary"] & ~tr["terminated"] & ~tr["timeout"]
                )
                mt, mb = trainer.targets(b, candidate, mb)
                if trainer.model.state_only:
                    if trainer.task is None:
                        raise RuntimeError(
                            "Ranking pure dynamics requires AnalyticTask"
                        )
                    if trainer.cfg.condition_external_events:
                        pre = trainer.event_scheduler.codec.pre_features(
                            b["event_state"], episode, trainer.task.dt
                        )
                        ns = trainer.model.predict_physical(s, mt, b["context"], pre)
                    else:
                        ns = trainer.model(s, mt, b["context"])
                    predicted_task = trainer.task.evaluate_step(
                        b, s, ns, candidate, previous, phase, history, episode, event, t
                    )
                    reward = predicted_task.rewards
                    if trainer.cfg.pure_soft_survival:
                        from .events_v24.soft_survival import expected_terminal_reward

                        probability = trainer.termination(
                            predicted_task.termination_features
                        ).sigmoid()
                        reward = expected_terminal_reward(
                            reward,
                            probability,
                            trainer.terminal_reward_index,
                            trainer.terminal_reward_cost,
                        )
                    if horizon == 1:
                        real_height.append(-tr["termination_features"][:, 2].square())
                        predicted_height.append(
                            -predicted_task.termination_features[:, 2].square()
                        )
                    history, episode, event, phase = (
                        predicted_task.history,
                        predicted_task.episode_steps,
                        predicted_task.event_seconds,
                        predicted_task.phase,
                    )
                    valid &= ~predicted_task.terminated & ~predicted_task.truncated
                else:
                    ns, m, reward, _ = trainer.model(
                        s,
                        mt,
                        b["context"],
                        m,
                        b["phase"] + t,
                        mb,
                        trainer.observation(b, s, previous, b["phase"] + t),
                        candidate,
                    )
                ms = trainer.prior.reward(
                    amp_features(s, trainer.bridge.fk),
                    amp_features(ns, trainer.bridge.fk),
                )
                mr += trainer.cfg.gamma**t * (
                    trainer.prior._w_task * reward.sum(-1)
                    + trainer.prior._w_style * trainer.prior.ramp * ms
                )
                s, previous = ns, candidate
            real.append(rr)
            predicted.append(mr)
            masks.append(valid)
    finally:
        e._reset_idx = reset_idx
        restore()
    valid = masks[0] & masks[1] & masks[2]
    repeated = (real[0] - real[1]).abs()
    valid &= repeated <= repeat_tolerance
    rd, md = real[0] - real[2], predicted[0] - predicted[2]
    valid &= (rd.abs() > 1e-5) & torch.isfinite(rd) & torch.isfinite(md)
    n = int(valid.sum())
    agreement = float(((rd[valid] * md[valid]) > 0).float().mean()) if n else 0.0
    height_result = {}
    if len(real_height) == 3:
        hd = real_height[0] - real_height[2]
        hp = predicted_height[0] - predicted_height[2]
        repeat_height = (real_height[0] - real_height[1]).abs()
        hm = (
            masks[0]
            & masks[1]
            & masks[2]
            & (hd.abs() > 1e-8)
            & (repeat_height <= 0.25 * hd.abs())
            & torch.isfinite(hp)
        )
        hn = int(hm.sum())
        height_result = {
            "height_valid_pairs": hn,
            "height_action_agreement": float(((hd[hm] * hp[hm]) > 0).float().mean())
            if hn
            else None,
            "height_repeat_max_error": float(repeat_height.max()),
            "height_difference_abs_mean": float(hd.abs().mean()),
            "height_repeat_filter": "repeat difference <=25% of candidate difference",
        }
    return {
        **height_result,
        "passed": n >= min_pairs and agreement >= min_agreement,
        "valid_pairs": n,
        "agreement": agreement,
        "horizon": horizon,
        "repeat_max_error": float(repeated.max()),
        "repeat_tolerance": repeat_tolerance,
        "min_pairs": min_pairs,
        "min_agreement": min_agreement,
        "limitation": "Exposed state restored; PhysX contact warm-start cache is not restorable. Repeated branch checked.",
    }
