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
            mb = b["blend"]
            rr = torch.zeros(e.num_envs, device=e.device)
            mr = rr.clone()
            valid = b["launched"].bool() & (
                b["phase"] + horizon < b["reference_length"]
            )
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
    return {
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
