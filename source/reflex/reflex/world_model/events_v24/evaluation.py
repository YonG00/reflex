"""Separate teacher-event conditioning from causal scheduler-driven forecasts."""

import torch
from ..physics import state_error
from ..diagnostics import calibration


@torch.no_grad()
def evaluate(trainer, batch=64):
    t = trainer
    scheduler = t.event_scheduler
    saved_rng = scheduler.rng.clone()
    metrics = {}
    try:
        for h in (1, 8, 16, 32):
            try:
                seq = t.replay.sample(batch, h, validation=True)
            except ValueError:
                metrics[f"validation_v24/{h}/available"] = 0
                continue
            for protocol in ("conditioned", "free"):
                s = seq["state"][0]
                event = seq["event_state"][0]
                pools = seq["event_pool"][0]
                b = {k: v[0] for k, v in seq.items()}
                physical_errors, post_errors, launch_errors = [], [], []
                phase_errors = {"waiting": [], "launched": [], "launch_write": []}
                for j in range(h):
                    if protocol == "conditioned":
                        event = seq["event_state"][j]
                    episode = seq["episode_steps"][0] + j
                    pre = scheduler.codec.pre_features(event, episode, t.task.dt)
                    physical = t.model.predict_physical(
                        s, seq["target"][j], seq["context"][j], pre
                    )
                    if protocol == "conditioned":
                        targets, mask = seq["event_targets"][j], seq["event_mask"][j]
                    else:
                        bb = {
                            **b,
                            "phase": b["phase"] + j,
                            "launched": scheduler.codec.mode(event),
                        }
                        poses = t.bridge.fk(physical)
                        plan, event, pools = scheduler.step(
                            bb,
                            physical,
                            poses,
                            event,
                            pools,
                            episode + 1,
                            b["common_step"] + j + 1,
                        )
                        targets, mask = plan.targets, plan.mask
                    ns = t.model.predict_post(physical, targets, mask)
                    pe = state_error(physical, seq["physical_next_state"][j]).square()
                    se = state_error(ns, seq["next_state"][j]).square()
                    physical_errors.append(pe)
                    post_errors.append(se)
                    waiting = ~seq["launched"][j].bool()
                    launch = (
                        waiting
                        & seq["after_launched"][j].bool()
                        & seq["event_mask"][j, :, 78:81].any(-1)
                        & (seq["event_targets"][j, :, 78:81].norm(dim=-1) > 1)
                    )
                    for name, selected in [
                        ("waiting", waiting),
                        ("launched", ~waiting),
                        ("launch_write", launch),
                    ]:
                        if selected.any():
                            phase_errors[name].append(se[selected])
                    if launch.any():
                        launch_errors.append(
                            (
                                ns[launch, 78:81] - seq["next_state"][j, launch, 78:81]
                            ).square()
                        )
                    if h == 1 and protocol == "conditioned":
                        features = t.task.predicted_termination_features(
                            b, s, physical, b["phase"]
                        )
                        metrics.update(
                            calibration(t.termination(features), seq["terminated"][j])
                        )
                    s = ns
                prefix = f"validation_v24/{protocol}/{h}"
                metrics[prefix + "/physical_rmse"] = float(
                    torch.stack(physical_errors).mean().sqrt()
                )
                metrics[prefix + "/post_control_rmse"] = float(
                    torch.stack(post_errors).mean().sqrt()
                )
                for name, values in phase_errors.items():
                    metrics[prefix + "/" + name + "_samples"] = sum(
                        len(x) for x in values
                    )
                    if values:
                        metrics[prefix + "/" + name + "_rmse"] = float(
                            torch.cat(values).mean().sqrt()
                        )
                if launch_errors:
                    metrics[prefix + "/launch_velocity_rmse_mps"] = float(
                        torch.cat(launch_errors).mean().sqrt()
                    )
    finally:
        scheduler.rng = saved_rng
    return metrics
