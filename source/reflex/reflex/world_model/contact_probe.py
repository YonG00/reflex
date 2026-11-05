"""Controlled sensor positive/negative checks, never inserted into Replay."""

import torch


@torch.no_grad()
def _check(adapter, trainer, steps=8):
    e = adapter.env
    sensor = e.scene.sensors["wm_box_contacts"]
    result = {
        "replay_insertions": 0,
        "optimizer_updates": 0,
        "cases": {},
        "filters": list(sensor.cfg.filter_prim_paths_expr),
        "matrix_shape": list(sensor.data.force_matrix_w.shape),
        "robot_bodies": list(adapter.robot.body_names),
    }
    for name in ("robot", "ground"):
        e.reset(seed=614)
        e._box_launched.fill_(True)
        pos = e.scene.env_origins.clone()
        if name == "robot":
            pos = adapter.robot.data.body_pos_w[
                :, adapter.robot.body_names.index("torso_link")
            ].clone()
            pos[:, 0] += 0.04
        else:
            pos[:, :2] += 1.5
            pos[:, 2] += 0.5 * adapter.snapshot()["dimensions"][:, 2] * 0.7
        q = torch.zeros(e.num_envs, 4, device=e.device)
        q[:, 0] = 1
        adapter.box.write_root_pose_to_sim(torch.cat((pos, q), -1))
        adapter.box.write_root_velocity_to_sim(
            torch.zeros(e.num_envs, 6, device=e.device)
        )
        seen = torch.zeros(e.num_envs, dtype=torch.bool, device=e.device)
        pair_peak = net_peak = 0.0
        for _ in range(steps):
            b = adapter.snapshot()
            raw = torch.zeros(e.num_envs, 29, device=e.device)
            target, blend = trainer.targets(b, raw)
            tr = adapter.step(raw, target, blend)
            seen |= tr["box_contact"] & tr["contact_known"]
            pair_peak = max(
                pair_peak, float(adapter.pre_reset["pair_force_peak"].max())
            )
            net_peak = max(net_peak, float(adapter.pre_reset["net_force_peak"].max()))
        result["cases"][name] = {
            "detected_envs": int(seen.sum()),
            "envs": e.num_envs,
            "pair_force_peak_n": pair_peak,
            "all_contact_force_peak_n": net_peak,
        }
    result["passed"] = (
        result["cases"]["robot"]["detected_envs"] > 0
        and result["cases"]["ground"]["detected_envs"] == 0
        and result["cases"]["ground"]["all_contact_force_peak_n"] > 1.0
    )
    return result


@torch.no_grad()
def check(adapter, trainer, steps=8):
    # Large artificial penetrations can violate ideal-FK joint constraints. This
    # test concerns sensor filtering, not the reward/FK contract of normal data.
    task = adapter.task
    adapter.task = None
    try:
        result = _check(adapter, trainer, steps)
        result["analytic_task_checks"] = "disabled_only_for_artificial_sensor_probe"
        return result
    finally:
        adapter.task = task
