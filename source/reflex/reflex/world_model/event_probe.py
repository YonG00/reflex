"""Controlled Isaac state-write boundary tests; no policy/model updates."""

import torch
from .replay import Replay
from .oracle_env import box_dimensions


def check(adapter, trainer, steps=12, trace_path=None):
    e, box = adapter.env, adapter.box
    if e.num_envs < 8:
        raise ValueError("Event probe needs at least eight environments")
    n, device = e.num_envs, adapter.device
    macro = adapter.learn_external_events
    # These are reset-fixture events, unrelated to the controlled throw cases.
    # Leaving pending hold-init enabled cancels the waiting throw on step one.
    for name in ("_hold_init_pending", "_hold_init_active"):
        value = getattr(e, name, None)
        if torch.is_tensor(value):
            value.zero_()
    pos = box.data.root_pos_w.clone()
    pos[:, :2] = e.scene.env_origins[:, :2] + pos.new_tensor((3.0, 0.0))
    pos[:, 2] = e.scene.env_origins[:, 2] + 2.0
    pos[3, 2] = e.scene.env_origins[3, 2] + box_dimensions(e)[3, 2] / 2 + 0.001
    quat = torch.zeros(n, 4, device=device)
    quat[:, 0] = 1
    vel = torch.zeros(n, 6, device=device)
    vel[3, 0] = 0.5  # Ground collision/friction is physical, not a script write.
    vel[6, 0] = 2.0  # Continuous damped flight.
    e._box_p0.copy_(pos)
    e._box_q0.copy_(quat)
    e._box_launched.fill_(True)
    e._box_launched[:2] = False
    e._box_launch_step.copy_(e.episode_length_buf + 100)
    e._box_launch_step[0] = 1000000
    e._box_launch_step[1] = e.episode_length_buf[1] + 1
    if hasattr(e, "_box_lib_mask"):
        e._box_lib_mask.zero_()
    box.write_root_pose_to_sim(torch.cat((pos, quat), -1))
    box.write_root_velocity_to_sim(vel)
    records = []
    post_states = []
    traces = []
    dispatch = getattr(adapter, "step_environment", e.step)
    tick = 0

    def controlled_step(raw):
        nonlocal tick
        result = dispatch(raw)
        if tick == 2:
            # Real velocity API jump, with launched remaining TRUE throughout.
            ids = torch.tensor([4], device=device)
            box.write_root_velocity_to_sim(
                torch.tensor([[4.0, 0.0, 0.0, 0.0, 0.0, 0.0]], device=device),
                env_ids=ids,
            )
            # A no-op write must not mark unrelated flight as discontinuous.
            ids = torch.tensor([5], device=device)
            same = torch.cat(
                (box.data.root_lin_vel_w[ids], box.data.root_ang_vel_w[ids]), -1
            ).clone()
            box.write_root_velocity_to_sim(same, env_ids=ids)
        tick += 1
        return result

    adapter.step_environment = controlled_step
    before_updates = (
        trainer.model_updates,
        trainer.actor_updates,
        trainer.termination_updates,
    )
    replay = Replay(steps, device)
    try:
        for j in range(steps):
            b = adapter.snapshot()
            raw = torch.zeros_like(b["previous_action"])
            target, blend = trainer.targets(b, raw)
            tr = adapter.step(raw, target, blend)
            physical = tr["physical_next_state"] if macro else tr["next_state"]
            torch.testing.assert_close(
                physical, adapter.pre_reset["state"], rtol=0, atol=0
            )
            post = adapter.physical().clone()
            records.append(tr)
            post_states.append(post)
            traces.append(
                {
                    "step": j,
                    "physics_box_velocity": physical[:, 78:81].cpu().tolist(),
                    "post_control_box_velocity": post[:, 78:81].cpu().tolist(),
                    "physics_box_position": physical[:, 71:74].cpu().tolist(),
                    "post_control_box_position": post[:, 71:74].cpu().tolist(),
                    "external_write": adapter.events.cpu().tolist(),
                    "boundary": tr["boundary"].cpu().tolist(),
                    "launch_event": tr["launch_event"].cpu().tolist(),
                    "valid": tr["valid"].cpu().tolist(),
                    "timeout": tr["timeout"].cpu().tolist(),
                    "episode_steps": tr["episode_steps"].cpu().tolist(),
                    "launch_step": tr["launch_step"].cpu().tolist(),
                    "terminated": tr["terminated"].cpu().tolist(),
                    "before_launched": tr["launched"].cpu().tolist(),
                    "after_launched": tr["after_launched"].cpu().tolist(),
                }
            )
            replay.add(tr)
    finally:
        adapter.step_environment = dispatch
    if trace_path is not None:
        import json
        from pathlib import Path

        Path(trace_path).write_text(json.dumps(traces, indent=2))
    assert before_updates == (
        trainer.model_updates,
        trainer.actor_updates,
        trainer.termination_updates,
    )
    launch = records[0]

    def physics(r):
        return r["physical_next_state"] if macro else r["next_state"]

    assert bool(launch["launch_event"][1])
    assert bool(launch["boundary"][1]) == (not macro)
    assert (post_states[0][1, 78:81] - physics(launch)[1, 78:81]).norm() > 0.5
    torch.testing.assert_close(records[1]["state"][1], post_states[0][1])
    assert records[1]["valid"][1]  # Post-launch state is kept.
    assert bool(records[2]["launched"][4]) and not bool(records[2]["launch_event"][4])
    assert bool(records[2]["boundary"][4]) == (not macro)
    # Actual API write, independent of launched; macro training keeps it.
    torch.testing.assert_close(
        post_states[2][4, 78:81], vel.new_tensor((4.0, 0.0, 0.0))
    )
    assert not bool(records[2]["boundary"][5])  # No-op velocity write stays continuous.
    assert all(
        bool(r["valid"][0]) and bool(r["boundary"][0]) == (not macro) for r in records
    )
    if macro:
        assert all(bool(r["external_boundary"][0]) for r in records)
        for r, post in zip(records, post_states):
            torch.testing.assert_close(r["next_state"], post)
        assert records[2]["external_boundary"][4]
    assert any(
        (p[0, 71:84] - physics(r)[0, 71:84]).abs().max() > 1e-5
        for r, p in zip(records, post_states)
    )
    for j in range(steps - 1):
        continuous = ~records[j]["boundary"]
        torch.testing.assert_close(
            records[j + 1]["state"][continuous],
            records[j]["next_state"][continuous],
            atol=2e-5,
            rtol=2e-5,
        )
    windows = replay.indices(8)
    assert bool(((windows[:, 1] == 1) & (windows[:, 0] == 0)).any()) == macro
    assert bool(((windows[:, 1] == 1) & (windows[:, 0] >= 1)).any())
    assert (
        bool(
            (
                (windows[:, 1] == 4) & (windows[:, 0] <= 2) & (windows[:, 0] + 7 >= 3)
            ).any()
        )
        == macro
    )
    assert bool((windows[:, 1] == 3).any())  # Real ground contact is not rejected.
    assert all(bool(r["valid"][3]) for r in records)
    speed3 = [float(r["next_state"][3, 78:81].norm()) for r in records]
    speed6 = [float(r["next_state"][6, 78:81].norm()) for r in records]
    assert records[-1]["next_state"][3, 73] > 0
    assert records[-1]["next_state"][3, 80] > -0.5
    assert abs(float(records[-1]["next_state"][3, 78])) < 0.5
    return {
        "passed": True,
        "learn_external_events": macro,
        "optimizer_updates": 0,
        "steps": steps,
        "ground_contact_speed": speed3,
        "flight_speed": speed6,
        "valid_eight_step_windows": len(windows),
        "transitions": traces,
    }
