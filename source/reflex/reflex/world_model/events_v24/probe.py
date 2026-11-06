"""Native callback equivalence, with the identical pre-event RNG state."""

import torch

from .writes import WritePlan


def check(adapter, trainer, steps=12):
    scheduler = adapter.event_scheduler
    if scheduler is None:
        raise ValueError("V24 probe needs the event scheduler")
    env = adapter.env
    n = env.num_envs
    if not hasattr(env, "_box_regrace"):
        env._box_regrace = torch.zeros(n, dtype=torch.long, device=adapter.device)
    if n < 8:
        raise ValueError("V24 probe needs at least 8 environments")
    manager = env.event_manager
    original = manager.apply
    traces = []
    before = (trainer.model_updates, trainer.actor_updates, trainer.termination_updates)

    def interval_checked(mode, *args, **kwargs):
        if mode != "interval":
            return original(mode, *args, **kwargs)
        physical = adapter.physical()
        clock = scheduler.snapshot(env)
        bb = {
            **adapter._before,
            **clock,
            "phase": adapter.command.time_steps.clone(),
            "task_history": adapter.task_history(),
        }
        episode = env.episode_length_buf.clone()
        common = torch.full_like(episode, env.common_step_counter)
        cpu_rng = torch.get_rng_state()
        cuda_rng = (
            torch.cuda.get_rng_state_all()
            if str(adapter.device).startswith("cuda")
            else None
        )
        parent = adapter.event_plan
        adapter.event_plan = WritePlan(physical)
        result = original(mode, *args, **kwargs)
        actual = adapter.event_plan
        post_control = adapter.physical()
        applied = torch.where(actual.mask, actual.targets, physical)
        torch.testing.assert_close(applied, post_control, atol=2e-4, rtol=2e-5)
        after = scheduler.snapshot(env)["event_state"]
        devices = (
            [torch.device(adapter.device).index or 0] if cuda_rng is not None else []
        )
        with torch.random.fork_rng(devices=devices):
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
            poses = trainer.bridge.fk(physical)
            pools = clock["event_pool"].unsqueeze(0).expand(n, -1, -1).clone()
            planned, event, _ = scheduler.step(
                bb,
                physical,
                poses,
                clock["event_state"],
                pools,
                episode,
                common,
                shared_world=True,
                isolated_rng=False,
            )
        torch.testing.assert_close(planned.mask, actual.mask, rtol=0, atol=0)
        torch.testing.assert_close(
            planned.targets, actual.targets, atol=2e-4, rtol=2e-5
        )
        torch.testing.assert_close(event, after, atol=2e-4, rtol=2e-5)
        traces.append(
            {
                "step": len(traces),
                "max_deterministic_post_error": float(
                    (applied - post_control).abs().max()
                ),
                "max_command_error": float(
                    (planned.targets - actual.targets).abs().max()
                ),
                "max_event_state_error": float((event - after).abs().max()),
                "box_velocity_writes": int(actual.mask[:, 78:84].any(-1).sum()),
                "joint_writes": int(actual.mask[:, 7:36].any(-1).sum()),
                "launched_after": int(scheduler.codec.mode(event).sum()),
            }
        )
        if parent is not None:
            parent.targets = torch.where(actual.mask, actual.targets, parent.targets)
            parent.mask |= actual.mask
        adapter.event_plan = parent
        return result

    manager.apply = interval_checked
    try:
        for j in range(steps):
            # Valid controlled prefixes cover launch, reset-time hold and rethrow.
            if j == 0:
                env._hold_init_pending.zero_()
                env._hold_init_pending[3] = True
                env._box_launched[:3] = torch.tensor(
                    [False, False, True], device=adapter.device
                )
                env._box_launch_step[0] = env.episode_length_buf[0] + 20
                env._box_launch_step[1] = env.episode_length_buf[1] + 1
                env.common_step_counter = 4
            if j % 5 == 0:
                pose = torch.cat(
                    (
                        adapter.box.data.root_pos_w[[2]],
                        adapter.box.data.root_quat_w[[2]],
                    ),
                    -1,
                ).clone()
                pose[:, 2] = env.scene.env_origins[2, 2] + 0.11
                ids = torch.tensor([2], device=adapter.device)
                adapter.box.write_root_pose_to_sim(pose, env_ids=ids)
                adapter.box.write_root_velocity_to_sim(
                    torch.zeros(1, 6, device=adapter.device), env_ids=ids
                )
                env._box_launched[2] = True
                env._box_regrace[2] = 30
            # Force the existing timer to be due, not a new probability/rule.
            for i, term in enumerate(scheduler.terms):
                if term.name == "push_robot" and j == 2:
                    manager._interval_term_time_left[i][:] = 0
                if term.name == "v38_cradle_perturb" and j == 4:
                    manager._interval_term_time_left[i][:] = 0
            b = adapter.snapshot()
            raw = torch.zeros_like(b["previous_action"])
            target, blend = trainer.targets(b, raw)
            adapter.step(raw, target, blend)
    finally:
        manager.apply = original
    assert before == (
        trainer.model_updates,
        trainer.actor_updates,
        trainer.termination_updates,
    )
    return {
        "passed": True,
        "optimizer_updates": 0,
        "steps": len(traces),
        "scheduler": scheduler.specification(),
        "transitions": traces,
    }
