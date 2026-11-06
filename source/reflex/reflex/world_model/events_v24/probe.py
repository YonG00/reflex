"""Native callback equivalence, with the identical pre-event RNG state."""

import torch

from .writes import WritePlan


def check(adapter, trainer, steps=24):
    scheduler = adapter.event_scheduler
    if scheduler is None:
        raise ValueError("V24 probe needs the event scheduler")
    env = adapter.env
    n = env.num_envs
    if not hasattr(env, "_box_regrace"):
        env._box_regrace = torch.zeros(n, dtype=torch.long, device=adapter.device)
    if not hasattr(env, "_v38_cradle_ticks"):
        env._v38_cradle_ticks = torch.zeros(n, dtype=torch.long, device=adapter.device)
    if n < 8:
        raise ValueError("V24 probe needs at least 8 environments")
    manager = env.event_manager
    original = manager.apply
    traces = []
    perturb_writes = []
    perturb_cfgs = []
    for cfg in manager._mode_term_cfgs["interval"]:
        if cfg.func.__name__ == "v38_cradle_perturb":
            original_perturb = cfg.func

            def measured_perturb(e, *args, **kwargs):
                velocity = adapter.box.data.root_vel_w.clone()
                result = original_perturb(e, *args, **kwargs)
                perturb_writes.append(
                    int((adapter.box.data.root_vel_w - velocity).abs().any(-1).sum())
                )
                return result

            perturb_cfgs.append((cfg, cfg.func))
            cfg.func = measured_perturb
    before = (trainer.model_updates, trainer.actor_updates, trainer.termination_updates)

    def interval_checked(mode, *args, **kwargs):
        if mode != "interval":
            return original(mode, *args, **kwargs)
        # Controlled test prefix at the interval boundary exercises a real
        # eligible perturb branch without changing its probability or callback.
        if len(traces) >= 4 and perturb_cfgs:
            from ...envs.teacher_terms import _chest_w

            ids = torch.arange(4, n, device=adapter.device)
            pose = torch.cat(
                (_chest_w(env)[ids], adapter.box.data.root_quat_w[ids]), -1
            )
            adapter.box.write_root_pose_to_sim(pose, env_ids=ids)
            adapter.box.write_root_velocity_to_sim(
                adapter.robot.data.root_vel_w[ids], env_ids=ids
            )
            env._box_launched[ids] = True
            env._v38_cradle_ticks[ids] = 75
            for i, term in enumerate(scheduler.terms):
                if term.name == "v38_cradle_perturb":
                    manager._interval_term_time_left[i][:] = 0
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
        for cfg, function in perturb_cfgs:
            cfg.func = function
    if perturb_cfgs:
        assert sum(perturb_writes) > 0, "No positive native perturb branch exercised"
    assert before == (
        trainer.model_updates,
        trainer.actor_updates,
        trainer.termination_updates,
    )
    # A causal controlled waiting prefix verifies native re-aim derivatives
    # and an actual Actor H32 launch, without optimizer or simulator updates.
    from ..diagnostics import eager_diagnostic

    b = adapter.snapshot()
    event_values, timers = scheduler.codec.decode(b["event_state"])
    event_values["_box_launched"] = torch.zeros(
        n, dtype=torch.bool, device=adapter.device
    )
    event_values["_box_launch_step"] = b["episode_steps"] + 1
    event_values["_hold_init_pending"] = torch.zeros(
        n, dtype=torch.bool, device=adapter.device
    )
    event_values["_box_lib_mask"] = torch.zeros(
        n, dtype=torch.bool, device=adapter.device
    )
    for i, term in enumerate(scheduler.terms):
        if term.interval[1] > 0:
            timers[:, i] = 100
    event = scheduler.codec.encode(event_values, timers)
    pools = b["event_pool"].unsqueeze(0).expand(n, -1, -1).clone().fill_(100)
    b = {
        **b,
        "launched": event_values["_box_launched"],
        "event_state": event,
        "event_pool": pools,
    }
    physical = b["state"].detach().clone().requires_grad_(True)
    saved_rng = scheduler.rng.clone()
    try:
        plan, _, _ = scheduler.step(
            b,
            physical,
            trainer.bridge.fk(physical),
            event,
            pools,
            b["episode_steps"] + 1,
            b["common_step"] + 1,
        )
        objective = (plan.targets[:, 78] + 0.37 * plan.targets[:, 79]).sum()
        gradient = torch.autograd.grad(objective, physical)[0][:, :3]
        assert torch.isfinite(gradient).all() and gradient.norm() > 0
        with eager_diagnostic(trainer), torch.no_grad():
            imagined = trainer.imagine(start_batch=b)
        modes = imagined["predicted_launch_probability"]
        # imagine returns the mode AFTER each step; include the waiting start
        # so that a launch on the very first step is counted.
        before_modes = torch.cat((b["launched"][None].float(), modes[:-1]), 0)
        launches = int(((modes > 0.5) & (before_modes < 0.5)).sum())
        assert len(modes) == trainer.cfg.horizon and launches > 0
    finally:
        scheduler.rng = saved_rng
    return {
        "passed": True,
        "optimizer_updates": 0,
        "steps": len(traces),
        "positive_perturb_writes": sum(perturb_writes),
        "launch_velocity_robot_position_gradient_norm": float(gradient.norm()),
        "actor_imagination_horizon": len(modes),
        "actor_waiting_to_launch_transitions": launches,
        "scheduler": scheduler.specification(),
        "transitions": traces,
    }
