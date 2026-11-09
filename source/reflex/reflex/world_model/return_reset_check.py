"""GPU-only integration gate; scratch diagnostic weights, no formal training."""

import json
import time

import torch


def check(trainer, adapter, out, steps=320):
    hook = trainer.return_reset
    if hook is None:
        raise ValueError("Return reset configuration required")
    env = adapter.env
    command = adapter.command
    original_resample = command._resample_command.__func__
    # The configured scale is a conservative positive-weight envelope. Confirm
    # it against the live reward manager, never silently normalize with another task.
    scale = env.step_dt * sum(
        max(0.0, env.reward_manager.get_term_cfg(n).weight)
        for n in env.reward_manager.active_terms
    )
    if abs(scale - trainer.cfg.return_reset_max_step_reward) > 1e-6:
        raise ValueError(f"Reward scale/config mismatch: live={scale}")
    started = time.monotonic()
    original_steps = trainer.cfg.collect_steps
    trainer.cfg.collect_steps = 1
    counts = {
        "terminated": 0,
        "timeout": 0,
        "reference_jump": 0,
        "samples": 0,
    }
    try:
        for _ in range(steps):
            trainer.collect(adapter)
            p = adapter.pre_reset
            counts["terminated"] += int(p["terminated"].sum())
            counts["timeout"] += int(p["timeout"].sum())
            # Inspect the just-added transition, not an arbitrary ring slot.
            ix = (trainer.replay.pointer - 1) % trainer.replay.capacity
            tr = trainer.replay.data
            counts["samples"] += env.num_envs
            if "event_mask" in tr:
                assert torch.isfinite(tr["next_state"][ix]).all()
            assert torch.isfinite(hook.sampler.returns).all()
            assert torch.isfinite(hook.sampler.past).all()
            assert (
                (command.time_steps >= 0)
                & (command.time_steps < command.motion.time_step_total)
            ).all()
        # Force a real native episode reset without replacing its event sequence.
        ids = torch.arange(min(8, env.num_envs), device=env.device)
        env._reset_idx(ids)
        assert (hook.lengths[ids] == 0).all()
        assert (hook.total[ids] == 0).all()
        assert (hook.discounted[ids] == 0).all()
        counts["explicit_native_reset"] = len(ids)
        # Exercise reference exhaustion through the original command compute path.
        ids = torch.arange(min(8, env.num_envs), device=env.device)
        command.time_steps[ids] = command.motion.time_step_total - 1
        trainer.collect(adapter)
        assert (command.time_steps[ids] < command.motion.time_step_total).all()
        assert (hook.lengths[ids] == 0).all()
        counts["reference_jump"] = len(ids)
        assert command._resample_command.__func__ is original_resample
        # Save/load the actual reset weighting and verify transactional persistence.
        state = hook.state_dict()
        hook.sampler.returns.zero_()
        hook.load_state_dict(state)
        torch.testing.assert_close(
            hook.sampler.returns, state["returns"], rtol=0, atol=0
        )
        # WM and Actor smoke updates on the collected real Replay. Diagnostic
        # warmup progress is explicit; this does not alter production500-round gate.
        trainer.fit_normalizers()
        from .events_v24.probe import check as event_check

        native_events = event_check(adapter, trainer)
        assert native_events["passed"]
        (out / "return_reset_events.json").write_text(
            json.dumps(native_events, indent=2)
        )
        trainer.wm_rounds_completed = trainer.cfg.wm_only_rounds
        model = trainer.fit_model()
        before = [p.detach().clone() for p in trainer.actor.parameters()]
        actor = trainer.update_actor()
        changed = any(
            not torch.equal(a, b) for a, b in zip(before, trainer.actor.parameters())
        )
        assert changed
        result = {
            "completed": True,
            "passed": counts["terminated"] > 0,
            "reward_scale": scale,
            "counts": counts,
            "actor_parameters_changed": changed,
            "original_resample_preserved": True,
            "seconds": time.monotonic() - started,
            "wm_metrics": model,
            "actor_metrics": actor,
            "native_events_passed": native_events["passed"],
            "scope": "scratch reset/event/gradient smoke; no long training",
        }
        (out / "return_reset_check.json").write_text(json.dumps(result, indent=2))
        return result
    finally:
        trainer.cfg.collect_steps = original_steps
