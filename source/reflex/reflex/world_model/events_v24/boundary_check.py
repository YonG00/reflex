"""Read-only terminal-transition regression on a loaded real replay."""

import json
from dataclasses import replace
from pathlib import Path

import torch

from ..diagnostics import eager_diagnostic
from ..joint import returns
from ..trainer import freeze


def check(t, output):
    if not t.cfg.hard_imagination_boundaries:
        raise ValueError("Boundary regression requires explicit repaired configuration")
    rng = t.event_scheduler.rng.clone()
    counts = (t.actor_updates, t.model_updates)
    original = t.task.evaluate_step
    device_ids = (
        [torch.device(t.device).index or 0] if str(t.device).startswith("cuda") else []
    )
    try:
        with (
            torch.random.fork_rng(devices=device_ids),
            eager_diagnostic(t),
            freeze(t.model, t.termination, t.critic, t.prior._disc),
        ):
            b = t.sample(starts=True, n=16)
            real = t.imagine(b)
            result = {
                "natural_valid_fraction": float(real["valid"].float().mean()),
                "natural_terminal_samples": int(real["done"].eq(1).sum()),
            }
            del real

            def terminal(*args, **kwargs):
                value = original(*args, **kwargs)
                return replace(value, terminated=torch.ones_like(value.terminated))

            t.task.evaluate_step = terminal
            imagined = t.imagine(b)
            assert imagined["valid"][0].all()
            assert not imagined["valid"][1:].any()
            assert imagined["done"][0].eq(1).all()
            value = returns(imagined, t.critic, t.cfg.gamma)
            torch.testing.assert_close(value, imagined["reward"][0])
            reward = imagined["reward"].clone()
            reward[1:] = 10000
            other = returns({**imagined, "reward": reward}, t.critic, t.cfg.gamma)
            torch.testing.assert_close(value, other)
            grads = torch.autograd.grad(
                -value.mean(), tuple(t.actor.parameters()), allow_unused=True
            )
            norm = torch.stack([g.norm() for g in grads if g is not None]).norm()
            assert torch.isfinite(norm)
            result.update(
                {
                    "passed": True,
                    "terminal_reward_max_error": float(
                        (value - imagined["reward"][0]).abs().max()
                    ),
                    "future_reward_influence": float((value - other).abs().max()),
                    "actor_gradient_norm": float(norm),
                    "batch": 16,
                    "horizon": t.cfg.horizon,
                    "optimizer_updates": 0,
                }
            )
        assert counts == (t.actor_updates, t.model_updates)
        Path(output).mkdir(parents=True, exist_ok=True)
        (Path(output) / "boundary_check.json").write_text(json.dumps(result, indent=2))
        return result
    finally:
        t.task.evaluate_step = original
        t.event_scheduler.rng = rng


def check_soft(t, output):
    """Analytic terminal flags must not alter the pure-soft rollout."""
    if not t.cfg.pure_soft_survival or t.cfg.hard_imagination_boundaries:
        raise ValueError("Explicit pure-soft configuration required")
    rng = t.event_scheduler.rng.clone()
    original = t.task.evaluate_step
    counts = (t.actor_updates, t.model_updates)
    devices = (
        [torch.device(t.device).index or 0] if str(t.device).startswith("cuda") else []
    )
    try:
        with (
            torch.random.fork_rng(devices=devices),
            eager_diagnostic(t),
            freeze(t.model, t.termination, t.critic, t.prior._disc),
        ):
            b = t.sample(starts=True, n=16)
            initial_rng = t.event_scheduler.rng.clone()
            natural = t.imagine(b)

            def force_terminal(*args, **kwargs):
                value = original(*args, **kwargs)
                return replace(value, terminated=torch.ones_like(value.terminated))

            t.task.evaluate_step = force_terminal
            t.event_scheduler.rng = initial_rng
            forced = t.imagine(b)
            errors = {}
            for key in (
                "reward",
                "done",
                "obs",
                "valid",
                "predicted_launch_probability",
            ):
                torch.testing.assert_close(forced[key], natural[key])
                errors[key] = float(
                    (forced[key].float() - natural[key].float()).abs().max()
                )
            value = returns(natural, t.critic, t.cfg.gamma)
            grads = torch.autograd.grad(
                -value.mean(), tuple(t.actor.parameters()), allow_unused=True
            )
            norm = torch.stack([g.norm() for g in grads if g is not None]).norm()
            assert torch.isfinite(norm)
            result = {
                "passed": True,
                "analytic_terminal_flag_errors": errors,
                "actor_gradient_norm": float(norm),
                "original_terminal_cost": t.terminal_reward_cost,
                "optimizer_updates": 0,
                "batch": 16,
                "horizon": t.cfg.horizon,
            }
        assert counts == (t.actor_updates, t.model_updates)
        Path(output).mkdir(parents=True, exist_ok=True)
        (Path(output) / "soft_survival_check.json").write_text(
            json.dumps(result, indent=2)
        )
        return result
    finally:
        t.task.evaluate_step = original
        t.event_scheduler.rng = rng
