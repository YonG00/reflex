"""Read-only gradient/calibration diagnostics. Never performs an optimizer step."""

import time
import torch
from .trainer import freeze


@torch.no_grad()
def calibration(logits, labels):
    p, y = logits.sigmoid(), labels.float()
    out = {
        "termination/validation_bce": float(
            torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
        ),
        "termination/brier": float((p - y).square().mean()),
        "termination/predicted_rate": float(p.mean()),
        "termination/true_rate": float(y.mean()),
    }
    ece = torch.zeros((), device=p.device)
    for i in range(10):
        mask = (p >= i / 10) & ((p < (i + 1) / 10) if i < 9 else (p <= 1))
        count = mask.sum()
        out[f"termination/calibration_bin{i}_count"] = int(count)
        if bool(count):
            ece += mask.float().mean() * (p[mask].mean() - y[mask].mean()).abs()
    out["termination/ece"] = float(ece)
    return out


def reward_return(imagined, reward, gamma, detach_survival=False):
    survival = torch.ones_like(reward[0])
    result = torch.zeros_like(survival)
    done = imagined["done"].detach() if detach_survival else imagined["done"]
    for j in range(len(reward)):
        result = result + gamma**j * survival * reward[j]
        survival = survival * (1 - done[j]) * ~imagined["truncated"][j]
    return result


def gradient_audit(t):
    from .joint import returns

    started = time.monotonic()
    devices = (
        [torch.device(t.device).index or 0] if str(t.device).startswith("cuda") else []
    )
    with (
        torch.random.fork_rng(devices=devices),
        freeze(t.model, t.termination, t.prior._disc, t.critic),
    ):
        b = t.sample(starts=True, n=t.cfg.diagnostic_batch)
        imagined = t.imagine(start_batch=b)
        params = list(t.actor.parameters())

        def vector(value):
            gs = (
                torch.autograd.grad(value, params, retain_graph=True, allow_unused=True)
                if value.requires_grad
                else [None] * len(params)
            )
            return torch.cat(
                [
                    (torch.zeros_like(p) if g is None else g).reshape(-1)
                    for p, g in zip(params, gs)
                ]
            )

        scale = t.cfg.horizon if t.cfg.actor_horizon_scale else 1
        total = vector(-returns(imagined, t.critic, t.cfg.gamma).mean() / scale)
        metrics = {"diagnostic/total_gradient_norm": float(total.norm())}
        values = {
            "task": t.prior._w_task * imagined["terms"].sum(-1),
            "amp": t.prior._w_style * t.prior.ramp * imagined["style"],
        }
        values.update(
            {
                f"reward/{n}": t.prior._w_task * imagined["terms"][..., i]
                for i, n in enumerate(t.reward_names)
            }
        )
        for name, r in values.items():
            for mode in ("direct", "with_survival"):
                g = vector(
                    -reward_return(
                        imagined, r, t.cfg.gamma, detach_survival=mode == "direct"
                    ).mean()
                    / scale
                )
                metrics[f"diagnostic/{mode}/{name}/norm"] = float(g.norm())
                metrics[f"diagnostic/{mode}/{name}/projection"] = float(
                    (g * total).sum() / total.square().sum().clamp_min(1e-20)
                )
        reward_grad = vector(
            -reward_return(imagined, imagined["reward"], t.cfg.gamma).mean() / scale
        )
        bootstrap = total - reward_grad
        metrics["diagnostic/bootstrap_gradient_norm"] = float(bootstrap.norm())
        task_grad = vector(
            -reward_return(imagined, values["task"], t.cfg.gamma).mean() / scale
        )
        amp_grad = vector(
            -reward_return(imagined, values["amp"], t.cfg.gamma).mean() / scale
        )
        metrics["diagnostic/additive_gradient_relative_error"] = float(
            (total - task_grad - amp_grad - bootstrap).norm()
            / total.norm().clamp_min(1e-20)
        )
        # Hold both rewards AND critic values fixed, retain hazard derivatives in
        # every survival multiplier, including truncation and horizon bootstrap.
        hazard = vector(
            -returns(
                {**imagined, "reward": imagined["reward"].detach()},
                lambda obs: t.critic(obs).detach(),
                t.cfg.gamma,
            ).mean()
            / scale
        )
        metrics["diagnostic/survival_only_gradient_norm"] = float(hazard.norm())
    metrics["time/gradient_diagnostic_seconds"] = time.monotonic() - started
    return metrics


def diagnostic_due(trainer, elapsed, budget):
    every = trainer.cfg.diagnostic_every
    if not every or trainer.actor_updates % every:
        return False, False
    estimate = getattr(trainer, "last_diagnostic_seconds", 0.0)
    if budget is not None and estimate > 0 and budget - elapsed < estimate + 5.0:
        return False, True
    return True, False


def saved_diagnostic_duration(state, checkpoint_path):
    import math
    import json
    from pathlib import Path

    def valid(value):
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and value > 0
        )

    value = state.get("last_diagnostic_seconds", 0.0)
    if valid(value):
        return float(value)
    try:
        lines = (
            (Path(checkpoint_path).parent / "metrics.jsonl")
            .read_text(errors="replace")
            .splitlines()
        )
    except OSError:
        return 0.0
    for line in reversed(lines):
        try:
            row = json.loads(line)
        except (ValueError, TypeError):
            continue
        value = (
            row.get("time/gradient_diagnostic_seconds")
            if isinstance(row, dict)
            else None
        )
        if valid(value):
            return float(value)
    return 0.0
