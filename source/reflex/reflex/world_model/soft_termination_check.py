"""Paired learned/analytic hazard probe; restore source state, never long train."""

import copy
import json
import random
from pathlib import Path

import numpy as np
import torch

from .analytic_termination import probability
from .events_v24.compare import seed
from .joint import returns
from .timing import timestamp
from .trainer import freeze


def distribution(p):
    p = p.detach().float().reshape(-1)
    if not p.numel():
        return {"samples": 0}
    return {
        "samples": p.numel(),
        "mean": float(p.mean()),
        "min": float(p.min()),
        "max": float(p.max()),
        "exact_zero_fraction": float((p == 0).float().mean()),
        "exact_one_fraction": float((p == 1).float().mean()),
        "below_1e_4_fraction": float((p < 1e-4).float().mean()),
        "above_1_minus_1e_4_fraction": float((p > 1 - 1e-4).float().mean()),
    }


def matrix_reference(z, q, rz, rq):
    # Materialize full matrices, independently of the optimized Rzz helper.
    def matrix(v):
        w, x, y, zz = v.unbind(-1)
        return torch.stack(
            (
                1 - 2 * (y * y + zz * zz),
                2 * (x * y - w * zz),
                2 * (x * zz + w * y),
                2 * (x * y + w * zz),
                1 - 2 * (x * x + zz * zz),
                2 * (y * zz - w * x),
                2 * (x * zz - w * y),
                2 * (y * zz + w * x),
                1 - 2 * (x * x + y * y),
            ),
            -1,
        ).reshape(*v.shape[:-1], 3, 3)

    ez = (rz - z).abs()
    et = (matrix(rq)[..., 2, 2] - matrix(q)[..., 2, 2]).abs()
    return 1 - (1 - torch.sigmoid((ez - 0.25) / 0.02)) * (
        1 - torch.sigmoid((et - 0.8) / 0.04)
    )


def check(t, output, repeats=3):
    if (
        t.termination is None
        or not t.cfg.learned_termination
        or not t.cfg.pure_soft_survival
    ):
        raise ValueError(
            "Use the learned-head pure-soft source config/checkpoint for paired comparison"
        )
    if t.cfg.hard_imagination_boundaries or t.cfg.horizon != 32:
        raise ValueError("Paired probe requires H32 and no hard boundaries")
    out = Path(output) / "soft_termination_check.json"
    if out.exists():
        raise FileExistsError(out)
    names = ("actor", "critic", "actor_opt", "critic_opt")
    initial = {n: copy.deepcopy(getattr(t, n).state_dict()) for n in names}
    gradients = {
        n: [
            None if p.grad is None else p.grad.clone()
            for p in getattr(t, n).parameters()
        ]
        for n in ("actor", "critic")
    }
    original_cfg = t.cfg
    cfg, head, head_opt = copy.deepcopy(t.cfg), t.termination, t.termination_opt
    counter, profile, sample = t.actor_updates, t.profile_next_actor, t.sample
    event_rng = t.event_scheduler.rng.clone() if t.event_scheduler is not None else None
    rng = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
        torch.cuda.get_rng_state_all() if str(t.device).startswith("cuda") else None,
    )
    report = {
        "completed": False,
        "variants": {},
        "same_critic_layout": t.critic_dim,
        "note": "Different objectives: gradients/updates are compared descriptively, not required equal.",
    }
    reference_gradient = None

    def restore():
        for n in names:
            getattr(t, n).load_state_dict(copy.deepcopy(initial[n]))
        for n, values in gradients.items():
            for p, g in zip(getattr(t, n).parameters(), values):
                p.grad = None if g is None else g.clone()
        t.actor_updates, t.profile_next_actor = counter, False
        if event_rng is not None:
            t.event_scheduler.rng = event_rng.clone()
        seed(29032)

    try:
        seed(29032)
        starts = sample(starts=True, n=t.cfg.actor_batch or t.cfg.batch)
        t.sample = lambda *args, **kwargs: {k: v.clone() for k, v in starts.items()}
        # Same observed real states and reference frames; no labels used as inputs.
        real = sample(1, validation=True, n=min(256, t.cfg.batch))
        real_state = real.get("physical_next_state", real["next_state"])[0]
        real_phase = (real["phase"][0] + 1).clamp_max(len(t.task.ref_q) - 1)
        root = t.task.tracked_names.index("pelvis")
        real_probability = probability(t.task, real_state, real_phase)
        real_expected = matrix_reference(
            real_state[:, 2],
            real_state[:, 3:7],
            t.task.ref_p[real_phase, root, 2],
            t.task.ref_quat[real_phase, root],
        )
        torch.testing.assert_close(real_probability, real_expected)
        report["real_states"] = {
            "formula_max_abs": float((real_probability - real_expected).abs().max()),
            **distribution(real_probability),
        }
        with torch.no_grad():
            report["real_states"]["learned_probability"] = distribution(
                head(real["termination_features"][0]).sigmoid()
            )
        for label in ("learned", "analytic"):
            restore()
            t.cfg = copy.deepcopy(cfg)
            t.cfg.analytic_soft_termination = label == "analytic"
            t.cfg.learned_termination = label == "learned"
            t.cfg.joint_predicted_termination = (
                cfg.joint_predicted_termination if label == "learned" else False
            )
            t.termination, t.termination_opt = (
                (head, head_opt) if label == "learned" else (None, None)
            )
            cold_begin = timestamp(t.device)
            with freeze(t.model, head, t.critic, t.prior._disc):
                imagined = t.imagine(t.sample(), trace=True)
                value = -returns(imagined, t.critic, t.cfg.gamma).mean()
                if t.cfg.actor_horizon_scale:
                    value = value / t.cfg.horizon
                grad = torch.cat(
                    [
                        g.flatten()
                        for g in torch.autograd.grad(value, tuple(t.actor.parameters()))
                    ]
                )
                finite = bool(torch.isfinite(grad).all())
                if not finite:
                    raise RuntimeError("Non-finite Actor gradient")
                if reference_gradient is None:
                    reference_gradient = grad.detach().cpu().clone()
                grad_cpu = grad.detach().cpu()
                p = imagined["done"]
                survival = torch.cumprod(1 - p, dim=0)
                effective_survival = torch.cumprod(
                    (1 - p) * (~imagined["truncated"]), dim=0
                )
                stats = {
                    "probability": distribution(p[imagined["valid"]]),
                    "gradient_norm": float(grad.detach().norm()),
                    "gradient_cosine_vs_learned": float(
                        torch.nn.functional.cosine_similarity(
                            grad_cpu, reference_gradient, dim=0
                        )
                    ),
                    "gradient_finite": finite,
                    "soft_survival_after": {
                        str(h): float(survival[h - 1].detach().mean())
                        for h in (1, 8, 16, 32)
                    },
                    "effective_survival_after_truncation": {
                        str(h): float(effective_survival[h - 1].detach().mean())
                        for h in (1, 8, 16, 32)
                    },
                    "valid_fraction": float(imagined["valid"].float().mean()),
                }
                if label == "analytic":
                    state, phase = (
                        imagined["physical_states"],
                        imagined["reference_next_phases"],
                    )
                    expected = matrix_reference(
                        state[..., 2],
                        state[..., 3:7],
                        t.task.ref_p[phase, root, 2],
                        t.task.ref_quat[phase, root],
                    )
                    torch.testing.assert_close(p, expected)
                    stats["trajectory_formula_max_abs"] = float(
                        (p - expected).detach().abs().max()
                    )
                mode = imagined["predicted_launch_probability"]
                stats["waiting_to_launch"] = (
                    int(((mode[1:] > 0.5) & (mode[:-1] <= 0.5)).sum())
                    if mode.numel()
                    else None
                )
            del imagined, value, grad, p, survival, effective_survival, grad_cpu
            if label == "analytic":
                del state, phase, expected
            stats["cold_diagnostic_includes_compile_seconds"] = (
                timestamp(t.device) - cold_begin
            )
            rows = []
            for j in range(repeats + 1):
                restore()
                begin = timestamp(t.device)
                with freeze(t.model, head, t.critic, t.prior._disc):
                    imagined = t.imagine(t.sample())
                    value = -returns(imagined, t.critic, t.cfg.gamma).mean()
                    if t.cfg.actor_horizon_scale:
                        value = value / t.cfg.horizon
                    mid = timestamp(t.device)
                    torch.autograd.grad(value, tuple(t.actor.parameters()))
                    end = timestamp(t.device)
                del imagined, value
                restore()
                metrics = t.update_actor()
                rows.append(
                    {
                        "forward_seconds": mid - begin,
                        "BPTT_seconds": end - mid,
                        **metrics,
                    }
                )
            stats.update(warmup=rows[0], steady=rows[1:])
            report["variants"][label] = stats
            out.write_text(json.dumps(report, indent=2))
        report["completed"] = True
    finally:
        restore()
        t.cfg, t.termination, t.termination_opt, t.sample, t.profile_next_actor = (
            original_cfg,
            head,
            head_opt,
            sample,
            profile,
        )
        if event_rng is not None:
            t.event_scheduler.rng = event_rng
        random.setstate(rng[0])
        np.random.set_state(rng[1])
        torch.set_rng_state(rng[2])
        if rng[3] is not None:
            torch.cuda.set_rng_state_all(rng[3])
        out.write_text(json.dumps(report, indent=2))
    return report
