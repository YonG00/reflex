"""Paired opt-in AMP/Critic benchmark, without changing or saving training weights."""

import copy
import json
import random
import statistics
from pathlib import Path

import numpy as np
import torch

from .events_v24.compare import seed
from .joint import returns
from .timing import timestamp
from .trainer import freeze


def check(t, output, repeats=3):
    if t.event_scheduler is None:
        raise ValueError("Batching check requires the native event scheduler")
    out = Path(output) / "imagination_batching.json"
    if out.exists():
        raise FileExistsError(out)
    names = ("actor", "critic", "actor_opt", "critic_opt", "obs_norm", "critic_norm")
    initial = {n: copy.deepcopy(getattr(t, n).state_dict()) for n in names}
    gradients_before = {
        n: [
            None if p.grad is None else p.grad.detach().clone()
            for p in getattr(t, n).parameters()
        ]
        for n in ("actor", "critic")
    }
    profile_before = t.profile_next_actor
    flags = t.cfg.batch_imagination_amp, t.cfg.batch_imagination_critic
    rng = t.event_scheduler.rng.clone()
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    cpu_rng = torch.get_rng_state()
    cuda_rng = (
        torch.cuda.get_rng_state_all() if str(t.device).startswith("cuda") else None
    )
    count = t.actor_updates
    old_sample = t.sample
    report = {
        "completed": False,
        "variants": {},
        "errors": {},
        "tolerance": "torch.testing.assert_close default dtype tolerance",
    }
    expected = None
    devices = (
        [torch.device(t.device).index or 0] if str(t.device).startswith("cuda") else []
    )

    def restore():
        for n in names:
            getattr(t, n).load_state_dict(copy.deepcopy(initial[n]))
        for name, gradients in gradients_before.items():
            for parameter, gradient in zip(getattr(t, name).parameters(), gradients):
                parameter.grad = None if gradient is None else gradient.clone()
        t.actor_updates = count
        t.profile_next_actor = False
        t.event_scheduler.rng = rng.clone()
        seed(28032)

    def compare(label, a, b):
        error = float((a.float() - b.float()).abs().max()) if a.numel() else 0.0
        report["errors"][label] = error
        try:
            torch.testing.assert_close(a, b)
        except AssertionError as exc:
            report.setdefault("failed_checks", []).append(label)
            report.setdefault("failure_details", {})[label] = str(exc)

    try:
        with torch.random.fork_rng(devices=devices):
            seed(28032)
            starts = old_sample(starts=True, n=t.cfg.actor_batch or t.cfg.batch)
            # Force exactly the same sampled starts for the complete update as the trajectory check.
            t.sample = lambda *args, **kwargs: {k: v.clone() for k, v in starts.items()}
            for label, amp, critic in [
                ("reference", False, False),
                ("amp", True, False),
                ("amp_critic", True, True),
            ]:
                t.cfg.batch_imagination_amp, t.cfg.batch_imagination_critic = (
                    amp,
                    critic,
                )
                restore()
                current = {}
                if devices:
                    torch.cuda.reset_peak_memory_stats(t.device)
                cold_begin = timestamp(t.device)
                with freeze(t.model, t.termination, t.prior._disc, t.critic):
                    imagined = t.imagine(t.sample(), trace=True)
                    values = None
                    if critic:
                        obs = imagined["critic_obs"]
                        values = t.critic(obs.flatten(0, 1)).reshape(obs.shape[:2])
                        del obs
                    loss = -returns(imagined, t.critic, t.cfg.gamma, values).mean()
                    if t.cfg.actor_horizon_scale:
                        loss = loss / t.cfg.horizon
                    gradient = torch.autograd.grad(loss, tuple(t.actor.parameters()))
                    current["gradient"] = torch.cat(
                        [g.detach().flatten().cpu() for g in gradient]
                    )
                    current.update(
                        {k: v.detach().cpu().clone() for k, v in imagined.items()}
                    )
                cold_seconds = timestamp(t.device) - cold_begin
                mode = current.get("predicted_launch_probability")
                coverage = {"truncations": int(current["truncated"].sum())}
                if mode is not None and mode.numel():
                    coverage["waiting_to_launch"] = int(
                        ((mode[1:] > 0.5) & (mode[:-1] <= 0.5)).sum()
                    )
                    coverage["launched_to_waiting"] = int(
                        ((mode[1:] <= 0.5) & (mode[:-1] > 0.5)).sum()
                    )
                del imagined, loss, gradient, values
                rows = []
                for j in range(repeats + 1):
                    restore()
                    with freeze(t.model, t.termination, t.prior._disc, t.critic):
                        begin = timestamp(t.device)
                        imag = t.imagine(t.sample())
                        values = None
                        if critic:
                            obs = imag["critic_obs"]
                            values = t.critic(obs.flatten(0, 1)).reshape(obs.shape[:2])
                            del obs
                        objective = -returns(imag, t.critic, t.cfg.gamma, values).mean()
                        if t.cfg.actor_horizon_scale:
                            objective = objective / t.cfg.horizon
                        middle = timestamp(t.device)
                        torch.autograd.grad(objective, tuple(t.actor.parameters()))
                        end = timestamp(t.device)
                    del imag, objective, values
                    restore()
                    metrics = t.update_actor()
                    rows.append(
                        {
                            "forward_seconds": middle - begin,
                            "BPTT_seconds": end - middle,
                            **metrics,
                        }
                    )
                current.update(
                    {
                        f"{n}.{k}": v.detach().cpu().clone()
                        for n in ("actor", "critic")
                        for k, v in getattr(t, n).state_dict().items()
                    }
                )
                if expected is None:
                    expected = current
                else:
                    for k, value in current.items():
                        compare(f"{label}/{k}", value, expected[k])
                report["variants"][label] = {
                    "cold_diagnostic_includes_compile_and_CPU_snapshot_seconds": cold_seconds,
                    "warmup_update": rows[0],
                    "steady": rows[1:],
                    "coverage": coverage,
                    "gradient_norm": float(current["gradient"].norm()),
                    "gradient_cosine": float(
                        torch.nn.functional.cosine_similarity(
                            current["gradient"], expected["gradient"], dim=0
                        )
                    ),
                    "median_actor_seconds": statistics.median(
                        row["time/actor_update_seconds"] for row in rows[1:]
                    ),
                    "median_critic_seconds": statistics.median(
                        row["time/critic_update_seconds"] for row in rows[1:]
                    ),
                }
                out.write_text(json.dumps(report, indent=2))
            report["completed"] = True
            report["numeric_passed"] = not report.get("failed_checks")
            report["coverage_note"] = (
                "Missing event scenarios require dedicated native probes; numeric_passed alone does not authorize training."
            )
    finally:
        restore()
        t.sample = old_sample
        t.profile_next_actor = profile_before
        t.cfg.batch_imagination_amp, t.cfg.batch_imagination_critic = flags
        t.event_scheduler.rng = rng
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        out.write_text(json.dumps(report, indent=2))
    return report
