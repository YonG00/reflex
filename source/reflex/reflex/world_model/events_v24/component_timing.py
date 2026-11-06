"""Bounded, read-only host attribution and synchronized H32 totals."""

import json
import sys
import time
from pathlib import Path

import torch

from ..joint import returns
from ..trainer import freeze


def check(t, output):
    scopes = {}
    hooks = []
    rng = t.event_scheduler.rng.clone()
    counts = (t.actor_updates, t.model_updates)
    devices = (
        [torch.device(t.device).index or 0] if str(t.device).startswith("cuda") else []
    )

    def sync():
        if devices:
            torch.cuda.synchronize()

    def wrap(owner, name, label):
        original = getattr(owner, name)
        if original is None:
            return

        depth = 0

        def measured(*args, **kwargs):
            nonlocal depth
            outer = depth == 0
            depth += 1
            begin = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                depth -= 1
                if outer:
                    row = scopes.setdefault(label, {"calls": 0, "host_seconds": 0.0})
                    row["calls"] += 1
                    row["host_seconds"] += time.perf_counter() - begin

        hooks.append((owner, name, original))
        setattr(owner, name, measured)

    def run(batch):
        t.event_scheduler.rng = rng.clone()
        sync()
        begin = time.perf_counter()
        imagined = t.imagine(batch)
        sync()
        mid = time.perf_counter()
        loss = -returns(imagined, t.critic, t.cfg.gamma).mean() / t.cfg.horizon
        gradients = torch.autograd.grad(
            loss, tuple(t.actor.parameters()), allow_unused=True
        )
        sync()
        end = time.perf_counter()
        return {
            "forward_seconds": mid - begin,
            "backward_seconds": end - mid,
            "total_seconds": end - begin,
            "gradient_norm": float(
                torch.stack([g.norm() for g in gradients if g is not None]).norm()
            ),
        }

    try:
        with (
            torch.random.fork_rng(devices=devices),
            freeze(t.model, t.termination, t.critic, t.prior._disc),
        ):
            batch = t.sample(starts=True, n=t.cfg.actor_batch)
            warm = run(batch)
            records = [run(batch) for _ in range(3)]
            for owner, name, label in (
                (t, "_dynamics", "physical_WM"),
                (t, "_fk", "position_FK"),
                (t, "_v24_heads", "reward_termination"),
                (t, "_observations", "observations"),
                (t.actor, "forward", "actor_MLP"),
                (t, "targets", "action_targets"),
                (t.prior, "reward", "AMP_discriminator"),
                (sys.modules[type(t).__module__], "amp_features", "AMP_features"),
                (t.event_scheduler, "step", "native_scheduler"),
                (t, "_post", "deterministic_apply"),
            ):
                wrap(owner, name, label)
            measured = run(batch)
        assert counts == (t.actor_updates, t.model_updates)
        result = {
            "completed": True,
            "batch": t.cfg.actor_batch,
            "horizon": t.cfg.horizon,
            "optimizer_updates": 0,
            "warm_excluded": warm,
            "records": records,
            "instrumented": measured,
            "host_scopes": scopes,
            "warning": "Host scopes include launch/synchronization and may nest; not GPU exclusive time; do not sum.",
        }
        Path(output).mkdir(parents=True, exist_ok=True)
        (Path(output) / "component_timing.json").write_text(
            json.dumps(result, indent=2)
        )
        return result
    finally:
        for owner, name, original in reversed(hooks):
            setattr(owner, name, original)
        t.event_scheduler.rng = rng
