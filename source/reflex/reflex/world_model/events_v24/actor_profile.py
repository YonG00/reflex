"""Warm Actor timing and CPU/CUDA attribution on frozen replay; no long run."""

import json
from pathlib import Path

import torch

from .compare import seed


def run(trainer, output, profile=True):
    out = Path(output)
    records = []
    for j in range(3):
        seed(7300 + j)
        result = trainer.update_actor()
        records.append(result)
        print(json.dumps({"actor_benchmark": j, **result}), flush=True)
    if not profile:
        result = {
            "completed": True,
            "records": records,
            "batch": trainer.cfg.actor_batch,
            "horizon": trainer.cfg.horizon,
            "tf32": trainer.cfg.tf32,
            "fusion_scope": trainer.cfg.v24_fusion_scope,
            "diagnostic_actor_updates": len(records),
            "checkpoint_saved": False,
        }
        (out / "actor_timing.json").write_text(json.dumps(result, indent=2))
        return result
    scheduler = trainer.event_scheduler
    original_step = scheduler.step
    original_fk = trainer._fk
    original_targets = trainer.targets

    def event_step(*args, **kwargs):
        with torch.profiler.record_function("imagine/native_event_scheduler"):
            return original_step(*args, **kwargs)

    def fk(*args, **kwargs):
        with torch.profiler.record_function("imagine/all_FK_calls"):
            return original_fk(*args, **kwargs)

    def targets(*args, **kwargs):
        with torch.profiler.record_function("imagine/action_targets"):
            return original_targets(*args, **kwargs)

    scheduler.step, trainer._fk, trainer.targets = event_step, fk, targets
    activities = [torch.profiler.ProfilerActivity.CPU]
    if str(trainer.device).startswith("cuda"):
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    try:
        seed(7400)
        with torch.profiler.profile(activities=activities) as profiler:
            profiled = trainer.update_actor()
    finally:
        scheduler.step, trainer._fk, trainer.targets = (
            original_step,
            original_fk,
            original_targets,
        )
    stats = profiler.key_averages()
    rows = [
        {
            "name": e.key,
            "count": e.count,
            "cpu_total_ms": e.cpu_time_total / 1000,
            "self_cpu_ms": e.self_cpu_time_total / 1000,
            "device_total_ms": getattr(e, "device_time_total", 0) / 1000,
            "self_device_ms": getattr(e, "self_device_time_total", 0) / 1000,
        }
        for e in stats
    ]
    profiler.export_chrome_trace(str(out / "actor_trace.json"))
    (out / "actor_cpu.txt").write_text(
        stats.table(sort_by="self_cpu_time_total", row_limit=50)
    )
    (out / "actor_cuda.txt").write_text(
        stats.table(sort_by="self_cuda_time_total", row_limit=50)
    )
    result = {
        "completed": True,
        "warm_updates": records,
        "profiled_update": profiled,
        "operators": rows,
        "batch": trainer.cfg.actor_batch,
        "horizon": trainer.cfg.horizon,
        "compile_step": trainer.cfg.compile_step,
        "compile_dynamics": trainer.cfg.compile_dynamics,
        "compile_fk": trainer.cfg.compile_fk,
        "checkpoint_saved": False,
    }
    (out / "actor_profile.json").write_text(json.dumps(result, indent=2))
    return result


def benchmark_views(t, output, repeats=3):
    """No optimizer steps: exact trajectory/gradient check, then bounded timing."""
    import time
    import types

    from ..joint import returns
    from ..trainer import freeze
    from .fast_view import build

    out = Path(output)
    scheduler = t.event_scheduler
    reference = scheduler._view
    event_rng = scheduler.rng.clone()
    counters = t.actor_updates, t.model_updates

    def sync():
        if str(t.device).startswith("cuda"):
            torch.cuda.synchronize()

    def rollout(b, trial, timings=None):
        seed(8300 + trial)
        scheduler.rng = event_rng.clone()
        with freeze(t.model, t.termination, t.prior._disc, t.critic):
            sync()
            started = time.perf_counter()
            imagined = t.imagine(start_batch=b)
            sync()
            forward = time.perf_counter() - started
            loss = -returns(imagined, t.critic, t.cfg.gamma).mean() / t.cfg.horizon
            sync()
            started = time.perf_counter()
            gradients = torch.autograd.grad(loss, tuple(t.actor.parameters()))
            sync()
            backward = time.perf_counter() - started
        if timings is not None:
            timings.update(
                forward_seconds=forward,
                backward_seconds=backward,
                total_seconds=forward + backward,
            )
        return {
            k: imagined[k].detach()
            for k in ("obs", "reward", "done", "predicted_launch_probability")
        }, tuple(g.detach() for g in gradients)

    try:
        seed(8200)
        small = t.sample(starts=True, n=8)
        expected, eg = rollout(small, 0)
        scheduler._view = types.MethodType(build, scheduler)
        actual, ag = rollout(small, 0)
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], atol=2e-4, rtol=3e-3)
        for x, y in zip(ag, eg):
            torch.testing.assert_close(x, y, atol=2e-5, rtol=3e-3)
        result = {
            "equivalence_passed": True,
            "gradient_max_abs_error": max(
                float((x - y).abs().max()) for x, y in zip(ag, eg)
            ),
            "batch": t.cfg.actor_batch,
            "horizon": t.cfg.horizon,
            "optimizer_updates": 0,
        }
        del expected, actual, eg, ag
        seed(8201)
        batch = t.sample(starts=True, n=t.cfg.actor_batch)
        for variant in ("reference", "minimal_native_view"):
            scheduler._view = (
                reference
                if variant == "reference"
                else types.MethodType(build, scheduler)
            )
            rollout(batch, 1)  # warm graph/cache; excluded from measurements
            records = []
            for j in range(repeats):
                row = {}
                rollout(batch, 2 + j, row)
                records.append(row)
            scopes = {}

            def instrument(owner, name, label, scopes=scopes):
                original = getattr(owner, name)
                depth = 0

                def wrapped(*args, **kwargs):
                    nonlocal depth
                    outer = depth == 0
                    start = time.perf_counter()
                    depth += 1
                    try:
                        return original(*args, **kwargs)
                    finally:
                        depth -= 1
                        if outer:
                            d = scopes.setdefault(
                                label, {"calls": 0, "host_seconds": 0.0}
                            )
                            d["calls"] += 1
                            d["host_seconds"] += time.perf_counter() - start

                setattr(owner, name, wrapped)
                return owner, name, original

            hooks = []
            try:
                for owner, name, label in (
                    (t, "_dynamics", "WM"),
                    (t, "_fk", "FK_positions"),
                    (t, "_observations", "observations"),
                    (t, "targets", "action_targets"),
                    (t.task, "evaluate_step", "reward_termination"),
                    (t.task, "_body_data", "body_data_velocity_FK"),
                    (scheduler, "step", "native_scheduler"),
                ):
                    hooks.append(instrument(owner, name, label))
                profiled = {}
                rollout(batch, 6, profiled)
            finally:
                for owner, name, original in hooks:
                    setattr(owner, name, original)
            result[variant] = {
                "times": records,
                "instrumented": profiled,
                "host_scopes": scopes,
            }
            (out / "view_benchmark.json").write_text(json.dumps(result, indent=2))
            print(
                json.dumps({"view_benchmark": variant, **result[variant]}), flush=True
            )
        assert counters == (t.actor_updates, t.model_updates)
        result["completed"] = True
        (out / "view_benchmark.json").write_text(json.dumps(result, indent=2))
        return result
    finally:
        scheduler._view = reference
        scheduler.rng = event_rng


def update_equivalence(t, output, fusion=False):
    """Compare one complete Actor/Critic Adam update and restore initial state."""
    import copy
    import types

    from .fast_view import build
    from .runtime import NativeEventScheduler

    names = ("actor", "critic", "actor_opt", "critic_opt", "obs_norm", "critic_norm")
    initial = {name: copy.deepcopy(getattr(t, name).state_dict()) for name in names}
    original_view = t.event_scheduler._view
    original_step, original_observations = t._v24_step, t._observations
    if fusion and original_step is None:
        raise ValueError("Fusion update equivalence requires compiled V24 step")
    event_rng = t.event_scheduler.rng.clone()
    original_updates = t.actor_updates
    expected = None
    records = {}
    try:
        labels = (
            ("baseline", "fused") if fusion else ("reference", "minimal_native_view")
        )
        for label in labels:
            for name in names:
                getattr(t, name).load_state_dict(copy.deepcopy(initial[name]))
            t.actor_updates = original_updates
            if fusion:
                t._v24_step = original_step if label == "fused" else None
                t._observations = (
                    original_observations
                    if label == "fused"
                    else t.imagined_observations
                )
            else:
                t.event_scheduler._view = (
                    NativeEventScheduler._view.__get__(t.event_scheduler)
                    if label == "reference"
                    else types.MethodType(build, t.event_scheduler)
                )
            t.event_scheduler.rng = event_rng.clone()
            seed(9501)
            metrics = t.update_actor()
            current = {
                name: copy.deepcopy(getattr(t, name).state_dict())
                for name in ("actor", "critic")
            }
            records[label] = metrics
            if expected is None:
                expected = current
            else:
                errors = {
                    name + "_parameter_max_abs_error": max(
                        float((value - expected[name][key]).abs().max())
                        for key, value in state.items()
                    )
                    for name, state in current.items()
                }
                records.update(errors)
                records["passed"] = False
                (Path(output) / "actor_update_equivalence.json").write_text(
                    json.dumps(records, indent=2)
                )
                for name, parameter_state in current.items():
                    for key, value in parameter_state.items():
                        torch.testing.assert_close(
                            value,
                            expected[name][key],
                            atol=2e-6,
                            rtol=3e-3,
                        )
                    errors[name + "_parameter_max_abs_error"] = max(
                        float((value - expected[name][key]).abs().max())
                        for key, value in parameter_state.items()
                    )
                records.update(errors)
        records.update(
            passed=True,
            batch=t.cfg.actor_batch,
            horizon=t.cfg.horizon,
            diagnostic_optimizer_steps=2,
            initial_state_restored=True,
        )
    finally:
        for name in names:
            getattr(t, name).load_state_dict(copy.deepcopy(initial[name]))
        t.actor_updates = original_updates
        t.event_scheduler._view = original_view
        t._v24_step, t._observations = original_step, original_observations
        t.event_scheduler.rng = event_rng
    (Path(output) / "actor_update_equivalence.json").write_text(
        json.dumps(records, indent=2)
    )
    print(json.dumps(records), flush=True)
    return records
