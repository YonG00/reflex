"""Paired fixed-replay GPU check; trial updates are restored, never checkpointed."""

import copy
import hashlib
import importlib.util
import json
import random
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from .timing import timestamp


def check(
    t, output, repeats=3, deterministic=False, include_cudagraph=False, graph_only=False
):
    out = Path(output)
    if t.termination is None:
        raise ValueError(
            "Legacy V27 paired efficiency baseline requires a learned head; use soft_termination_check or actor_timing for analytic mode"
        )
    if not t.actor_warmup_complete():
        raise ValueError(
            "Use an already trained compatible checkpoint for performance comparison"
        )
    names = (
        "model",
        "termination",
        "actor",
        "critic",
        "model_opt",
        "termination_opt",
        "actor_opt",
        "critic_opt",
        "obs_norm",
        "critic_norm",
    )

    def cpu(value):
        if torch.is_tensor(value):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {k: cpu(v) for k, v in value.items()}
        if isinstance(value, list):
            return [cpu(v) for v in value]
        if isinstance(value, tuple):
            return tuple(cpu(v) for v in value)
        return copy.deepcopy(value)

    initial = {n: cpu(getattr(t, n).state_dict()) for n in names}
    amp = cpu(t.prior.amp_state_dict())
    attrs = (
        "actor_updates",
        "model_updates",
        "termination_updates",
        "wm_rounds_completed",
        "wm_warmup_inherited",
        "normalized",
        "_termination_probe_due",
        "replay_seconds",
    )
    counters = {n: getattr(t, n) for n in attrs}
    rng = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
        torch.cuda.get_rng_state_all(),
    )
    event_rng = t.event_scheduler.rng.clone()
    cfg = copy.deepcopy(t.cfg)
    original_indices = t.replay.indices
    original_dynamics = t._dynamics
    original_cache = t.replay.cache_indices
    original_wm_step = t._wm_training_step
    original_precision = t.model.net.precision
    original_termination_features = t.task.predicted_termination_features
    feature_override = "predicted_termination_features" in t.task.__dict__

    def restore():
        for n in names:
            getattr(t, n).load_state_dict(copy.deepcopy(initial[n]))
        t.prior.load_amp_state_dict(amp)
        for n, v in counters.items():
            setattr(t, n, v)
        random.setstate(rng[0])
        np.random.set_state(rng[1])
        torch.set_rng_state(rng[2])
        torch.cuda.set_rng_state_all(rng[3])
        t.event_scheduler.rng = event_rng.clone()
        t.replay._index_cache.clear()

    records = {
        "config": {
            "envs": t.replay.data["state"].shape[1],
            "horizon": cfg.horizon,
            "actor_batch": cfg.actor_batch,
            "wm_steps": cfg.model_steps,
            "wm_horizon": cfg.model_window,
            "wm_mlp_precision": cfg.wm_mlp_precision,
            "minimal_termination_fk": cfg.minimal_termination_fk,
            "compile_wm_training_step": cfg.compile_wm_training_step,
            "replay_count": t.replay.count,
            "replay_pointer": t.replay.pointer,
        },
        "gpu": torch.cuda.get_device_name(t.device),
        "torch": torch.__version__,
        "source_sha256": {
            name: hashlib.sha256(
                Path(__file__).with_name(name).read_bytes()
            ).hexdigest()
            for name in (
                "trainer.py",
                "models.py",
                "physics.py",
                "task.py",
                "efficiency_benchmark.py",
            )
        },
        "variants": {},
        "replay_memory": t.replay.memory_report(1000),
    }
    original_determinism = torch.are_deterministic_algorithms_enabled()
    expected = None
    try:
        with tempfile.TemporaryDirectory(prefix="reflex_v26_baseline_") as directory:

            def baseline(file):
                name = "reflex.world_model._gpu_baseline_" + file
                path = Path(directory) / (file + ".py")
                path.write_bytes(
                    subprocess.check_output(
                        [
                            "git",
                            "show",
                            f"63b3ec8:source/reflex/reflex/world_model/{file}.py",
                        ]
                    )
                )
                spec = importlib.util.spec_from_file_location(name, path)
                module = importlib.util.module_from_spec(spec)
                sys.modules[name] = module
                spec.loader.exec_module(module)
                return module

            old_trainer, old_joint, old_replay, old_task = (
                baseline(n) for n in ("trainer", "joint", "replay", "task")
            )
            labels = (
                ("baseline", "cudagraph_trial")
                if graph_only
                else ("baseline", "optimized", "cudagraph_trial")
                if include_cudagraph
                else ("baseline", "optimized")
            )
            for label in labels:
                restore()
                if label == "baseline":
                    t.task.predicted_termination_features = (
                        old_task.AnalyticTask.predicted_termination_features.__get__(
                            t.task
                        )
                    )
                elif feature_override:
                    t.task.predicted_termination_features = (
                        original_termination_features
                    )
                elif "predicted_termination_features" in t.task.__dict__:
                    del t.task.predicted_termination_features
                t.cfg.synchronize_replay_timing = label == "baseline"
                t.cfg.minimal_termination_fk = (
                    False if label == "baseline" else cfg.minimal_termination_fk
                )
                t.model.net.precision = (
                    "fp32" if label == "baseline" else original_precision
                )
                t._wm_training_step = None if label == "baseline" else original_wm_step
                t.replay.cache_indices = label != "baseline"
                t.replay.indices = (
                    old_replay.Replay.indices.__get__(t.replay)
                    if label == "baseline"
                    else original_indices
                )
                if label == "cudagraph_trial":
                    t._dynamics = torch.compile(
                        t.model.predict_physical,
                        fullgraph=True,
                        dynamic=False,
                        mode="reduce-overhead",
                    )
                    if original_wm_step is not None:
                        t._wm_training_step = torch.compile(
                            t.v24_model_training_step,
                            fullgraph=True,
                            dynamic=False,
                            mode="reduce-overhead",
                        )
                else:
                    t._dynamics = original_dynamics
                fit = (
                    old_trainer.Trainer.fit_model.__get__(t)
                    if label == "baseline"
                    else t.fit_model
                )
                update = (
                    (lambda: old_joint._update_actor(t))
                    if label == "baseline"
                    else t.update_actor
                )
                from .events_v24.evaluation import evaluate

                forecasts = evaluate(t, batch=16)
                records["variants"].setdefault(label, {})[
                    "initial_checkpoint_forecasts"
                ] = forecasts
                rows = []
                variant_reference = None
                records["variants"][label].update(rows=rows, stage="running")
                numerical_failures = []
                try:
                    for trial in range(repeats + 2):
                        torch.use_deterministic_algorithms(deterministic and trial == 0)
                        restore()
                        records["variants"][label].update(
                            last_trial=trial, trial_started_unix=time.time()
                        )
                        (out / "efficiency_benchmark.json").write_text(
                            json.dumps(records, indent=2)
                        )
                        torch.cuda.reset_peak_memory_stats()
                        start = timestamp(t.device)
                        wm = fit()
                        after_wm = timestamp(t.device)
                        # Validation snapshots must precede the next graph phase:
                        # CUDA Graph backward outputs may own the leaf grad storage.
                        saved_wm_gradients = (
                            {
                                n: tuple(
                                    None if p.grad is None else p.grad.detach().clone()
                                    for p in getattr(t, n).parameters()
                                )
                                for n in ("model", "termination")
                            }
                            if trial == 0
                            else None
                        )
                        actor = update()
                        end = timestamp(t.device)
                        row = {
                            "deterministic_validation": deterministic and trial == 0,
                            "excluded_compile_warmup": trial <= 1,
                            "wm10_including_amp_seconds": after_wm - start,
                            "actor_critic_wall_seconds": end - after_wm,
                            "update_wall_seconds": end - start,
                            "wm": wm,
                            "actor": actor,
                            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                        }
                        rows.append(row)
                        (out / "efficiency_benchmark.json").write_text(
                            json.dumps(records, indent=2)
                        )
                        print(
                            json.dumps(
                                {
                                    "efficiency_variant": label,
                                    "trial": trial,
                                    "wall_seconds": row["update_wall_seconds"],
                                }
                            ),
                            flush=True,
                        )
                        if trial == 0:
                            current = {
                                n: cpu(getattr(t, n).state_dict()) for n in names
                            }
                            variant_reference = {
                                n: current[n]
                                for n in ("model", "termination", "actor", "critic")
                            }
                            for n in ("model", "termination", "actor", "critic"):
                                current[n + "_gradients"] = (
                                    cpu(saved_wm_gradients[n])
                                    if n in ("model", "termination")
                                    else tuple(
                                        cpu(p.grad) for p in getattr(t, n).parameters()
                                    )
                                )
                            if expected is None:
                                expected = current
                            else:
                                errors = {}
                                for n, value in current.items():

                                    def compare(
                                        a, b, failures=numerical_failures, component=n
                                    ):
                                        if torch.is_tensor(a):
                                            try:
                                                torch.testing.assert_close(
                                                    a, b, atol=2e-5, rtol=3e-3
                                                )
                                            except AssertionError as difference_error:
                                                if len(failures) < 30:
                                                    failures.append(
                                                        {
                                                            "component": component,
                                                            "error": str(
                                                                difference_error
                                                            ),
                                                        }
                                                    )
                                            return (
                                                float((a - b).abs().max())
                                                if a.numel()
                                                else 0.0
                                            )
                                        if isinstance(a, dict):
                                            assert a.keys() == b.keys()
                                            return max(
                                                (compare(a[k], b[k]) for k in a),
                                                default=0.0,
                                            )
                                        if isinstance(a, (list, tuple)):
                                            return max(
                                                (compare(x, y) for x, y in zip(a, b)),
                                                default=0.0,
                                            )
                                        assert a == b
                                        return 0.0

                                    errors[n] = compare(value, expected[n])
                                records["variants"].setdefault(label, {})[
                                    "parameter_and_adam_max_abs_errors"
                                ] = errors
                                gradient_metrics = {}
                                for name in ("model", "termination", "actor", "critic"):
                                    pairs = [
                                        (a.double(), b.double())
                                        for a, b in zip(
                                            current[name + "_gradients"],
                                            expected[name + "_gradients"],
                                        )
                                        if a is not None and b is not None
                                    ]
                                    aa = sum(a.square().sum() for a, b in pairs)
                                    bb = sum(b.square().sum() for a, b in pairs)
                                    dot = sum((a * b).sum() for a, b in pairs)
                                    difference = sum(
                                        (a - b).square().sum() for a, b in pairs
                                    )
                                    gradient_metrics[name] = {
                                        "cosine_similarity": float(
                                            dot / (aa * bb).sqrt().clamp_min(1e-30)
                                        ),
                                        "relative_l2_error": float(
                                            (difference / bb.clamp_min(1e-30)).sqrt()
                                        ),
                                        "reference_norm": float(bb.sqrt()),
                                        "actual_norm": float(aa.sqrt()),
                                    }
                                records["variants"][label]["gradient_comparison"] = (
                                    gradient_metrics
                                )
                        elif variant_reference is not None:
                            row["within_variant_parameter_max_abs_noise"] = {
                                n: max(
                                    float(
                                        (v.detach().cpu() - variant_reference[n][key])
                                        .abs()
                                        .max()
                                    )
                                    for key, v in getattr(t, n).state_dict().items()
                                    if v.numel()
                                )
                                for n in variant_reference
                            }
                        (out / "efficiency_benchmark.json").write_text(
                            json.dumps(records, indent=2)
                        )
                    if label == "cudagraph_trial":
                        restore()
                        with torch.profiler.profile(
                            activities=[
                                torch.profiler.ProfilerActivity.CPU,
                                torch.profiler.ProfilerActivity.CUDA,
                            ]
                        ) as profiler:
                            fit()
                            update()
                            timestamp(t.device)
                        events = profiler.events()
                        launches = sum("graphlaunch" in e.name.lower() for e in events)
                        records["variants"][label]["cuda_graph_launch_events"] = (
                            launches
                        )
                        records["variants"][label]["profiler_extra_updates"] = {
                            "wm": cfg.model_steps,
                            "actor": 1,
                        }
                        profiler.export_chrome_trace(
                            str(out / "cudagraph_update_trace.json")
                        )
                        (out / "cudagraph_update_profile.txt").write_text(
                            profiler.key_averages().table(
                                sort_by="self_cpu_time_total", row_limit=50
                            )
                        )
                    records["variants"].setdefault(label, {}).update(
                        {
                            "stage": "completed",
                            "completed": True,
                            "passed": not numerical_failures,
                            "update_equivalence_passed": not numerical_failures,
                            "numerical_failures": numerical_failures,
                            "cold_first_update_seconds": rows[0]["update_wall_seconds"],
                            "steady_median_update_seconds": statistics.median(
                                r["update_wall_seconds"] for r in rows[2:]
                            ),
                            "rows": rows,
                        }
                    )
                except (KeyboardInterrupt, SystemExit) as e:
                    records["variants"][label].update(
                        stage="interrupted",
                        completed=False,
                        passed=False,
                        error=repr(e),
                    )
                    (out / "efficiency_benchmark.json").write_text(
                        json.dumps(records, indent=2)
                    )
                    raise
                except Exception as e:
                    records["variants"].setdefault(label, {}).update(
                        passed=False, error=repr(e), rows=rows
                    )
                    print(
                        json.dumps({"efficiency_variant": label, "error": repr(e)}),
                        flush=True,
                    )
                    if label != "cudagraph_trial":
                        raise
                finally:
                    (out / "efficiency_benchmark.json").write_text(
                        json.dumps(records, indent=2)
                    )
    finally:
        torch.use_deterministic_algorithms(original_determinism)
        t.cfg = cfg
        t.replay.indices = original_indices
        t.replay.cache_indices = original_cache
        t._dynamics = original_dynamics
        t._wm_training_step = original_wm_step
        t.model.net.precision = original_precision
        if feature_override:
            t.task.predicted_termination_features = original_termination_features
        elif "predicted_termination_features" in t.task.__dict__:
            del t.task.predicted_termination_features
        restore()
    return records


def capacity_check(t, adapter, output):
    """Independent capacity trial; collect full ring, run bounded updates, no save."""
    out = Path(output)
    result = {
        "capacity_steps": t.cfg.capacity_steps,
        "checkpoint_saved": False,
        "trial_weights_discarded": True,
    }
    try:
        for j in range(t.cfg.capacity_steps + 1):
            t.collect(adapter)
            if (j + 1) % 64 == 0:
                print(json.dumps({"capacity_collection_step": j + 1}), flush=True)
        result["replay"] = t.replay.memory_report()
        result["rows"] = []
        for j in range(4):
            torch.cuda.reset_peak_memory_stats()
            start = timestamp(t.device)
            model = t.fit_model()
            actor = t.update_actor()
            elapsed = timestamp(t.device) - start
            row = {
                "trial": j,
                "model": model,
                "actor": actor,
                "update_seconds": elapsed,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
            result["rows"].append(row)
            print(json.dumps({"capacity_update": j, "seconds": elapsed}), flush=True)
        result["passed"] = True
    except Exception as e:
        result.update(passed=False, error=repr(e))
        raise
    finally:
        (out / "capacity_check.json").write_text(json.dumps(result, indent=2))
    return result
