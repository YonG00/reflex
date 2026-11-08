"""Profile the complete current Actor update, without removing task features."""

import copy
import hashlib
import json
import statistics
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch

from .. import amp_batch, joint
from ..timing import timestamp
from ..trainer import freeze
from .attribution import summarize
from .state import Snapshot, digest, parameter_result


def distribution(values):
    values = sorted(values)
    return {
        "median": statistics.median(values),
        "p90": values[min(len(values) - 1, int(0.9 * len(values)))],
        "min": values[0],
        "max": values[-1],
    }


@contextmanager
def scopes(t):
    """Only eager call boundaries; never patch inside compiled model/FK code."""
    if (
        t.cfg.fuse_actor_core
        or (t.cfg.compile_v24_step and t.cfg.v24_fusion_scope == "full")
        or t.cfg.compile_step
    ):
        raise ValueError(
            "Use the verified packed-head configuration; fused units require separate attribution"
        )
    hooks = []
    region_present = "_v31_regions" in t.__dict__
    region_saved = t.__dict__.get("_v31_regions")
    t._v31_regions = True
    phase = "actor"

    def wrap(owner, name, label):
        original = getattr(owner, name)
        if original is None:
            return
        present = name in owner.__dict__
        saved = owner.__dict__.get(name)
        depth = 0

        def scoped(*args, **kwargs):
            nonlocal depth, phase
            if owner is t and name == "sample" and kwargs.get("starts", False):
                phase = "actor"
            outer = depth == 0
            depth += 1
            try:
                if outer:
                    active_label = (
                        f"{phase}/{label}"
                        if label in ("critic_forward", "gradient_clip")
                        else label
                    )
                    with torch.profiler.record_function("v31/" + active_label):
                        result = original(*args, **kwargs)
                    if owner is t.actor_opt and name == "step":
                        phase = "critic"
                    return result
                return original(*args, **kwargs)
            finally:
                depth -= 1

        hooks.append((owner, name, present, saved))
        setattr(owner, name, scoped)

    try:
        for owner, name, label in (
            (t, "sample", "replay/sample"),
            (t, "_observations", "observation"),
            (t.actor, "forward", "actor_mlp"),
            (t, "targets", "action_targets"),
            (t, "_dynamics", "physical_wm"),
            (t, "_fk", "fk"),
            (t, "_v24_heads", "reward_termination_packed"),
            (t.event_scheduler, "step", "native_events"),
            (t.event_scheduler.codec, "decode", "event_decode"),
            (t.event_scheduler.codec, "encode", "event_encode"),
            (t.event_scheduler.codec, "pre_features", "wm_event_features"),
            (t, "_post", "deterministic_apply"),
            (amp_batch, "feature_pairs", "amp_features_batched"),
            (sys.modules[type(t).__module__], "amp_features", "amp_features_per_step"),
            (t.prior, "reward", "amp_reward"),
            (t.prior._disc, "forward", "amp_discriminator"),
            (joint, "returns", "actor_return"),
            (joint, "critic_targets", "critic_targets"),
            (t.critic, "forward", "critic_forward"),
            (t.actor_opt, "step", "actor_adam"),
            (t.critic_opt, "step", "critic_adam"),
            (torch.nn.utils, "clip_grad_norm_", "gradient_clip"),
            (torch.nn.utils, "clip_grad_value_", "gradient_clip"),
        ):
            wrap(owner, name, label)
        if not t.cfg.compile_v24_step:
            wrap(t.task, "evaluate_step", "reward_termination_eager")
        yield
    finally:
        if region_present:
            t._v31_regions = region_saved
        else:
            del t._v31_regions
        for owner, name, present, saved in reversed(hooks):
            if present:
                setattr(owner, name, saved)
            else:
                delattr(owner, name)


def compare(a, b):
    errors, failures = {}, {}
    if set(a) != set(b):
        raise ValueError("Diagnostic output schema changed")
    for key, value in a.items():
        errors[key] = (
            float((value.float() - b[key].float()).abs().max())
            if value.numel()
            else 0.0
        )
        try:
            torch.testing.assert_close(value, b[key])
        except AssertionError as exc:
            failures[key] = str(exc)
    return {"passed": not failures, "max_abs": errors, "failures": failures}


def semantic_probe(t, snapshot, extra_scopes=None):
    outputs = []
    coverage = {}
    for instrumented in (False, True):
        snapshot.reset()
        with freeze(t.model, t.termination, t.prior._disc, t.critic):
            if instrumented:
                with scopes(t), extra_scopes(t) if extra_scopes else nullcontext():
                    batch = t.sample(starts=True, n=t.cfg.actor_batch or t.cfg.batch)
                    result = t.imagine(batch, trace=True)
            else:
                batch = t.sample(starts=True, n=t.cfg.actor_batch or t.cfg.batch)
                result = t.imagine(batch, trace=True)
            objective = joint.returns(result, t.critic, t.cfg.gamma).mean()
            gradients = torch.autograd.grad(objective, tuple(t.actor.parameters()))
            row = {k: v.detach().cpu().clone() for k, v in result.items()}
            row["actor_gradient"] = torch.cat(
                [v.detach().flatten().cpu() for v in gradients]
            )
            outputs.append(row)
            modes = result.get("predicted_launch_probability")
            if modes is not None:
                before = torch.cat((batch["launched"][None].float(), modes[:-1]))
                coverage = {
                    "waiting_to_launch": int(((modes > 0.5) & (before <= 0.5)).sum()),
                    "return_to_waiting": int(((modes <= 0.5) & (before > 0.5)).sum()),
                    "truncations": int(result["truncated"].sum()),
                }
            del result, objective, gradients, batch
    return compare(*outputs), coverage


def compile_counts():
    from torch._dynamo.utils import counters

    return {
        "unique_graphs": int(counters["stats"]["unique_graphs"]),
        "frames": int(counters["frames"]["total"]),
        "graph_breaks": int(sum(counters["graph_break"].values())),
    }


def count_delta(after, before):
    return {k: after[k] - before[k] for k in before}


def sampling_audit(t, snapshot, sampler, repeats):
    records, signatures = {}, {}
    for label in ("cold_index_cache", "warm_index_cache"):
        times = []
        for _ in range(repeats):
            snapshot.reset()
            if label == "cold_index_cache":
                t.replay._index_cache.clear()
            begin = timestamp(t.device)
            batch = sampler(starts=True, n=t.cfg.actor_batch or t.cfg.batch)
            times.append(timestamp(t.device) - begin)
            if label not in signatures:
                signatures[label] = digest(batch)
            del batch
        records[label] = distribution(times)
    return {
        "seconds": records,
        "input_hash_equal": len(set(signatures.values())) == 1,
        "cache_enabled": t.replay.cache_indices,
        "note": "Start sampling only; cold invalidates the existing index cache without changing Replay or sampling rules.",
    }


def check(
    t,
    output,
    repeats=20,
    warmups=2,
    semantic=True,
    extra_scopes=None,
    extra_summary=None,
):
    if repeats < 1 or warmups < 1:
        raise ValueError("Positive repeat and warmup counts are required")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "actor_full_v31.json"
    if path.exists():
        raise FileExistsError(path)
    snapshot = Snapshot(t)
    sample_present = "sample" in t.__dict__
    sample_saved = t.__dict__.get("sample")
    original_sample = t.sample
    observed = {}

    def observe_sample(*args, **kwargs):
        batch = original_sample(*args, **kwargs)
        if kwargs.get("starts", False):
            observed["starts"] = batch
        return batch

    t.sample = observe_sample
    result = {
        "completed": False,
        "config": copy.deepcopy(vars(t.cfg)),
        "device": str(t.device),
        "torch": torch.__version__,
        "weights_saved": False,
        "timing": "Unprofiled complete update with actual sampling from frozen Replay; excludes state restoration",
    }
    sources = Path(__file__).resolve().parents[1]
    result["source_sha256"] = {
        name: hashlib.sha256((sources / name).read_bytes()).hexdigest()
        for name in (
            "trainer.py",
            "joint.py",
            "amp_batch.py",
            "physics.py",
            "models.py",
            "diagnostics_v31/full.py",
            "diagnostics_v31/attribution.py",
        )
    }
    result["actual_precision"] = {
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "wm_mlp": getattr(t.model.net, "precision", "fp32"),
        "actor_dtype": str(next(t.actor.parameters()).dtype),
    }
    result["cuda_version"] = torch.version.cuda
    if str(t.device).startswith("cuda"):
        result["gpu"] = torch.cuda.get_device_name(t.device)
        result["device_memory_before"] = torch.cuda.mem_get_info(t.device)
    try:
        initial_actor = digest(t.actor.state_dict())
        result["actor_initial_sha256"] = initial_actor
        result["model_initial_sha256"] = digest(t.model.state_dict())
        result["rng_sha256"] = digest(
            {"cpu": snapshot.rng[2], "event": snapshot.scheduler_rng}
        )
        cold = []
        for _ in range(warmups):
            snapshot.reset()
            begin = timestamp(t.device)
            t.update_actor()
            cold.append(timestamp(t.device) - begin)
            observed.clear()
        result["warmup_seconds"] = cold
        if semantic:
            result["scope_semantics"], result["coverage"] = semantic_probe(
                t, snapshot, extra_scopes
            )
            if not result["scope_semantics"]["passed"]:
                raise RuntimeError("Instrumentation changes trajectories or gradients")
            observed.clear()
        rows = []
        reference = None
        for _ in range(repeats):
            snapshot.reset()
            if str(t.device).startswith("cuda"):
                torch.cuda.reset_peak_memory_stats(t.device)
            begin = timestamp(t.device)
            metrics = t.update_actor()
            seconds = timestamp(t.device) - begin
            rows.append({"complete_wall_seconds": seconds, **metrics})
            if str(t.device).startswith("cuda"):
                rows[-1].update(
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(t.device),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(t.device),
                )
            if reference is None:
                reference = parameter_result(t)
                result["sample_sha256"] = digest(observed["starts"])
            observed.clear()
        result["unprofiled"] = rows
        result["wall_seconds"] = distribution(
            [r["complete_wall_seconds"] for r in rows]
        )
        result["actor_seconds"] = distribution(
            [r["time/actor_update_seconds"] for r in rows]
        )
        result["critic_seconds"] = distribution(
            [r["time/critic_update_seconds"] for r in rows]
        )
        snapshot.reset()
        scope_counts_before = compile_counts()
        with scopes(t), extra_scopes(t) if extra_scopes else nullcontext():
            # Establish identical wrapper guards outside profiler, discard update.
            t.update_actor()
            observed.clear()
            snapshot.reset()
            activities = [torch.profiler.ProfilerActivity.CPU]
            if str(t.device).startswith("cuda"):
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            counts_before_capture = compile_counts()
            begin = timestamp(t.device)
            with (
                torch.profiler.profile(
                    activities=activities, record_shapes=True, with_stack=False
                ) as profiler,
                torch.profiler.record_function("v31/complete_update"),
            ):
                profiled_metrics = t.update_actor()
            result["profiled_wall_seconds"] = timestamp(t.device) - begin
        result["scope_warmup_compile_delta"] = count_delta(
            counts_before_capture, scope_counts_before
        )
        result["capture_compile_delta"] = count_delta(
            compile_counts(), counts_before_capture
        )
        result["capture_excludes_compilation"] = not any(
            result["capture_compile_delta"].values()
        )
        result["profiled_sample_sha256"] = digest(observed["starts"])
        observed.clear()
        result["profiled_parameter_and_gradient_equivalence"] = compare(
            reference, parameter_result(t)
        )
        result["profiled_metrics_not_speed_benchmark"] = profiled_metrics
        result["attribution"] = summarize(profiler.events())
        if extra_summary is not None:
            result["execution_sites"] = extra_summary(profiler.events())
        result["operators"] = [
            {
                "name": e.key,
                "calls": e.count,
                "self_cpu_us": e.self_cpu_time_total,
                "device_total_us_inclusive": getattr(e, "device_time_total", 0),
                "self_device_us": getattr(e, "self_device_time_total", 0),
            }
            for e in profiler.key_averages()
        ]
        profiler.export_chrome_trace(str(out / "actor_full_trace.json"))
        result["sampling_audit"] = sampling_audit(t, snapshot, original_sample, repeats)
        result["completed"] = True
        result["numeric_passed"] = (
            result["profiled_parameter_and_gradient_equivalence"]["passed"]
            and result["sample_sha256"] == result["profiled_sample_sha256"]
        )
        result["limitations"] = [
            "Frozen Replay index-cache timing is not live collection throughput.",
            "Profiler timing is diagnostic only; use unprofiled median/P90.",
            "Backward shared or unmatched kernels are not guessed into individual modules.",
        ]
    finally:
        if sample_present:
            t.sample = sample_saved
        else:
            del t.sample
        observed.clear()
        snapshot.restore()
        result["model_unchanged"] = digest(t.model.state_dict()) == result.get(
            "model_initial_sha256"
        )
        result["actor_restored"] = digest(t.actor.state_dict()) == result.get(
            "actor_initial_sha256"
        )
        if result.get("completed"):
            result["attribution_usable"] = (
                result["capture_excludes_compilation"]
                and result["sampling_audit"]["input_hash_equal"]
            )
            result["numeric_passed"] &= (
                result["model_unchanged"] and result["actor_restored"]
            )
        path.write_text(json.dumps(result, indent=2))
    return result
