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


def check(
    t,
    output,
    repeats=3,
    execution_variants=False,
    graph_trial=False,
    graph_all_static=False,
    graph_only=False,
    stable_heading=False,
    amp_feature_variants=False,
    subset_variants=False,
    event_tensor_variants=False,
):
    if stable_heading:
        from .stable_canonical import trial

        with trial(t):
            return _check(
                t,
                output,
                repeats,
                execution_variants,
                graph_trial,
                graph_all_static,
                graph_only,
                amp_feature_variants,
                subset_variants,
                event_tensor_variants,
            )
    return _check(
        t,
        output,
        repeats,
        execution_variants,
        graph_trial,
        graph_all_static,
        graph_only,
        amp_feature_variants,
        subset_variants,
        event_tensor_variants,
    )


def _check(
    t,
    output,
    repeats,
    execution_variants,
    graph_trial,
    graph_all_static,
    graph_only,
    amp_feature_variants,
    subset_variants,
    event_tensor_variants,
):
    if t.event_scheduler is None:
        raise ValueError("Batching check requires the native event scheduler")
    out = Path(output) / (
        "actor_execution_v30.json"
        if execution_variants
        else "imagination_batching.json"
    )
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
    feature_flag = t.cfg.batch_imagination_amp_features
    subset_flag = t.cfg.collapse_full_event_subsets
    tensor_flag = t.cfg.compile_event_tensors
    flags = (
        t.cfg.batch_imagination_amp,
        t.cfg.batch_imagination_critic,
        t.cfg.reuse_post_event_fk,
        t.cfg.compact_event_pool_updates,
        t.cfg.sparse_actor_bootstrap,
        t.cfg.fuse_actor_core,
    )
    scheduler_attrs = {
        k: (k in t.event_scheduler.__dict__, t.event_scheduler.__dict__.get(k))
        for k in (
            "step",
            "_v30_original_step",
            "_invoke",
            "_v30_original_invoke",
            "event_asset_factory",
            "robot_pose_write_issued",
            "tensor_preparer",
        )
    }
    codec_attrs = {
        k: (
            k in t.event_scheduler.codec.__dict__,
            t.event_scheduler.codec.__dict__.get(k),
        )
        for k in ("decode", "encode", "_v30_original_decode", "_v30_original_encode")
    }
    compiled_handles = {
        k: getattr(t, k)
        for k in (
            "_dynamics",
            "_fk",
            "_v24_heads",
            "_actor_core",
            "_bootstrap_critic_observation",
            "_observations",
        )
    }
    if graph_trial and (not execution_variants or not str(t.device).startswith("cuda")):
        raise ValueError("Graph trial requires CUDA execution comparison")
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
        "actor_core_include_fk": t.cfg.actor_core_include_fk,
        "stable_heading_trial": getattr(t.model, "_stable_heading_trial", False),
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
            variants = (
                [
                    (
                        "reference",
                        flags[0],
                        flags[1],
                        False,
                        False,
                        False,
                        False,
                        False,
                    ),
                    ("fk_reuse", flags[0], flags[1], True, False, False, False, False),
                    ("fk_pool", flags[0], flags[1], True, True, False, False, False),
                    (
                        "sparse_bootstrap",
                        flags[0],
                        flags[1],
                        True,
                        True,
                        True,
                        False,
                        False,
                    ),
                    ("fused_core", flags[0], flags[1], True, True, True, True, False),
                ]
                if execution_variants
                else [
                    ("reference", False, False, *flags[2:], False),
                    ("amp", True, False, *flags[2:], False),
                    ("amp_critic", True, True, *flags[2:], False),
                ]
            )
            if amp_feature_variants:
                if not execution_variants or not flags[0] or graph_trial:
                    raise ValueError(
                        "AMP feature trial requires batched AMP execution mode without Graph trial"
                    )
                variants = [
                    variants[0],
                    ("fk_pool", flags[0], flags[1], True, True, False, False, False),
                    (
                        "amp_features",
                        flags[0],
                        flags[1],
                        True,
                        True,
                        False,
                        False,
                        False,
                    ),
                ]
            if subset_variants:
                if not execution_variants or graph_trial:
                    raise ValueError(
                        "Subset trial requires execution mode without Graph trial"
                    )
                variants = [
                    variants[0],
                    ("fk_pool", flags[0], flags[1], True, True, False, False, False),
                    (
                        "full_subsets",
                        flags[0],
                        flags[1],
                        True,
                        True,
                        False,
                        False,
                        False,
                    ),
                ]
                if amp_feature_variants:
                    variants.append(
                        (
                            "full_subsets_amp_features",
                            flags[0],
                            flags[1],
                            True,
                            True,
                            False,
                            False,
                            False,
                        )
                    )
            if event_tensor_variants:
                if (
                    not execution_variants
                    or graph_trial
                    or subset_variants
                    or amp_feature_variants
                ):
                    raise ValueError(
                        "Event tensor trial requires its own execution comparison"
                    )
                variants = [
                    (label, *flags, False) for label in ("reference", "event_tensors")
                ]
            if graph_only:
                if not graph_trial:
                    raise ValueError("Graph-only comparison requires graph_trial")
                variants = variants[:1]
            if graph_trial:
                variants.append(
                    (
                        "segmented_graph_trial"
                        if graph_all_static
                        else "fused_core_graph_trial",
                        flags[0],
                        flags[1],
                        True,
                        True,
                        not graph_all_static,
                        not graph_all_static,
                        True,
                    )
                )
            for (
                label,
                amp,
                critic,
                reuse_fk,
                compact_pool,
                sparse,
                fused,
                graph,
            ) in variants:
                print(f"[V30 Actor] start {label}", flush=True)
                t.cfg.batch_imagination_amp_features = (
                    (label in ("amp_features", "full_subsets_amp_features"))
                    if amp_feature_variants
                    else feature_flag
                )
                t.cfg.collapse_full_event_subsets = (
                    (label in ("full_subsets", "full_subsets_amp_features"))
                    if subset_variants
                    else subset_flag
                )
                t.cfg.compile_event_tensors = (
                    (label == "event_tensors") if event_tensor_variants else tensor_flag
                )
                t.cfg.sparse_actor_bootstrap = sparse
                t.cfg.fuse_actor_core = fused
                t.cfg.batch_imagination_amp, t.cfg.batch_imagination_critic = (
                    amp,
                    critic,
                )
                t.cfg.reuse_post_event_fk, t.cfg.compact_event_pool_updates = (
                    reuse_fk,
                    compact_pool,
                )
                if execution_variants:
                    from .events_v24.execution_v30 import configure

                    configure(
                        t.event_scheduler,
                        reuse_fk,
                        compact_pool,
                        t.cfg.collapse_full_event_subsets,
                    )
                from .events_v24.tensors_v30 import configure as configure_tensors

                configure_tensors(
                    t.event_scheduler,
                    t.cfg.compile_event_tensors,
                    t.cfg.compile_backend,
                )
                for name, handle in compiled_handles.items():
                    setattr(t, name, handle)
                if fused:
                    t._actor_core = torch.compile(
                        t.actor_physical_core,
                        fullgraph=True,
                        dynamic=False,
                        backend=t.cfg.compile_backend,
                        **(
                            {"mode": "reduce-overhead" if graph else "default"}
                            if t.cfg.compile_backend == "inductor"
                            else {}
                        ),
                    )
                elif execution_variants:
                    t._actor_core = None
                if sparse:
                    t._bootstrap_critic_observation = torch.compile(
                        t.critic_observation,
                        fullgraph=True,
                        dynamic=False,
                        backend=t.cfg.compile_backend,
                    )
                if graph and graph_all_static:
                    t._dynamics = torch.compile(
                        t.model.predict_physical,
                        fullgraph=True,
                        dynamic=False,
                        mode="reduce-overhead",
                    )
                    t._observations = torch.compile(
                        t.imagined_observations,
                        fullgraph=True,
                        dynamic=False,
                        mode="reduce-overhead",
                    )
                    t._fk = torch.compile(
                        t.bridge.fk,
                        fullgraph=True,
                        dynamic=False,
                        mode="reduce-overhead",
                    )
                    t._v24_heads = torch.compile(
                        t.v24_packed_reward_termination,
                        fullgraph=True,
                        dynamic=False,
                        mode="reduce-overhead",
                    )
                restore()
                current = {}
                if devices:
                    torch.cuda.reset_peak_memory_stats(t.device)
                cold_begin = timestamp(t.device)
                if graph:
                    torch.compiler.cudagraph_mark_step_begin()
                fk_count = [0]
                original_fk = t._fk

                def counted_fk(*args, _counter=fk_count, _fk=original_fk, **kwargs):
                    _counter[0] += 1
                    return _fk(*args, **kwargs)

                t._fk = counted_fk
                with freeze(t.model, t.termination, t.prior._disc, t.critic):
                    imagined = t.imagine(t.sample(), trace=True)
                    values = imagined.get("actor_values")
                    if values is None and critic:
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
                t._fk = original_fk
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
                    if graph:
                        torch.compiler.cudagraph_mark_step_begin()
                    with freeze(t.model, t.termination, t.prior._disc, t.critic):
                        begin = timestamp(t.device)
                        imag = t.imagine(t.sample())
                        values = imag.get("actor_values")
                        if values is None and critic:
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
                    if graph:
                        torch.compiler.cudagraph_mark_step_begin()
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
                        if k != "actor_values":
                            compare(f"{label}/{k}", value, expected[k])
                report["variants"][label] = {
                    "cold_diagnostic_includes_compile_and_CPU_snapshot_seconds": cold_seconds,
                    "explicit_fk_calls_per_horizon": fk_count[0],
                    "graph_trial": graph,
                    "batch_amp_features": t.cfg.batch_imagination_amp_features,
                    "collapse_full_event_subsets": t.cfg.collapse_full_event_subsets,
                    "compile_event_tensors": t.cfg.compile_event_tensors,
                    "graph_scope": "original_partitions"
                    if graph and graph_all_static
                    else "core"
                    if graph
                    else None,
                    "warmup_update": rows[0],
                    "steady": rows[1:],
                    "coverage": coverage,
                    "gradient_norm": float(current["gradient"].norm()),
                    "gradient_cosine": float(
                        torch.nn.functional.cosine_similarity(
                            current["gradient"].double(),
                            expected["gradient"].double(),
                            dim=0,
                        )
                    ),
                    "median_actor_seconds": statistics.median(
                        row["time/actor_update_seconds"] for row in rows[1:]
                    ),
                    "median_critic_seconds": statistics.median(
                        row["time/critic_update_seconds"] for row in rows[1:]
                    ),
                }
                print(
                    f"[V30 Actor] done {label}: {report['variants'][label]['median_actor_seconds']:.6f}s",
                    flush=True,
                )
                out.write_text(json.dumps(report, indent=2))
                if graph:
                    restore()
                    torch.compiler.cudagraph_mark_step_begin()
                    with torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ]
                    ) as profiler:
                        t.update_actor()
                    events = profiler.key_averages()
                    report["variants"][label]["cuda_graph_launch_events"] = sum(
                        e.count for e in events if "cudagraphlaunch" in e.key.lower()
                    )
                    profiler.export_chrome_trace(
                        str(Path(output) / "actor_graph_trial_trace.json")
                    )
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
        t.cfg.batch_imagination_amp_features = feature_flag
        t.cfg.collapse_full_event_subsets = subset_flag
        t.cfg.compile_event_tensors = tensor_flag
        (
            t.cfg.batch_imagination_amp,
            t.cfg.batch_imagination_critic,
            t.cfg.reuse_post_event_fk,
            t.cfg.compact_event_pool_updates,
            t.cfg.sparse_actor_bootstrap,
            t.cfg.fuse_actor_core,
        ) = flags
        for key, (present, value) in scheduler_attrs.items():
            if present:
                setattr(t.event_scheduler, key, value)
            elif key in t.event_scheduler.__dict__:
                delattr(t.event_scheduler, key)
        for key, (present, value) in codec_attrs.items():
            if present:
                setattr(t.event_scheduler.codec, key, value)
            elif key in t.event_scheduler.codec.__dict__:
                delattr(t.event_scheduler.codec, key)
        for key, value in compiled_handles.items():
            setattr(t, key, value)
        t.event_scheduler.rng = rng
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        out.write_text(json.dumps(report, indent=2))
    return report
