"""Real collection attribution; never performs WM/Actor/AMP optimizer updates."""

import json
import time
from pathlib import Path

import torch


def check(t, adapter, output, steps=24):
    cache = adapter.property_cache
    if cache is None:
        raise ValueError("Instantiate property cache to compare both paths")
    variants = [
        ("full_checks", 1, False, False),
        ("sampled_checks", 100, False, False),
        ("native_sampled_checks", 100, False, True),
        ("cached_native_sampled_checks", 100, True, True),
    ]
    records = []
    counters = (t.actor_updates, t.model_updates, t.termination_updates)
    original = (
        adapter.parity_every,
        adapter.property_cache,
        adapter.native_collection_inputs,
        t.cfg.native_collection_inputs,
    )
    try:
        for name, every, cached, native in variants + variants[::-1]:
            adapter.native_collection_inputs = t.cfg.native_collection_inputs = native
            adapter.parity_every = every
            adapter.property_cache = cache if cached else None
            for _ in range(4):
                t.collect(adapter)
            torch.cuda.synchronize()
            before = t.real_transitions
            start = time.perf_counter()
            for _ in range(steps):
                t.collect(adapter)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            row = {
                "variant": name,
                "steps": steps,
                "envs": adapter.env.num_envs,
                "seconds": seconds,
                "seconds_per_vector_step": seconds / steps,
                "real_transitions_per_second": (t.real_transitions - before) / seconds,
                "parity_every": every,
                "cache": cached,
                "native_inputs": native,
            }
            if cached:
                packed = adapter.context()
                adapter.property_cache = None
                fresh = adapter.context()
                adapter.property_cache = cache
                torch.testing.assert_close(packed, fresh, atol=0, rtol=0)
                row["context_max_error"] = float((packed - fresh).abs().max())
            records.append(row)
            print(json.dumps(row), flush=True)
        assert counters == (t.actor_updates, t.model_updates, t.termination_updates)
        result = {
            "completed": True,
            "records": records,
            "optimizer_updates": 0,
            "native_parity_checked_transitions": adapter.task_parity_steps,
            "cache_reads": cache.reads,
            "cache_hits": cache.hits,
            "cache_invalidations": cache.invalidations,
            "limitation": "Sequential real trajectories, reversed order repeated; not identical simulator snapshots.",
        }
        Path(output).mkdir(parents=True, exist_ok=True)
        (Path(output) / "collect_timing.json").write_text(json.dumps(result, indent=2))
        return result
    finally:
        (
            adapter.parity_every,
            adapter.property_cache,
            adapter.native_collection_inputs,
            t.cfg.native_collection_inputs,
        ) = original
