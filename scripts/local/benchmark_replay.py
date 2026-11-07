#!/usr/bin/env python3
"""Replay metadata benchmark, or exact checkpoint schema/memory planning.

Synthetic metadata timings are NOT end-to-end training or PhysX throughput.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from reflex.world_model.replay import Replay


def legacy_indices(r, horizon):
    valid = r.ordered("valid").clone()
    for offset in range(horizon):
        valid &= torch.roll(r.ordered("valid"), -offset, 0)
        if offset < horizon - 1:
            valid &= ~torch.roll(
                r.ordered("boundary") | r.ordered("terminated") | r.ordered("timeout"),
                -offset,
                0,
            )
    if horizon > 1:
        valid[-horizon + 1 :] = False
    return valid.nonzero()


def benchmark(r, repeats=10):
    def sync():
        if str(r.device).startswith("cuda"):
            torch.cuda.synchronize(r.device)

    expected = legacy_indices(r, 8)
    torch.testing.assert_close(r.indices(8), expected, rtol=0, atol=0)
    records = {}

    def uncached_choices():
        r.cache_indices = False
        try:
            return r._choices(False, variable=True)
        finally:
            r.cache_indices = True

    for label, call in (
        ("legacy_h8", lambda: legacy_indices(r, 8)),
        ("uncached_h8", lambda: r._indices(8)),
        ("cached_h8", lambda: r.indices(8)),
        ("uncached_variable_pool", uncached_choices),
        ("cached_variable_pool", lambda: r._choices(False, variable=True)),
    ):
        call()
        times = []
        for _ in range(repeats):
            sync()
            start = time.perf_counter()
            call()
            sync()
            times.append(time.perf_counter() - start)
        records[label + "_median_seconds"] = statistics.median(times)
    # Each real collection invalidates once. Report rebuild separately from hits.
    r._index_cache.clear()
    sync()
    start = time.perf_counter()
    r.indices(8)
    sync()
    records["cache_rebuild_seconds"] = time.perf_counter() - start
    records["indices_exact_equal"] = True
    records["valid_h8_windows"] = len(expected)
    records["synthetic_metadata_only"] = "state" not in r.data
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--envs", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--schema-only", action="store_true")
    parser.add_argument("--target-capacity", type=int, default=1000)
    parser.add_argument("--target-envs", type=int)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if state.get("replay") is None:
            parser.error("Checkpoint has no replay; collect fresh schema first")
        saved = state["replay"]
        if args.schema_only:
            fields = {
                k: v[0].numel() * v.element_size() for k, v in saved["data"].items()
            }
            source_envs = saved["data"]["valid"].shape[1]
            target_envs = args.target_envs or source_envs
            globals_ = set(saved.get("global_fields", ()))
            projected = {
                k: n if k in globals_ else n // source_envs * target_envs
                for k, n in fields.items()
            }
            result = {
                "source": str(args.checkpoint),
                "source_envs": source_envs,
                "target_envs": target_envs,
                "bytes_per_vector_step": sum(fields.values()),
                "target_capacity": args.target_capacity,
                "target_payload_gib": sum(projected.values())
                * args.target_capacity
                / 2**30,
                "field_bytes_per_vector_step": fields,
                "target_field_bytes_per_vector_step": projected,
                "excludes": "PhysX, optimizer, BPTT, index cache and allocator",
            }
        else:
            r = Replay(saved["capacity"], args.device, saved.get("global_fields", ()))
            r.load_state_dict(saved)
            result = benchmark(r, args.repeats)
    else:
        r = Replay(args.steps, args.device)
        torch.manual_seed(147)
        shape = (args.steps, args.envs)
        r.data = {
            "valid": torch.rand(shape, device=args.device) > 0.01,
            "boundary": torch.rand(shape, device=args.device) < 0.005,
            "terminated": torch.zeros(shape, device=args.device, dtype=torch.bool),
            "timeout": torch.zeros(shape, device=args.device, dtype=torch.bool),
        }
        r.count, r.pointer = args.steps, 17 % args.steps
        result = {
            "device": args.device,
            "steps": args.steps,
            "envs": args.envs,
            "threads": args.threads,
            **benchmark(r, args.repeats),
        }
    payload = json.dumps(result, indent=2)
    if args.out:
        with args.out.open("x") as f:  # Never overwrite earlier experiments.
            f.write(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
