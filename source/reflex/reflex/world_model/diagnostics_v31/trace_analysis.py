"""Post-process Chrome trace activities, excluding GPU user annotation ranges."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from .attribution import union_duration


def analyze(events):
    root = next(
        e
        for e in events
        if e.get("cat") == "user_annotation"
        and e.get("name") == "v31/complete_update"
        and e.get("ph") == "X"
    )
    begin, end = root["ts"], root["ts"] + root["dur"]
    intervals, kernels = [], []
    copies, copy_bytes = Counter(), Counter()
    runtime = defaultdict(lambda: {"calls": 0, "cpu_duration_us": 0.0})
    kernel_names = defaultdict(lambda: {"calls": 0, "duration_us": 0.0})
    categories = Counter()
    for e in events:
        if e.get("ph") != "X" or not begin <= e.get("ts", -1) < end:
            continue
        category, name = e.get("cat"), e["name"]
        categories[category] += 1
        if category in ("kernel", "gpu_memcpy", "gpu_memset"):
            intervals.append((e["ts"], min(end, e["ts"] + e["dur"])))
        if category == "kernel":
            kernels.append(e["dur"])
            kernel_names[name]["calls"] += 1
            kernel_names[name]["duration_us"] += e["dur"]
        elif category == "gpu_memcpy":
            copies[name] += 1
            copy_bytes[name] += e.get("args", {}).get("bytes", 0)
        elif category in ("cuda_runtime", "cuda_driver"):
            runtime[name]["calls"] += 1
            runtime[name]["cpu_duration_us"] += e["dur"]
    active = union_duration(intervals)
    return {
        "root_span_us": end - begin,
        "device_activity_union_us": active,
        "unobserved_device_gap_us": max(0.0, end - begin - active),
        "real_device_activity_count": len(intervals),
        "kernel_count": len(kernels),
        "kernel_duration_sum_us": sum(kernels),
        "kernel_le_5us_count": sum(v <= 5 for v in kernels),
        "memcpy_counts": dict(copies),
        "memcpy_bytes": dict(copy_bytes),
        "runtime": dict(runtime),
        "trace_categories": dict(categories),
        "largest_kernel_work": sorted(
            ({"name": k, **v} for k, v in kernel_names.items()),
            key=lambda v: v["duration_us"],
            reverse=True,
        )[:30],
        "notes": [
            "Only kernel/gpu_memcpy/gpu_memset count as device activities; gpu_user_annotation ranges are excluded.",
            "Profiler inflates host/runtime overhead. These durations are not unprofiled module wall times.",
            "Activity union is not SM utilization. Missing graph internals may create unobserved gaps.",
        ],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("trace")
    p.add_argument("output")
    a = p.parse_args()
    output = Path(a.output)
    if output.exists():
        raise FileExistsError(output)
    result = analyze(json.loads(Path(a.trace).read_text())["traceEvents"])
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "largest_kernel_work"}))


if __name__ == "__main__":
    main()
