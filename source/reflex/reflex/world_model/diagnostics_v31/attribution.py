"""Non-overlapping CPU self time and directly correlated GPU kernel work.

Backward origin uses autograd sequence/fwd_thread metadata. Unmatched work is
reported explicitly, never distributed as guessed module percentages.
"""

from collections import defaultdict

PREFIX = "v31/"


def ancestry(event):
    while event is not None:
        yield event
        event = event.cpu_parent


def forward_scope(event):
    return next(
        (e.name[len(PREFIX) :] for e in ancestry(event) if e.name.startswith(PREFIX)),
        None,
    )


def union_duration(intervals):
    total = 0.0
    end = None
    for a, b in sorted(intervals):
        if end is None or a > end:
            total += b - a
        else:
            total += max(0.0, b - end)
        end = b if end is None else max(end, b)
    return total


def activity_timeline(events):
    root = next(
        (
            e
            for e in events
            if e.name == "v31/complete_update" and getattr(e, "time_range", None)
        ),
        None,
    )
    if root is None:
        return None
    begin, end = root.time_range.start, root.time_range.end
    intervals = []
    for e in events:
        if str(e.device_type).split(".")[-1] == "CUDA" and getattr(
            e, "time_range", None
        ):
            a, b = max(begin, e.time_range.start), min(end, e.time_range.end)
            if b > a:
                intervals.append((a, b))
    if not intervals:
        return None
    active = union_duration(intervals)
    return {
        "root_span_us": end - begin,
        "device_activity_union_us": active,
        "unobserved_device_gap_us": max(0.0, end - begin - active),
        "device_activity_count": len(intervals),
        "note": "Union across captured CUDA activities includes transfers; not SM utilization. Gaps require CPU/runtime correlation and graph-tracing coverage checks.",
    }


def summarize(events):
    events = list(events)
    sequence = defaultdict(set)
    for e in events:
        owner = forward_scope(e)
        if (
            owner
            and not any(
                p.scope == 1 or p.name.startswith("autograd::engine::evaluate_function")
                for p in ancestry(e)
            )
            and not e.name.startswith((PREFIX, "autograd::", "profiler::"))
            and not getattr(e, "is_user_annotation", False)
            and e.sequence_nr >= 0
        ):
            sequence[(e.thread, e.sequence_nr)].add(owner)

    def owner(event):
        for parent in ancestry(event):
            if (
                parent.scope == 1
                or parent.name.startswith("autograd::engine::evaluate_function")
            ) and parent.sequence_nr >= 0:
                matches = sequence.get((parent.fwd_thread, parent.sequence_nr), set())
                if len(matches) == 1:
                    return "backward/" + next(iter(matches))
                return "backward/unmatched_or_shared"
        return forward_scope(event) or "unattributed"

    groups = defaultdict(
        lambda: {
            "cpu_self_us": 0.0,
            "kernel_work_us": 0.0,
            "direct_kernel_count": 0,
            "cpu_operator_count": 0,
        }
    )
    for e in events:
        if str(e.device_type).split(".")[-1] != "CPU" or e.is_async:
            continue
        row = groups[owner(e)]
        row["cpu_self_us"] += max(0.0, e.self_cpu_time_total)
        row["cpu_operator_count"] += 1
        for kernel in e.kernels:
            row["kernel_work_us"] += kernel.duration
            row["direct_kernel_count"] += 1
    return {
        "groups": dict(groups),
        "device_timeline": activity_timeline(events),
        "notes": [
            "CPU self sums across threads are not elapsed wall time.",
            "GPU kernel duration sums are work, not exclusive critical-path time; streams may overlap.",
            "Backward attribution uses sequence_nr plus fwd_thread; unmatched/shared work stays separate.",
            "Chrome trace remains authoritative for GPU gaps, runtime synchronization and memcpy direction.",
        ],
    }
