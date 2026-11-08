"""Complete-chain source-site profiler and unprofiled coarse execution timeline."""

import gc
import json
import sys
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from pathlib import Path

import torch

from .diagnostics_v31.attribution import ancestry
from .diagnostics_v31.full import check as full_check
from .diagnostics_v31.full import compare, compile_counts, count_delta, distribution
from .diagnostics_v31.state import Snapshot, parameter_result
from .events_v24.writes import PlannedAsset, WritePlan
from .timing import timestamp

PREFIX = "v33/site/"
CALLBACK = "v33/callback/"


def boolean_index(value):
    return (torch.is_tensor(value) and value.dtype == torch.bool) or (
        isinstance(value, tuple) and any(boolean_index(v) for v in value)
    )


@contextmanager
def source_sites(include_cpu=False):
    """Only active inside the eager scheduler; never around compiled plant/BPTT."""
    saved = {}
    for name in (
        "nonzero",
        "item",
        "__bool__",
        "__int__",
        "__float__",
        "__getitem__",
        "__setitem__",
    ):
        original = getattr(torch.Tensor, name)
        saved[name] = original

        def call(value, *args, _name=name, _original=original, **kwargs):
            selected = value.device.type == "cuda" or include_cpu
            if _name in ("__getitem__", "__setitem__"):
                selected = selected and bool(args) and boolean_index(args[0])
            if not selected:
                return _original(value, *args, **kwargs)
            frame = sys._getframe(1)
            location = frame.f_code.co_filename + ":" + str(frame.f_lineno)
            del frame
            with torch.profiler.record_function(PREFIX + _name + "|" + location):
                return _original(value, *args, **kwargs)

        setattr(torch.Tensor, name, call)
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(torch.Tensor, name, value)


@contextmanager
def detailed_scopes(t):
    if t.cfg.compile_event_tensors:
        raise ValueError(
            "Source-site hooks require eager event tensors; do not alter compiled plant execution"
        )
    s = t.event_scheduler
    saved = []

    def install(obj, name, function):
        saved.append((obj, name, name in obj.__dict__, obj.__dict__.get(name)))
        setattr(obj, name, function)

    original_step, original_invoke = s.step, s._invoke
    depth = 0

    def step(*args, **kwargs):
        nonlocal depth
        depth += 1
        try:
            with (
                source_sites(include_cpu=str(t.device) == "cpu")
                if depth == 1
                else nullcontext()
            ):
                return original_step(*args, **kwargs)
        finally:
            depth -= 1

    def invoke(term, *args, **kwargs):
        def callback(*cb_args, **cb_kwargs):
            with torch.profiler.record_function(CALLBACK + term.name):
                return term.function(*cb_args, **cb_kwargs)

        with torch.profiler.record_function("v33/invoke/" + term.name):
            return original_invoke(replace(term, function=callback), *args, **kwargs)

    try:
        install(s, "step", step)
        install(s, "_invoke", invoke)
        install(s, "_v33_regions", True)
        for cls, name, label in (
            (WritePlan, "record", "plan_record"),
            (WritePlan, "select", "plan_select"),
            (WritePlan, "merge", "plan_merge"),
            (PlannedAsset, "_refresh", "asset_refresh"),
        ):
            original = getattr(cls, name)

            def scoped(*args, _original=original, _label=label, **kwargs):
                with torch.profiler.record_function("v33/" + _label):
                    return _original(*args, **kwargs)

            install(cls, name, scoped)
        yield
    finally:
        for obj, name, present, value in reversed(saved):
            if present:
                setattr(obj, name, value)
            else:
                delattr(obj, name)


def source_summary(events):
    events = list(events)

    def forward(e):
        return next(
            (p.name for p in ancestry(e) if p.name.startswith((PREFIX, CALLBACK))), None
        )

    sequence = defaultdict(set)
    for e in events:
        label = forward(e)
        if (
            label
            and e.sequence_nr >= 0
            and not getattr(e, "is_user_annotation", False)
            and not any(
                p.scope == 1 or p.name.startswith("autograd::engine::")
                for p in ancestry(e)
            )
        ):
            sequence[(e.thread, e.sequence_nr)].add(label)

    def owner(e):
        for p in ancestry(e):
            if (
                p.scope == 1 or p.name.startswith("autograd::engine::")
            ) and p.sequence_nr >= 0:
                labels = sequence.get((p.fwd_thread, p.sequence_nr), ())
                return (
                    "backward/" + next(iter(labels))
                    if len(labels) == 1
                    else "backward/unmatched"
                )
        return forward(e) or "outside_native_or_unmatched"

    groups = defaultdict(
        lambda: {
            "stream_sync_calls": 0,
            "stream_sync_cpu_us": 0.0,
            "kernel_work_us": 0.0,
            "direct_device_activities": 0,
            "nonzero_calls": 0,
            "scalar_read_calls": 0,
        }
    )
    for e in events:
        if str(e.device_type).split(".")[-1] != "CPU" or e.is_async:
            continue
        g = groups[owner(e)]
        if e.name == "cudaStreamSynchronize":
            g["stream_sync_calls"] += 1
            g["stream_sync_cpu_us"] += e.self_cpu_time_total
        g["nonzero_calls"] += e.name == "aten::nonzero"
        g["scalar_read_calls"] += e.name == "aten::_local_scalar_dense"
        for k in e.kernels:
            g["kernel_work_us"] += k.duration
            g["direct_device_activities"] += 1
    return {
        "groups": dict(groups),
        "notes": [
            "Stream-sync durations are observed under Profiler, not causal unprofiled speed gains.",
            "Python source hooks cover eager methods; scripted/C++ operations can remain callback-level or unmatched.",
            "Backward attribution uses forward sequence/thread; ambiguous origins are not guessed.",
        ],
    }


class PhaseRecorder:
    def __init__(self, device):
        self.device, self.rows, self.gc_rows, self.active = str(device), [], [], []
        self.gc_started = None

    @contextmanager
    def phase(self, name):
        start, end = (
            (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            if self.device.startswith("cuda")
            else (None, None)
        )
        self.active.append(name)
        begin = time.perf_counter()
        if start is not None:
            start.record()
        try:
            yield
        finally:
            if end is not None:
                end.record()
            self.rows.append((name, time.perf_counter() - begin, start, end))
            self.active.pop()

    def gc_callback(self, event, info):
        if event == "start":
            self.gc_started = (
                time.perf_counter(),
                self.active[-1] if self.active else "update_unscoped",
                info["generation"],
            )
        elif self.gc_started is not None:
            begin, phase, generation = self.gc_started
            self.gc_rows.append(
                {
                    "phase": phase,
                    "generation": generation,
                    "seconds": time.perf_counter() - begin,
                }
            )
            self.gc_started = None

    def result(self):
        # Caller already completed the ordinary update's final synchronization.
        return {
            "phases": [
                {
                    "name": n,
                    "host_span_seconds": host,
                    "cuda_span_seconds_includes_gaps": a.elapsed_time(b) / 1000
                    if a is not None
                    else None,
                }
                for n, host, a, b in self.rows
            ],
            "gc": self.gc_rows,
        }


def phase_check(t, output, repeats):
    snap = Snapshot(t)
    present, saved = (
        "_v33_phase_recorder" in t.__dict__,
        t.__dict__.get("_v33_phase_recorder"),
    )
    result = {
        "completed": False,
        "reference": [],
        "recorded": [],
        "note": "Coarse CUDA event spans include idle gaps; host spans overlap GPU execution; do not add them.",
    }
    reference = None
    try:
        before = compile_counts()
        for _ in range(repeats):
            for recorded in (False, True):
                snap.reset()
                recorder = PhaseRecorder(t.device)
                t._v33_phase_recorder = recorder if recorded else None
                if recorded:
                    gc.callbacks.append(recorder.gc_callback)
                try:
                    begin = timestamp(t.device)
                    metrics = t.update_actor()
                    elapsed = timestamp(t.device) - begin
                finally:
                    if recorded:
                        gc.callbacks.remove(recorder.gc_callback)
                if reference is None:
                    reference = parameter_result(t)
                row = {"wall_seconds": elapsed, "metrics": metrics}
                if recorded:
                    row.update(recorder.result())
                    row["numeric"] = compare(reference, parameter_result(t))
                result["recorded" if recorded else "reference"].append(row)
        result["compile_delta"] = count_delta(compile_counts(), before)
        result["completed"] = True
        result["numeric_passed"] = all(
            x["numeric"]["passed"] for x in result["recorded"]
        )
        for name in ("reference", "recorded"):
            result[name + "_wall"] = distribution(
                [x["wall_seconds"] for x in result[name]]
            )
    finally:
        if present:
            t._v33_phase_recorder = saved
        else:
            del t._v33_phase_recorder
        snap.restore()
        Path(output).write_text(json.dumps(result, indent=2))
    return result


def check(t, output, repeats=20, warmups=2):
    out = Path(output)
    report = full_check(
        t,
        out / "full",
        repeats=repeats,
        warmups=warmups,
        extra_scopes=detailed_scopes,
        extra_summary=source_summary,
    )
    phases = phase_check(t, out / "phases.json", repeats)
    result = {
        "completed": report["completed"] and phases["completed"],
        "numeric_passed": report["numeric_passed"] and phases["numeric_passed"],
        "attribution_usable": report["attribution_usable"],
        "checkpoint_saved": False,
    }
    (out / "execution_study_v33.json").write_text(json.dumps(result, indent=2))
    return result
