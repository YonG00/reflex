"""Atomic progress and cold-call stack diagnostics, outside steady timing."""

import faulthandler
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    os.replace(temporary, path)


class ColdRecorder:
    def __init__(self, progress):
        self.progress, self.stack = progress, []

    @contextmanager
    def phase(self, name):
        begin = time.perf_counter()
        self.stack.append(name)
        self.progress({"operation_stack": list(self.stack), "event": "begin"})
        try:
            yield
        finally:
            self.progress(
                {
                    "operation_stack": list(self.stack),
                    "event": "end",
                    "host_span_seconds": time.perf_counter() - begin,
                }
            )
            self.stack.pop()


@contextmanager
def cold_watch(t, label, progress, stack_path, trace_seconds=0):
    recorder = ColdRecorder(progress)
    changed = []

    def set_attribute(owner, name, value):
        changed.append((owner, name, name in owner.__dict__, owner.__dict__.get(name)))
        setattr(owner, name, value)

    def wrap(name):
        original = getattr(t, name, None)
        if original is None:
            return

        def tracked(*args, **kwargs):
            with recorder.phase(name):
                return original(*args, **kwargs)

        set_attribute(t, name, tracked)

    stack_file = open(stack_path, "a")  # noqa: SIM115 - lifetime spans restored hooks
    try:
        set_attribute(t, "_v33_phase_recorder", recorder)
        # Leave _view untouched: V32 relies on its function identity.
        for name in ("_observations", "_dynamics", "_fk", "_v24_heads"):
            wrap(name)
        if trace_seconds:
            faulthandler.dump_traceback_later(
                trace_seconds, repeat=True, file=stack_file
            )
        with recorder.phase(label):
            yield
    finally:
        if trace_seconds:
            faulthandler.cancel_dump_traceback_later()
        for owner, name, present, value in reversed(changed):
            if present:
                setattr(owner, name, value)
            else:
                delattr(owner, name)
        stack_file.close()
