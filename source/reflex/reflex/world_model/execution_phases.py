"""Optional coarse host/CUDA-event timeline, without per-module synchronization."""

from contextlib import nullcontext


def phase(trainer, name):
    recorder = getattr(trainer, "_v33_phase_recorder", None)
    return nullcontext() if recorder is None else recorder.phase(name)
