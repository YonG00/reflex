"""Diagnostic-only eager regions; no tensor/gradient operations."""

from contextlib import nullcontext

import torch


def region(trainer, name):
    if getattr(trainer, "_v31_regions", False):
        return torch.profiler.record_function("v31/" + name)
    return nullcontext()
