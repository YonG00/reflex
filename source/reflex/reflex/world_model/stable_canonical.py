"""Explicit benchmark-only exact-zero de-yawed R10; no checkpoint migration."""

from contextlib import contextmanager

import torch

from .models import CompactContextWorldModel
from .physics import stable_canonical


def stable_features(self, state, target, physical_context, event):
    if physical_context.shape[-1] != self.full_context_dim:
        raise ValueError("Full analytical physical context is required")
    encoded = stable_canonical(state)
    context = physical_context.index_select(-1, self.wm_context_columns)
    return torch.cat((encoded, target, context, event), -1)


@contextmanager
def trial(trainer):
    model = trainer.model
    if not isinstance(model, CompactContextWorldModel):
        raise TypeError("Stable heading trial currently requires compact WM")
    present = "features" in model.__dict__
    previous = model.__dict__.get("features")
    model.features = stable_features.__get__(model)
    model._stable_heading_trial = True
    try:
        yield
    finally:
        if present:
            model.features = previous
        else:
            del model.features
        del model._stable_heading_trial
