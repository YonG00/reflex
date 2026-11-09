"""Bound temporary critic activations without changing the imagined batch."""

import torch
from torch.utils.checkpoint import checkpoint


def evaluate(critic, observations, chunk=0, recompute=False):
    if chunk < 0:
        raise ValueError("Critic value chunk must be nonnegative")
    flat = observations.flatten(0, -2)
    size = chunk or len(flat)
    if not size:
        raise ValueError("Empty critic observation batch")
    values = []
    for block in flat.split(size):
        if recompute and torch.is_grad_enabled() and block.requires_grad:
            # Critic is a deterministic frozen MLP during Actor optimization.
            # Keep every input gradient; only recompute hidden activations.
            value = checkpoint(
                critic, block, use_reentrant=False, preserve_rng_state=False
            )
        else:
            value = critic(block)
        values.append(value)
    return torch.cat(values).reshape(observations.shape[:-1])
