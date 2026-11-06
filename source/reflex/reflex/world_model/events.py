"""Discontinuities caused by native state writes, not by physical acceleration."""

import torch


def physical_write_mask(before, after, atol=1e-7):
    finite = torch.isfinite(before).all(-1) & torch.isfinite(after).all(-1)
    return ~finite | ((before - after).abs().amax(-1) > atol)


def macro_step_target(
    physical, post_control, done, reference_jump, before_mode, after_mode
):
    """Learn ordinary control writes, never a new episode/reference initializer."""
    reset = done | reference_jump
    return (
        torch.where(reset[:, None], physical, post_control),
        torch.where(reset, before_mode.bool(), after_mode.bool()),
        reset,
    )
