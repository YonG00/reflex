"""Opt-in eager execution changes; event callbacks and math remain unchanged."""

import torch

from .events_v24.fast_view import build as minimal_view

# NativeEventScheduler.step uses event_agent for foreign-world timer lookup.
# fast_view.build reads the other three fields and explicitly ignores poses.
MINIMAL_EVENT_FIELDS = ("event_origins", "task_history", "dimensions", "event_agent")


def gather_event_inputs(scheduler, batch, poses, ids, count):
    """Narrow only the audited native minimal facade; custom views fall back."""
    if getattr(scheduler._view, "__func__", None) is minimal_view:
        fields = (
            MINIMAL_EVENT_FIELDS if scheduler.pool_terms else MINIMAL_EVENT_FIELDS[:3]
        )
        return {k: batch[k][ids] for k in fields}, ()
    return (
        {
            k: v[ids] if torch.is_tensor(v) and v.ndim and len(v) == count else v
            for k, v in batch.items()
        },
        tuple({k: v[ids] for k, v in group.items()} for group in poses),
    )


def packed_pose_carry(previous, predicted, valid):
    """Mask once per homogeneous pose group, retaining differentiable views.

    unbind returns all body views in a single autograd node, rather than separate
    slice nodes. Names follow previous, exactly as the original per-body where.
    No mask->host conversion, detach, quaternion math or pose normalization.
    """
    result = []
    for old, new in zip(previous, predicted):
        if not old:
            result.append({})
            continue
        names = tuple(old)
        carried = torch.where(
            valid[:, None, None],
            torch.stack([new[name] for name in names], 1),
            torch.stack([old[name] for name in names], 1),
        )
        result.append(dict(zip(names, carried.unbind(1))))
    return tuple(result)
