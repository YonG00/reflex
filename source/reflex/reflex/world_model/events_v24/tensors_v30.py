"""Pure tensor facade preparation; callbacks and random scheduling stay eager."""

import torch

from ..task import HISTORY_FIELDS


def prepare(physical, targets, mask, origins, history, values):
    state = torch.where(mask.clone(), targets, physical)
    robot = (
        state[:, :3] + origins,
        state[:, 3:7].clone(),
        state[:, 36:39].clone(),
        state[:, 39:42].clone(),
        state[:, 7:36].clone(),
        state[:, 42:71].clone(),
    )
    box = (
        state[:, 71:74] + origins,
        state[:, 74:78].clone(),
        state[:, 78:81].clone(),
        state[:, 81:84].clone(),
    )
    counters = tuple(history[:, i].round().long() for i in range(len(HISTORY_FIELDS)))
    return robot, box, counters, {name: value.clone() for name, value in values.items()}


def configure(scheduler, enabled, backend="inductor"):
    codec = scheduler.codec
    for name in ("decode", "encode"):
        original = "_v30_original_" + name
        if not hasattr(codec, original):
            setattr(codec, original, getattr(codec, name))
        method = getattr(codec, original)
        setattr(
            codec,
            name,
            torch.compile(method, fullgraph=True, dynamic=True, backend=backend)
            if enabled
            else method,
        )
    if enabled:
        scheduler.tensor_preparer = torch.compile(
            prepare, fullgraph=True, dynamic=True, backend=backend
        )
    elif hasattr(scheduler, "tensor_preparer"):
        del scheduler.tensor_preparer
