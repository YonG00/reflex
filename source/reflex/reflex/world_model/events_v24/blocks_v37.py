"""Experimental functional block writes; native scheduling/RNG stay unchanged."""

import torch

from .writes import WritePlan

# Exactly the existing state layout, without quaternion/frame conversion.
WIDTHS = (7, 29, 6, 29, 7, 6)
RANGES = ((0, 7), (7, 36), (36, 42), (42, 71), (71, 78), (78, 84))


class BlockWritePlan(WritePlan):
    """Update only a written root/joint block, materialize full plans on demand.

    Irregular columns and duplicate row writes retain the original recorder.
    No detection of CUDA index contents and no value-based write inference.
    """

    needs_before = False

    def __init__(self, physical):
        self._set(
            torch.zeros_like(physical),
            torch.zeros_like(physical, dtype=torch.bool),
        )

    def _set(self, targets, mask):
        self._blocks = list(targets.split(WIDTHS, -1))
        self._masks = list(mask.split(WIDTHS, -1))
        self._targets, self._mask = targets, mask

    @property
    def targets(self):
        if self._targets is None:
            self._targets = torch.cat(self._blocks, -1)
        return self._targets

    @property
    def mask(self):
        if self._mask is None:
            self._mask = torch.cat(self._masks, -1)
        return self._mask

    def record(self, before, requested, env_ids, slots, intervention=None):
        # Only full-batch, known Python root/joint block writes are specialized.
        # Tensor-valued, partial/permuted/duplicate rows keep native index_put.
        block = next(
            (
                i
                for i, (a, b) in enumerate(RANGES)
                if isinstance(slots, (list, tuple)) and list(slots) == list(range(a, b))
            ),
            None,
        )
        if env_ids is not None or block is None:
            base = WritePlan.__new__(WritePlan)
            # Functional isolation: callbacks may still hold earlier plan views.
            base.targets, base.mask = self.targets.clone(), self.mask.clone()
            base.record(before, requested, env_ids, slots, intervention)
            self._set(base.targets, base.mask)
            return
        active = (
            torch.ones_like(requested, dtype=torch.bool)
            if intervention is None
            else intervention.expand_as(requested)
        )
        self._blocks[block] = torch.where(active, requested, self._blocks[block])
        self._masks[block] = self._masks[block] | active
        self._targets = self._mask = None

    def select(self, ids):
        child = type(self).__new__(type(self))
        child._set(self.targets[ids].clone(), self.mask[ids].clone())
        return child

    def merge(self, ids, other):
        self._set(
            self.targets.index_copy(0, ids, other.targets),
            self.mask.index_copy(0, ids, other.mask),
        )
