"""Reconstruct the Oracle policy observation at every imagined control step."""

import torch
from torch import nn
from .physics import apply, inverse, mul, rot6


class OracleObservation(nn.Module):
    def __init__(
        self,
        fk,
        motion_q,
        motion_dq,
        anchor_pos,
        anchor_quat,
        default_q,
        default_dq,
        anchor_name,
        reference_arm_columns,
        reference_down,
    ):
        super().__init__()
        self.fk, self.anchor_name = fk, anchor_name
        for name, tensor in locals().copy().items():
            if name in (
                "motion_q",
                "motion_dq",
                "anchor_pos",
                "anchor_quat",
                "default_q",
                "default_dq",
                "reference_arm_columns",
                "reference_down",
            ):
                self.register_buffer(name, tensor.detach().clone())

    def reference(self, phase, yaw):
        # Resampling/teleportation at reference exhaustion is an external boundary.
        if bool(((phase < 0) | (phase >= len(self.motion_q))).any()):
            raise ValueError("Imagination crossed reference resampling")
        z = torch.zeros_like(yaw)
        yq = torch.stack((torch.cos(yaw / 2), z, z, torch.sin(yaw / 2)), -1)
        return (
            self.motion_q[phase],
            self.motion_dq[phase],
            apply(yq, self.anchor_pos[phase]),
            mul(yq, self.anchor_quat[phase]),
        )

    def forward(
        self,
        s,
        previous_action,
        phase,
        yaw,
        launched,
        dimensions,
        mass,
        default_q=None,
        default_dq=None,
    ):
        positions, quats = self.fk(s)
        rq, rdq, rp, rquat = self.reference(phase, yaw)
        rq = rq.clone()
        idx = self.reference_arm_columns
        rq[..., idx] = torch.where(
            launched[..., None].bool(), rq[..., idx], self.reference_down
        )
        ai = inverse(quats[self.anchor_name])
        ri = inverse(s[..., 3:7])
        return torch.cat(
            (
                rq,
                rdq,
                apply(ai, rp - positions[self.anchor_name]),
                mul6(ai, rquat),
                apply(ri, s[..., 36:39]),
                apply(ri, s[..., 39:42]),
                s[..., 7:36] - (self.default_q if default_q is None else default_q),
                s[..., 42:71] - (self.default_dq if default_dq is None else default_dq),
                previous_action,
                apply(ri, s[..., 71:74] - s[..., :3]),
                apply(ri, s[..., 78:81]),
                mul6(ri, s[..., 74:78]),
                apply(ri, s[..., 81:84]),
                dimensions,
                mass,
            ),
            -1,
        )


def mul6(a, b):
    return rot6(mul(a, b))
