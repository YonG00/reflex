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
        in_bounds = ((phase >= 0) & (phase < len(self.motion_q))).all()
        if torch.compiler.is_compiling():
            torch._assert_async(in_bounds, "Imagination crossed reference resampling")
        elif not bool(in_bounds):
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
        fk_result=None,
    ):
        positions, quats = self.fk(s) if fk_result is None else fk_result
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


class PrivilegedObservation(nn.Module):
    """Legacy292 or explicit305: keep 14 bodies and optionally every Actor column."""

    def __init__(self, bridge, tracked_names, include_all_actor=False):
        super().__init__()
        if len(tracked_names) != 14:
            raise ValueError(
                "292-D critic requires exactly the original 14 tracked bodies"
            )
        self.include_all_actor = include_all_actor
        self.bridge = bridge
        self.tracked_names = tuple(tracked_names)

    def forward(self, *args, fk_result=None, actor_result=None):
        state = args[0]
        poses = self.bridge.fk(state) if fk_result is None else fk_result
        actor = (
            self.bridge(*args, fk_result=poses)
            if actor_result is None
            else actor_result
        )
        p, q = poses
        anchor = self.bridge.anchor_name
        ai = inverse(q[anchor])
        body_p = torch.stack(
            [apply(ai, p[n] - p[anchor]) for n in self.tracked_names], -2
        ).flatten(-2)
        # A frame relative to itself is exactly identity. Expanding conj(q)*q
        # creates cancellation noise (and spurious gradients) under GPU FMA;
        # Adam amplifies that noise on these otherwise constant critic columns.
        identity6 = state.new_tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]).expand(
            *state.shape[:-1], 6
        )
        body_q = torch.stack(
            [
                identity6 if n == anchor else rot6(mul(ai, q[n]))
                for n in self.tracked_names
            ],
            -2,
        ).flatten(-2)
        original = torch.cat((actor[..., :67], body_p, body_q, actor[..., 67:166]), -1)
        return (
            torch.cat((original, actor[..., 166:]), -1)
            if self.include_all_actor
            else original
        )
