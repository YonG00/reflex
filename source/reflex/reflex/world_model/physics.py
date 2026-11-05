"""World-frame SI state, wxyz rotations, differentiable URDF FK and control mapping.

State: root p/q/joints/v_world/w_world/dq (71), box p/q/v_world/w_world (13).
All link frames are URDF frames, not inertial/COM frames. No simulator body IDs.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import torch
from torch import nn
from reflex.amp.context import (
    _quat_mul as mul,
    _quat_rotate as rotate,
    _rot6d as amp_rot6,
)


def inverse(q):
    return torch.cat((q[..., :1], -q[..., 1:]), -1)


def apply(q, v):
    shape = v.shape
    return rotate(q.reshape(-1, 4), v.reshape(-1, 3)).reshape(shape)


def expmap(v):
    angle = v.norm(dim=-1, keepdim=True)
    return torch.cat(
        (torch.cos(angle / 2), 0.5 * torch.sinc(angle / (2 * math.pi)) * v), -1
    )


def normalize(q):
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def rot6(q):
    """Isaac observation convention: first two matrix columns, row-major flatten."""
    cols = amp_rot6(q.reshape(-1, 4)).reshape(*q.shape[:-1], 2, 3)
    return cols.transpose(-1, -2).reshape(*q.shape[:-1], 6)


def heading(q):
    w, x, y, z = q.unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    zero = torch.zeros_like(yaw)
    return torch.stack((torch.cos(yaw / 2), zero, zero, torch.sin(yaw / 2)), -1)


def canonical(s):
    h = inverse(heading(s[..., 3:7]))
    return torch.cat(
        (
            s[..., 2:3],
            rot6(mul(h, s[..., 3:7])),
            s[..., 7:36],
            apply(h, s[..., 36:39]),
            apply(h, s[..., 39:42]),
            s[..., 42:71] / 5,
            apply(h, s[..., 71:74] - s[..., :3]),
            rot6(mul(h, s[..., 74:78])),
            apply(h, s[..., 78:81] - s[..., 36:39]),
            apply(h, s[..., 81:84]),
        ),
        -1,
    )


def integrate(s, delta, dt, root_com=None, box_com=None):
    """82 residuals: dv,dw,djoint_vel,dp,dtheta,djoint_pos, box equivalents.

    Root/box residuals use the *current root heading* frame. Angular velocities
    in the state are world-frame; rotation increments multiply on the left.
    """
    h = heading(s[..., 3:7])
    dv, dw, djv, dp, dr, djp, bv, bw, bp, br = delta.split(
        (3, 3, 29, 3, 3, 29, 3, 3, 3, 3), -1
    )
    v = s[..., 36:39] + apply(h, dv * 0.3)
    w = s[..., 39:42] + apply(h, dw * 0.5)
    jv = s[..., 42:71] + djv * 2
    boxv = s[..., 78:81] + apply(h, bv * 0.3)
    boxw = s[..., 81:84] + apply(h, bw * 0.5)
    # Isaac root pose is a link pose, but root_lin_vel_w is COM velocity.
    # Convert velocity at COM to velocity at the link origin before integration.
    root_offset = (
        torch.zeros_like(v) if root_com is None else apply(s[..., 3:7], root_com)
    )
    box_offset = (
        torch.zeros_like(boxv) if box_com is None else apply(s[..., 74:78], box_com)
    )
    root_link_v = v - torch.cross(w, root_offset, dim=-1)
    box_link_v = boxv - torch.cross(boxw, box_offset, dim=-1)
    return torch.cat(
        (
            s[..., :3] + dt * root_link_v + apply(h, dp * 0.01),
            normalize(mul(expmap(dt * w + apply(h, dr * 0.02)), s[..., 3:7])),
            s[..., 7:36] + dt * jv + djp * 0.02,
            v,
            w,
            jv,
            s[..., 71:74] + dt * box_link_v + apply(h, bp * 0.01),
            normalize(mul(expmap(dt * boxw + apply(h, br * 0.02)), s[..., 74:78])),
            boxv,
            boxw,
        ),
        -1,
    )


def state_error(pred, true):
    """Sign-invariant quaternion error and nominal scales; 84 → 82 residuals."""

    def qerr(a, b):
        rel = mul(a, inverse(b))
        return 2 * rel[..., 1:] * torch.where(rel[..., :1] < 0, -1.0, 1.0)

    return torch.cat(
        (
            (pred[..., :3] - true[..., :3]) / 0.05,
            qerr(pred[..., 3:7], true[..., 3:7]) / 0.1,
            (pred[..., 7:36] - true[..., 7:36]) / 0.1,
            (pred[..., 36:39] - true[..., 36:39]) / 0.5,
            (pred[..., 39:42] - true[..., 39:42]),
            (pred[..., 42:71] - true[..., 42:71]) / 2,
            (pred[..., 71:74] - true[..., 71:74]) / 0.05,
            qerr(pred[..., 74:78], true[..., 74:78]) / 0.1,
            (pred[..., 78:81] - true[..., 78:81]) / 0.5,
            pred[..., 81:84] - true[..., 81:84],
        ),
        -1,
    )


class Kinematics(nn.Module):
    """URDF tree in simulator joint order. Fixed joints retained; meshes not needed."""

    def __init__(self, urdf, joint_names):
        super().__init__()
        tree = ET.parse(urdf).getroot()
        self.nodes = []
        children = {j.find("child").get("link") for j in tree.findall("joint")}
        roots = [
            l.get("name") for l in tree.findall("link") if l.get("name") not in children
        ]
        if roots != ["pelvis"]:
            raise ValueError(f"Expected pelvis URDF root, got {roots}")
        pending = list(tree.findall("joint"))
        known = {"pelvis"}
        used = set()
        while pending:
            ready = [j for j in pending if j.find("parent").get("link") in known]
            if not ready:
                raise ValueError("Disconnected or cyclic URDF")
            for j in ready:
                parent, child = (
                    j.find("parent").get("link"),
                    j.find("child").get("link"),
                )
                origin = j.find("origin")
                xyz = (
                    [float(v) for v in origin.get("xyz", "0 0 0").split()]
                    if origin is not None
                    else [0.0] * 3
                )
                rpy = (
                    [float(v) for v in origin.get("rpy", "0 0 0").split()]
                    if origin is not None
                    else [0.0] * 3
                )

                def axis_q(axis, angle):
                    return expmap(torch.tensor(axis) * angle)

                q = mul(
                    mul(
                        axis_q([0.0, 0.0, 1.0], rpy[2]), axis_q([0.0, 1.0, 0.0], rpy[1])
                    ),
                    axis_q([1.0, 0.0, 0.0], rpy[0]),
                )
                jid = -1
                axis = [0.0, 0.0, 0.0]
                if j.get("type") != "fixed":
                    if j.get("type") not in ("revolute", "continuous"):
                        raise ValueError(f"Unsupported joint type: {j.get('type')}")
                    jid = joint_names.index(j.get("name"))
                    used.add(jid)
                    axis = [float(v) for v in j.find("axis").get("xyz").split()]
                idx = len(self.nodes)
                self.register_buffer(f"xyz_{idx}", torch.tensor(xyz))
                self.register_buffer(f"quat_{idx}", q)
                self.register_buffer(f"axis_{idx}", torch.tensor(axis))
                self.nodes.append((parent, child, jid))
                known.add(child)
                pending.remove(j)
        if used != set(range(len(joint_names))):
            raise ValueError("URDF / simulator joint set differs")

    def forward(self, s):
        positions, quats = {"pelvis": s[..., :3]}, {"pelvis": s[..., 3:7]}
        for i, (parent, child, jid) in enumerate(self.nodes):
            xyz, q0, axis = (
                getattr(self, f"{key}_{i}") for key in ("xyz", "quat", "axis")
            )
            qp = quats[parent]
            positions[child] = positions[parent] + apply(
                qp, xyz.expand_as(positions[parent])
            )
            q = mul(qp, q0.expand_as(qp))
            if jid >= 0:
                q = mul(q, expmap(s[..., 7 + jid, None] * axis))
            quats[child] = q
        return positions, quats


def amp_features(s, fk):
    positions, _ = fk(s)
    hi = inverse(heading(s[..., 3:7]))
    keys = [
        apply(hi, positions[name] - s[..., :3])
        for name in (
            "left_wrist_yaw_link",
            "right_wrist_yaw_link",
            "left_ankle_roll_link",
            "right_ankle_roll_link",
        )
    ]
    # AMP uses concatenated columns, unlike Isaac's actor observation flatten.
    ori = amp_rot6(mul(hi, s[..., 3:7]).reshape(-1, 4)).reshape(*s.shape[:-1], 6)
    return torch.cat(
        (
            s[..., 7:36],
            s[..., 42:71],
            s[..., 2:3],
            ori,
            apply(hi, s[..., 36:39]),
            apply(hi, s[..., 39:42]),
            *keys,
        ),
        -1,
    )


class ActionMap(nn.Module):
    """Raw → pre-gimbal targets. The 4-substep gimbal/PD/PhysX is the WM plant.

    Only the launch-driven two-stage state machine is supported in this Oracle
    experiment. Perception-triggered latches deliberately fail preflight.
    """

    def __init__(
        self, arm_columns, ramp=8, delta_min=0.15, delta_gate=True, base_blend=True
    ):
        super().__init__()
        self.register_buffer("arms", torch.tensor(arm_columns, dtype=torch.long))
        self.ramp, self.delta_min = ramp, delta_min
        self.delta_gate, self.base_blend = delta_gate, base_blend

    def forward(self, raw, blend, launched, scale, offset, down, up, limits):
        b = (blend + (2 * launched.float()[..., None] - 1) / self.ramp).clamp(0, 1)
        target = raw * scale + offset
        arm = target[..., self.arms]
        delta = arm - offset[..., self.arms]
        if self.delta_gate:
            delta = delta * (self.delta_min + (1 - self.delta_min) * b)
        arm = (
            b * up + (1 - b) * down + delta
            if self.base_blend
            else b * arm + (1 - b) * down
        )
        # Out-of-place scatter preserves autograd and avoids modifying saved views.
        target = target.scatter(-1, self.arms.expand(*raw.shape[:-1], -1), arm)
        return torch.maximum(torch.minimum(target, limits[..., 1]), limits[..., 0]), b
