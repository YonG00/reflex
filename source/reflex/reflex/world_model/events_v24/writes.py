"""Record issued native writes for learned or deterministic event application."""

import torch

from ..physics import apply, heading, inverse, state_error
from ..task import Asset


class WritePlan:
    condition_dim = 84 + 82

    def __init__(self, physical):
        self.targets = torch.zeros_like(physical)
        self.mask = torch.zeros_like(physical, dtype=torch.bool)

    def record(self, before, requested, env_ids, slots, intervention=None):
        ids = (
            torch.arange(len(self.targets), device=self.targets.device)
            if env_ids is None
            else env_ids
        )
        if isinstance(ids, slice):
            ids = torch.arange(len(self.targets), device=self.targets.device)[ids]
        slots = torch.as_tensor(slots, device=self.targets.device)
        # A native write overwrites every requested component even when its
        # numeric value equals the previous value. This also preserves the
        # correct derivative through constant/predicted write targets.
        rr, cc = ids[:, None], slots[None, :]
        active = (
            torch.ones_like(requested, dtype=torch.bool)
            if intervention is None
            else intervention.expand_as(requested)
        )
        self.targets[rr, cc] = torch.where(active, requested, self.targets[rr, cc])
        self.mask[rr, cc] |= active

    def condition(self, physical):
        return event_condition(physical, self.targets, self.mask)

    def select(self, ids):
        result = WritePlan(self.targets[ids])
        result.targets = self.targets[ids].clone()
        result.mask = self.mask[ids].clone()
        return result

    def merge(self, ids, other):
        self.targets = self.targets.index_copy(0, ids, other.targets)
        self.mask = self.mask.index_copy(0, ids, other.mask)


class PlannedAsset(Asset):
    def __init__(self, data, body_names, joint_names, plan, origins, robot):
        super().__init__(data, body_names, joint_names)
        self.plan, self.origins, self.robot = plan, origins, robot
        self._refresh()

    def _refresh(self):
        d = self.data
        d.root_vel_w = torch.cat((d.root_lin_vel_w, d.root_ang_vel_w), -1)
        d.root_state_w = torch.cat((d.root_pos_w, d.root_quat_w, d.root_vel_w), -1)

    def _ids(self, env_ids):
        return (
            torch.arange(len(self.origins), device=self.device)
            if env_ids is None
            else env_ids
        )

    def write_root_pose_to_sim(self, pose, env_ids=None):
        ids = self._ids(env_ids)
        d = self.data
        before = torch.cat(
            (d.root_pos_w[ids] - self.origins[ids], d.root_quat_w[ids]), -1
        )
        local = torch.cat((pose[:, :3] - self.origins[ids], pose[:, 3:]), -1)
        slots = list(range(7)) if self.robot else list(range(71, 78))
        self.plan.record(
            before, local, ids, slots, getattr(self, "_reflex_event_write_mask", None)
        )
        d.root_pos_w = d.root_pos_w.index_copy(0, ids, pose[:, :3])
        d.root_quat_w = d.root_quat_w.index_copy(0, ids, pose[:, 3:])
        self._refresh()

    def write_root_velocity_to_sim(self, vel, env_ids=None):
        ids = self._ids(env_ids)
        d = self.data
        self.plan.record(
            d.root_vel_w[ids],
            vel,
            ids,
            list(range(36, 42)) if self.robot else list(range(78, 84)),
            getattr(self, "_reflex_event_write_mask", None),
        )
        d.root_lin_vel_w = d.root_lin_vel_w.index_copy(0, ids, vel[:, :3])
        d.root_ang_vel_w = d.root_ang_vel_w.index_copy(0, ids, vel[:, 3:])
        self._refresh()

    def write_joint_state_to_sim(
        self, positions, velocities, joint_ids=None, env_ids=None
    ):
        ids = self._ids(env_ids)
        joints = (
            torch.arange(29, device=self.device) if joint_ids is None else joint_ids
        )
        if isinstance(joints, slice):
            joints = torch.arange(29, device=self.device)[joints]
        d = self.data
        self.plan.record(
            d.joint_pos[ids[:, None], joints[None, :]],
            positions,
            ids,
            joints + 7,
            getattr(self, "_reflex_event_write_mask", None),
        )
        self.plan.record(
            d.joint_vel[ids[:, None], joints[None, :]],
            velocities,
            ids,
            joints + 42,
            getattr(self, "_reflex_event_write_mask", None),
        )
        q, v = d.joint_pos.clone(), d.joint_vel.clone()
        q[ids[:, None], joints[None, :]] = positions
        v[ids[:, None], joints[None, :]] = velocities
        d.joint_pos, d.joint_vel = q, v


def record_native(
    plan, physical, origins, robot, method, args, kwargs, intervention=None
):
    """Capture issued control parameters BEFORE the native API applies them."""
    if method == "write_joint_state_to_sim":
        ids = kwargs.get("env_ids", args[3] if len(args) > 3 else None)
        joints = kwargs.get("joint_ids", args[2] if len(args) > 2 else None)
        ids = (
            torch.arange(len(physical), device=physical.device) if ids is None else ids
        )
        joints = torch.arange(29, device=physical.device) if joints is None else joints
        if isinstance(joints, slice):
            joints = torch.arange(29, device=physical.device)[joints]
        plan.record(
            physical[ids[:, None], (joints + 7)[None, :]],
            args[0],
            ids,
            joints + 7,
            intervention,
        )
        plan.record(
            physical[ids[:, None], (joints + 42)[None, :]],
            args[1],
            ids,
            joints + 42,
            intervention,
        )
        return
    ids = kwargs.get("env_ids", args[1] if len(args) > 1 else None)
    ids = torch.arange(len(physical), device=physical.device) if ids is None else ids
    if isinstance(ids, slice):
        ids = torch.arange(len(physical), device=physical.device)[ids]
    if method == "write_root_pose_to_sim":
        slots = list(range(7)) if robot else list(range(71, 78))
        requested = torch.cat((args[0][:, :3] - origins[ids], args[0][:, 3:]), -1)
    elif method == "write_root_velocity_to_sim":
        slots = list(range(36, 42)) if robot else list(range(78, 84))
        requested = args[0]
    else:
        raise ValueError(f"Unverified V24 native writer: {method}")
    plan.record(physical[ids][:, slots], requested, ids, slots, intervention)


def event_condition(physical, targets, mask):
    desired = torch.where(mask, targets, physical)
    scales = physical.new_tensor(
        [0.05] * 3
        + [0.1] * 3
        + [0.1] * 29
        + [0.5] * 3
        + [1.0] * 3
        + [2.0] * 29
        + [0.05] * 3
        + [0.1] * 3
        + [0.5] * 3
        + [1.0] * 3
    )
    delta = state_error(desired, physical) * scales
    pieces = delta.split((3, 3, 29, 3, 3, 29, 3, 3, 3, 3), -1)
    frame = inverse(heading(physical[..., 3:7]))
    delta = torch.cat(
        [part if i in (2, 5) else apply(frame, part) for i, part in enumerate(pieces)],
        -1,
    )
    return torch.cat((mask.float(), delta), -1)
