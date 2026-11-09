"""Extra ground-truth box observations for the Stage 1 PPO ablation.

All rotations use the robot root frame. Velocities are rotated world velocities,
without subtracting robot velocity. Sizes are full box-local edge lengths (m).
"""
from __future__ import annotations

import torch
from isaaclab.utils.math import matrix_from_quat, quat_apply_inverse, quat_conjugate, quat_mul


def box_orientation_6d(env) -> torch.Tensor:
    """First two columns of R_root^T R_box, flattened row by row (N, 6)."""
    robot = env.scene["robot"]
    box = env.scene["balloon"]
    relative = quat_mul(quat_conjugate(robot.data.root_quat_w), box.data.root_quat_w)
    return matrix_from_quat(relative)[..., :2].reshape(env.num_envs, 6)


def box_angular_velocity_b(env) -> torch.Tensor:
    """World angular velocity expressed in the robot root frame (rad/s)."""
    box = env.scene["balloon"]
    velocity = torch.nan_to_num(box.data.root_ang_vel_w, nan=0.0, posinf=10.0, neginf=-10.0)
    return quat_apply_inverse(env.scene["robot"].data.root_quat_w, velocity)


def box_dimensions(env) -> torch.Tensor:
    """Full x/y/z edge lengths from the existing per-environment model cache."""
    if hasattr(env, "_box_half_extents"):
        return 2.0 * env._box_half_extents
    # ObservationManager probes shapes before the first reset.
    # Read geometry without touching throw buffers or using randomised mass.
    if not hasattr(env, "_box_info_dimensions"):
        from reflex.envs.throw import BOX_MODELS, infer_models_from_geometry
        box = env.scene["balloon"]
        _, half_extents, _ = infer_models_from_geometry(box, BOX_MODELS, box.device)
        env._box_info_dimensions = 2.0 * half_extents
    return env._box_info_dimensions


def box_mass(env) -> torch.Tensor:
    """Current PhysX mass (kg), including startup and per-reset mass DR."""
    box = env.scene["balloon"]
    # RigidObject has one body. Read live mass so curriculum updates are visible.
    return box.root_physx_view.get_masses().reshape(env.num_envs, 1).to(
        device=box.data.root_pos_w.device, dtype=box.data.root_pos_w.dtype
    )
