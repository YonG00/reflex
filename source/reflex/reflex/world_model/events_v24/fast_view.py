"""Minimal native-event-only facade; no reward/reference/FK construction."""

import copy

from ..task import HISTORY_FIELDS, NS, Scene
from .writes import PlannedAsset


def build(scheduler, b, physical, poses, values, episode, common, plan):
    del poses  # Supported native events use root/joint data, never body FK.
    import torch

    physical = torch.where(plan.mask.clone(), plan.targets, physical)
    origins = b["event_origins"]
    robot = NS(
        root_pos_w=physical[:, :3] + origins,
        root_quat_w=physical[:, 3:7].clone(),
        root_lin_vel_w=physical[:, 36:39].clone(),
        root_ang_vel_w=physical[:, 39:42].clone(),
        joint_pos=physical[:, 7:36].clone(),
        joint_vel=physical[:, 42:71].clone(),
    )
    box = NS(
        root_pos_w=physical[:, 71:74] + origins,
        root_quat_w=physical[:, 74:78].clone(),
        root_lin_vel_w=physical[:, 78:81].clone(),
        root_ang_vel_w=physical[:, 81:84].clone(),
    )
    scene = Scene(
        robot=PlannedAsset(
            robot,
            scheduler.task.body_names,
            scheduler.task.joint_names,
            plan,
            origins,
            True,
        ),
        balloon=PlannedAsset(box, ["box"], [], plan, origins, False),
    )
    scene.env_origins, scene.sensors = origins, {}
    view = NS(
        scene=scene,
        num_envs=len(physical),
        device=physical.device,
        step_dt=scheduler.task.dt,
        physics_dt=scheduler.task.dt / 4,
        common_step_counter=common,
        episode_length_buf=episode,
        max_episode_length=scheduler.task.max_steps,
        _wm_ephemeral_view=True,
    )
    for i, name in enumerate(HISTORY_FIELDS):
        setattr(view, name, b["task_history"][:, i].round().long())
    for name, value in scheduler.static.items():
        setattr(view, name, copy.deepcopy(value))
    for name, value in values.items():
        setattr(view, name, value.clone())
    view._box_half_extents = b["dimensions"] / 2
    return view
