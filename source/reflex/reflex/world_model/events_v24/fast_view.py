"""Minimal native-event-only facade; no reward/reference/FK construction."""

import copy

from ..task import HISTORY_FIELDS, NS, Scene
from .writes import PlannedAsset


def build(scheduler, b, physical, poses, values, episode, common, plan):
    del poses  # Supported native events use root/joint data, never body FK.
    import torch

    origins = b["event_origins"]
    preparer = getattr(scheduler, "tensor_preparer", None)
    if preparer is None:
        physical = torch.where(plan.mask.clone(), plan.targets, physical)
        r = (
            physical[:, :3] + origins,
            physical[:, 3:7].clone(),
            physical[:, 36:39].clone(),
            physical[:, 39:42].clone(),
            physical[:, 7:36].clone(),
            physical[:, 42:71].clone(),
        )
        x = (
            physical[:, 71:74] + origins,
            physical[:, 74:78].clone(),
            physical[:, 78:81].clone(),
            physical[:, 81:84].clone(),
        )
        counters = tuple(
            b["task_history"][:, i].round().long() for i in range(len(HISTORY_FIELDS))
        )
        copied_values = {name: value.clone() for name, value in values.items()}
    else:
        r, x, counters, copied_values = preparer(
            physical, plan.targets, plan.mask.clone(), origins, b["task_history"], values
        )
    robot = NS(
        **dict(
            zip(
                (
                    "root_pos_w",
                    "root_quat_w",
                    "root_lin_vel_w",
                    "root_ang_vel_w",
                    "joint_pos",
                    "joint_vel",
                ),
                r,
            )
        )
    )
    box = NS(
        **dict(
            zip(("root_pos_w", "root_quat_w", "root_lin_vel_w", "root_ang_vel_w"), x)
        )
    )
    asset_factory = getattr(scheduler, "event_asset_factory", PlannedAsset)
    scene = Scene(
        robot=asset_factory(
            robot,
            scheduler.task.body_names,
            scheduler.task.joint_names,
            plan,
            origins,
            True,
        ),
        balloon=asset_factory(box, ["box"], [], plan, origins, False),
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
        setattr(view, name, counters[i])
    for name, value in scheduler.static.items():
        setattr(view, name, copy.deepcopy(value))
    for name, value in copied_values.items():
        setattr(view, name, value)
    view._box_half_extents = b["dimensions"] / 2
    return view
