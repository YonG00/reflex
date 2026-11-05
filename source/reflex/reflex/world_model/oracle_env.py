"""Matched Oracle observations for both WM and PPO. Import only after AppLauncher."""

import torch
import os
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass
from reflex.envs.teacher_env_cfg import G1BoxCatchStepCatchEnvCfg
from .physics import inverse, mul, rot6, apply


def box_dimensions(env):
    """Read actual spawned geometry, never infer dimensions from randomized mass."""
    if not hasattr(env, "_wm_dimensions"):
        import omni.usd
        from pxr import Usd, UsdGeom

        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        )
        stage = omni.usd.get_context().get_stage()
        dims = [
            list(
                cache.ComputeUntransformedBound(stage.GetPrimAtPath(p))
                .ComputeAlignedRange()
                .GetSize()
            )
            for p in env.scene["balloon"].root_physx_view.prim_paths
        ]
        env._wm_dimensions = torch.tensor(dims, device=env.device, dtype=torch.float32)
        if not bool((env._wm_dimensions > 0).all()):
            raise ValueError("Cannot determine actual box geometry")
    return env._wm_dimensions


def box_mass(env):
    return (
        env.scene["balloon"]
        .root_physx_view.get_masses()
        .to(env.device)
        .reshape(env.num_envs, 1)
    )


def box_ori(env):
    return rot6(
        mul(
            inverse(env.scene["robot"].data.root_quat_w),
            env.scene["balloon"].data.root_quat_w,
        )
    )


def box_ang(env):
    return apply(
        inverse(env.scene["robot"].data.root_quat_w),
        env.scene["balloon"].data.root_ang_vel_w,
    )


@configclass
class OracleEnvCfg(G1BoxCatchStepCatchEnvCfg):
    def __post_init__(self):
        if (
            os.environ.get("BOX_DP_EST") != "0"
            or os.environ.get("BOX_V38_TS_FROM_PERC") != "0"
        ):
            raise ValueError(
                "Oracle config requires configs/world_model/oracle.env before imports"
            )
        super().__post_init__()
        self.observations.policy.enable_corruption = False
        self.observations.policy.blind_age = None
        self.observations.policy.box_trust = None
        self.observations.policy.box_ori = ObsTerm(func=box_ori)
        self.observations.policy.box_ang = ObsTerm(func=box_ang)
        self.observations.policy.box_dimensions = ObsTerm(func=box_dimensions)
        self.observations.policy.box_mass = ObsTerm(func=box_mass)


def register():
    import gymnasium as gym
    from reflex.envs.register import RSL_RL_CFG

    name = "Reflex-Teacher-Oracle-G1-v0"
    if name not in gym.registry:
        gym.register(
            name,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": OracleEnvCfg,
                "rsl_rl_cfg_entry_point": RSL_RL_CFG,
            },
        )
