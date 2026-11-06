"""Explicit standing-priority reward ablation; real and imagined use one formula."""

import copy
import torch

PROFILE = "standing_h16_v1"
TRACKING = (
    "motion_global_anchor_pos",
    "motion_global_anchor_ori",
    "motion_body_pos",
    "motion_body_ori",
    "motion_body_lin_vel",
    "motion_body_ang_vel",
)
WEIGHTS = {
    **{n: (0.125 if "anchor" in n else 0.25) for n in TRACKING},
    "fall_terminated": -150.0,
    "soft_fall_shaping": -4.0,
    "torso_uprightness_penalty": 2.0,
}


def standing_balance(env, command_name="motion", height_std=0.12, upright_std=0.2):
    robot = env.scene["robot"]
    index = robot.body_names.index("torso_link")
    q = robot.data.body_quat_w[:, index]
    up_z = 1 - 2 * (q[:, 1].square() + q[:, 2].square())
    command = env.command_manager.get_term(command_name)
    dz = command.robot_anchor_pos_w[:, 2] - command.anchor_pos_w[:, 2]
    return torch.exp(-(dz / height_std).square() - ((1 - up_z) / upright_std).square())


def apply_profile(cfg, profile):
    if profile == "original":
        return
    if profile != PROFILE:
        raise ValueError("Unknown reward profile")
    from isaaclab.managers import RewardTermCfg

    for name, weight in WEIGHTS.items():
        term = getattr(cfg.rewards, name)
        term.weight = weight
    cfg.rewards.standing_balance = RewardTermCfg(
        func=standing_balance,
        weight=5.0,
        params={"command_name": "motion", "height_std": 0.12, "upright_std": 0.2},
    )


def validate_migration(state, task, reward_names, config):
    if not config.deterministic_event_apply or not config.pure_soft_survival:
        raise ValueError(
            "Standing reward migration requires deterministic physical WM and pure soft survival"
        )
    if (
        state["config"].get("reward_profile", "original") != "original"
        or config.reward_profile != PROFILE
        or state["config"]["horizon"] != 32
        or config.horizon != 16
    ):
        raise ValueError(
            "Only explicit original H32 to standing H16 migration is supported"
        )
    if reward_names != state["reward_names"] + ["standing_balance"]:
        raise ValueError("Unexpected reward layout change")
    old, new = copy.deepcopy(state["task_spec"]), copy.deepcopy(task.specification())
    old_rewards, new_rewards = old.pop("rewards"), new.pop("rewards")
    if old != new:
        raise ValueError(
            "Standing profile cannot change physical task/termination contract"
        )
    expected = copy.deepcopy(old_rewards)
    for row in expected:
        if row["name"] in WEIGHTS:
            row["weight"] = WEIGHTS[row["name"]]
            row["kind"] = "original_torch_formula"
    if (
        expected != new_rewards[:-1]
        or new_rewards[-1]["name"] != "standing_balance"
        or new_rewards[-1]["weight"] != 5.0
        or new_rewards[-1]["function"]
        != "reflex.world_model.standing_profile.standing_balance"
        or new_rewards[-1]["params"]
        != {"command_name": "motion", "height_std": 0.12, "upright_std": 0.2}
    ):
        raise ValueError("Unapproved standing reward modification")
    return {
        "profile": PROFILE,
        "horizon_from": 32,
        "horizon_to": 16,
        "standing_weight": 5.0,
        "terminal_cost_from": -1.0,
        "terminal_cost_to": -3.0,
        "tracking_scale": 0.25,
        "optimizers_preserved": True,
        "replay_rebuilt": True,
    }
