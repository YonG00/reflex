"""Explicit, one-way checkpoint changes for soft termination and Critic inputs."""

import copy
from dataclasses import asdict

import torch


def migrate(state, trainer, analytic=False, critic_box=False):
    from .trainer import TrainConfig

    if state.get("version") != 3:
        raise ValueError("V29 migration requires a v3 checkpoint")
    result = dict(
        state
    )  # Replay/WM/Actor tensors stay immutable; do not double GPU Replay.
    cfg = asdict(TrainConfig(**state["config"]))
    learning = copy.deepcopy(state["learning_contract"])
    record = {}
    if analytic:
        if (
            not cfg["learned_termination"]
            or cfg["analytic_soft_termination"]
            or not trainer.cfg.analytic_soft_termination
        ):
            raise ValueError(
                "Only explicit learned -> analytic soft termination migration is supported"
            )
        if cfg["hard_imagination_boundaries"] or not cfg["pure_soft_survival"]:
            raise ValueError(
                "Migrate source to pure soft survival first; no combined boundary/reward changes"
            )
        cfg.update(
            learned_termination=False,
            joint_predicted_termination=False,
            analytic_soft_termination=True,
        )
        if not state.get("termination") or state.get("termination_opt") is None:
            raise ValueError("Source checkpoint lacks learned termination/optimizer")
        result.update(termination=None, termination_opt=None, termination_updates=0)
        for key in (
            "termination_training",
            "state_loss_reduction",
            "term_loss_weight",
            "variable_model_windows",
        ):
            learning.pop(key, None)
        learning.update(termination_geometry=None, termination_feature_dim=None)
        for key in ("analytic_soft_termination", "soft_survival"):
            learning[key] = copy.deepcopy(trainer.learning_contract[key])
        record["termination"] = (
            "learned head and head optimizer discarded; WM/Actor/Critic optimizers preserved unless Critic expanded"
        )
    if critic_box:
        if (
            not cfg["privileged_critic"]
            or cfg["critic_include_all_actor"]
            or not trainer.cfg.critic_include_all_actor
        ):
            raise ValueError(
                "Only explicit privileged292 -> privileged305 migration is supported"
            )
        if (
            learning.get("critic_layout") != "reflex_privileged292_v1"
            or learning.get("critic_input_dim") != 292
        ):
            raise ValueError(
                "Unrecognized source Critic layout; refuse column migration"
            )
        weight = state["critic"]["0.weight"]
        if weight.shape[1] != 292:
            raise ValueError("Expected original 292-column Critic input")
        result["critic"] = dict(state["critic"])
        result["critic"]["0.weight"] = torch.cat(
            (weight, weight.new_zeros(weight.shape[0], 13)), -1
        )
        result["critic_norm"] = {
            "mean": torch.cat((state["critic_norm"]["mean"], weight.new_zeros(13))),
            "std": torch.cat((state["critic_norm"]["std"], weight.new_ones(13))),
        }
        result["critic_opt"] = copy.deepcopy(trainer.critic_opt.state_dict())
        if result["critic_opt"]["state"]:
            raise ValueError("Critic expansion requires a fresh destination optimizer")
        cfg["critic_include_all_actor"] = True
        learning["critic_input_dim"] = 305
        learning["critic_layout"] = "reflex_privileged305_all_actor_v1"
        record["critic"] = (
            "append 13 zero-weight columns, mean0/std1; initial value unchanged; Critic Adam reset explicitly"
        )
    if not record:
        raise ValueError("No explicit V29 migration selected")
    result.update(config=cfg, learning_contract=learning)
    result["initialization"] = copy.deepcopy(state.get("initialization", {}))
    result["initialization"]["v29_migration"] = record
    return result
