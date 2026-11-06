"""Collect → fit joint WM / real AMP → imagine → pathwise actor / λ critic.

No PPO loss or likelihood ratios. All model/discriminator weights are frozen
during actor backprop; their derivatives with respect to inputs remain enabled.
"""

from __future__ import annotations

import random
from contextlib import contextmanager
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from .models import (
    ConditionedWorldModel,
    DeterministicEventWorldModel,
    MacroWorldModel,
    Standardizer,
    StateWorldModel,
    TerminationHead,
    WorldModel,
    mlp,
)
from .physics import amp_features, state_error
from .replay import Replay
from .timing import timestamp


@dataclass
class TrainConfig:
    monitor_contacts: bool = False
    learn_external_events: bool = False
    condition_external_events: bool = False
    deterministic_event_apply: bool = False
    minimal_event_view: bool = False
    event_supervision_protocol: str = "recorded_commands"
    privileged_critic: bool = False
    actor_zero_output: bool = True
    learned_termination: bool = False
    joint_predicted_termination: bool = False
    variable_model_windows: bool = False
    state_loss_reduction: str = "mean"
    term_loss_weight: float = 1.0
    termination_lr: float = 2e-4
    diagnostic_every: int = 0
    diagnostic_batch: int = 16
    compile_dynamics: bool = False
    compile_fk: bool = False
    compile_step: bool = False
    compile_backend: str = "inductor"
    tf32: bool = False
    schedule: str = "guarded"
    initial_collect_steps: int = 40
    updates_per_env_step: float = 0.4
    actor_batch: int = 0
    actor_microbatch: int = 0
    actor_beta2: float = 0.999
    actor_grad_clip_value: float = 0.0
    actor_horizon_scale: bool = False
    critic_epochs: int = 1
    critic_minibatches: int = 1
    critic_hidden_dims: list[int] | None = None
    validation_every: int = 1
    lr_total_steps: int = 320000
    lr_final_ratio: float = 0.03
    lr_decay_steps: int = 250000
    capacity_steps: int = 1024
    collect_steps: int = 24
    model_steps: int = 64
    model_window: int = 8
    batch: int = 256
    horizon: int = 32
    actor_steps: int = 4
    warmup_iterations: int = 24
    gamma: float = 0.99
    lam: float = 0.95
    lr: float = 3e-4
    model_lr: float | None = None
    model_grad_clip: float = 10.0
    exploration: float = 0.15
    hidden: int = 256
    wm_kind: str = "state"
    wm_hidden_dims: list[int] | None = None
    gradient_audit_every: int = 20
    normalizer_samples: int = 2048
    actor_hidden_dims: list[int] | None = None
    stage_balanced: bool = False
    # Normalized physical error / normalized per-term reward error. Provisional
    # acceptance limits, not literature-backed guarantees of safe imagination.
    max_state_rmse: float = 1.0
    max_reward_rmse: float = 1.0


@contextmanager
def freeze(*modules):
    params = [
        (p, p.requires_grad) for m in modules if m is not None for p in m.parameters()
    ]
    try:
        for p, _ in params:
            p.requires_grad_(False)
        yield
    finally:
        for p, old in params:
            p.requires_grad_(old)


def lambda_returns(rewards, done, values, gamma, lam):
    """Time limits are not terminal: only physical/MDP termination cuts bootstrap.

    Same λ-return equation as SuperMimic-Reproduced imagination/objective.py,
    MIT, © 2026 Yuxi Liu; attribution/license under docs/world_model/.
    """
    result = []
    ret = values[-1]
    for t in reversed(range(len(rewards))):
        ret = rewards[t] + gamma * (1 - done[t]) * (
            (1 - lam) * values[t + 1] + lam * ret
        )
        result.append(ret)
    return torch.stack(result[::-1])


class Trainer:
    def __init__(
        self,
        bridge,
        action_map,
        prior,
        context_dim,
        memory_dim,
        reward_names,
        contract,
        device="cpu",
        cfg=None,
        task=None,
        event_scheduler=None,
    ):
        self.cfg, self.device = cfg or TrainConfig(), device
        c = self.cfg
        self.bridge, self.action_map, self.prior = (
            bridge.to(device),
            action_map.to(device),
            prior,
        )
        self.reward_names, self.contract = list(reward_names), contract
        if c.tf32:
            torch.set_float32_matmul_precision("high")
        self.task = task.to(device) if task is not None else None
        self.event_scheduler = event_scheduler
        if c.minimal_event_view:
            if not c.condition_external_events or event_scheduler is None:
                raise ValueError(
                    "Minimal native view requires conditioned event scheduler"
                )
            import types

            from .events_v24.fast_view import build

            event_scheduler._view = types.MethodType(build, event_scheduler)
        if (
            c.condition_external_events
            and c.event_supervision_protocol != "recorded_commands"
        ):
            raise ValueError(
                "V24 H8 supervision requires recorded event commands; free rollouts are evaluation/Actor only"
            )
        if c.deterministic_event_apply and not c.condition_external_events:
            raise ValueError("Deterministic events require conditioned events")
        if c.condition_external_events and event_scheduler is None:
            raise ValueError("V24 needs a verified native event scheduler")
        if c.wm_kind not in ("state", "multitask"):
            raise ValueError("Unknown WM kind")
        self.model = (
            (
                DeterministicEventWorldModel
                if c.deterministic_event_apply
                else ConditionedWorldModel
                if c.condition_external_events
                else MacroWorldModel
                if c.learn_external_events
                else StateWorldModel
            )(
                context_dim,
                hidden_layers=c.wm_hidden_dims or [2048] * 3,
                com_indices=contract.get("com_indices"),
            )
            if c.wm_kind == "state"
            else WorldModel(
                context_dim,
                memory_dim,
                len(reward_names),
                hidden=c.hidden,
                com_indices=contract.get("com_indices"),
            )
        ).to(device)
        self.obs_norm = Standardizer(179).to(device)
        self.actor = mlp(179, 29, c.hidden, layers=c.actor_hidden_dims).to(device)
        if c.actor_zero_output:
            nn.init.zeros_(self.actor[-1].weight)
            nn.init.zeros_(self.actor[-1].bias)
        self.privileged_bridge = None
        if c.privileged_critic:
            from .observation import PrivilegedObservation

            self.privileged_bridge = PrivilegedObservation(
                self.bridge, contract["tracked_names"]
            ).to(device)
        self.critic_dim = 292 if c.privileged_critic else 179
        self.critic_norm = (
            Standardizer(292).to(device) if c.privileged_critic else self.obs_norm
        )
        self.critic = mlp(self.critic_dim, 1, c.hidden, layers=c.critic_hidden_dims).to(
            device
        )
        if c.learned_termination and (self.task is None or not self.model.state_only):
            raise ValueError(
                "Learned termination requires pure dynamics and an analytic task"
            )
        self.termination = (
            TerminationHead(self.task.termination_feature_dim).to(device)
            if c.learned_termination
            else None
        )
        self.termination_opt = (
            torch.optim.Adam(self.termination.parameters(), lr=c.termination_lr)
            if self.termination is not None
            else None
        )
        if c.joint_predicted_termination and self.termination is None:
            raise ValueError("Joint predicted termination requires the learned head")
        if c.state_loss_reduction not in ("sum", "mean"):
            raise ValueError("state_loss_reduction must be sum or mean")
        self._termination_probe_due = c.joint_predicted_termination
        self.last_diagnostic_seconds = 0.0
        self.contact_filter_schema = None
        self.contact_filter_migration = None
        self.profile_next_actor = False
        self.profile_dir = None
        if c.learn_external_events and (
            not self.model.state_only
            or c.compile_step
            or not c.joint_predicted_termination
        ):
            raise ValueError(
                "Macro experiment requires state WM with validated Dynamics/FK compile only"
            )
        dynamics_call = (
            self.model.predict_physical
            if c.condition_external_events
            else self.model.predict_all
            if c.learn_external_events
            else self.model
        )
        self._dynamics = (
            torch.compile(dynamics_call, fullgraph=True, backend=c.compile_backend)
            if c.compile_dynamics
            else dynamics_call
        )
        self._post = (
            torch.compile(
                self.model.predict_post, fullgraph=True, backend=c.compile_backend
            )
            if c.condition_external_events and c.compile_dynamics
            else self.model.predict_post
            if c.condition_external_events
            else None
        )
        self._fk = (
            torch.compile(self.bridge.fk, fullgraph=True, backend=c.compile_backend)
            if c.compile_fk
            else self.bridge.fk
        )
        self._transition_cache_ready = False
        self._task_step = (
            torch.compile(
                self.task.evaluate_step, fullgraph=True, backend=c.compile_backend
            )
            if c.compile_step
            else (self.task.evaluate_step if self.task is not None else None)
        )
        self._termination_fn = (
            torch.compile(self.termination, fullgraph=True, backend=c.compile_backend)
            if c.compile_step and self.termination is not None
            else self.termination
        )
        # Keep FK and reward graphs separate: Torch 2.7's fused backward
        # codegen expands their shared quaternion expressions excessively.
        self._transition = self.transition if c.compile_step else None
        self._observations = (
            torch.compile(
                self.imagined_observations, fullgraph=True, backend=c.compile_backend
            )
            if c.compile_step
            else self.imagined_observations
        )
        self.model_opt = torch.optim.Adam(
            self.model.parameters(), lr=c.model_lr or c.lr
        )
        self.actor_opt = torch.optim.Adam(
            self.actor.parameters(), lr=c.lr, betas=(0.9, c.actor_beta2)
        )
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=c.lr)
        self.replay = Replay(
            c.capacity_steps,
            device,
            global_fields=("event_pool",) if c.condition_external_events else (),
        )
        self.iteration, self.real_transitions = 0, 0
        self.probe_transitions = 0
        self.actor_updates = 0
        self.model_updates = 0
        self.termination_updates = 0
        self.replay_seconds = 0.0
        self.normalized = False
        self.initialization = {"kind": "scratch"}
        from .experience import ExperienceMetrics

        self.experience = ExperienceMetrics(monitor_contacts=self.cfg.monitor_contacts)

    def warm_start_macro(self, path):
        if not self.cfg.learn_external_events:
            raise ValueError("Macro warm start requires learn_external_events")
        state = torch.load(path, map_location=self.device, weights_only=False)
        if (
            state["contract"] != self.contract
            or state["reward_names"] != self.reward_names
        ):
            raise ValueError("Macro warm start physical/reward contract mismatch")
        if (
            state["config"].get("condition_external_events", False)
            and not self.cfg.deterministic_event_apply
        ) or (
            state["config"].get("learn_external_events", False)
            and not self.cfg.condition_external_events
        ):
            raise ValueError("Use full resume for an existing macro checkpoint")
        if state.get("task_spec") != self.task.specification():
            raise ValueError("Macro warm start task mismatch")
        for name in (
            "actor",
            "critic",
            "obs_norm",
            "critic_norm",
            "bridge",
            "action_map",
            "actor_opt",
            "critic_opt",
        ):
            getattr(self, name).load_state_dict(state[name])
        self.model.warm_start_physical(state["model"])
        self.termination.load_state_dict(state["termination"])
        self.termination_opt.load_state_dict(state["termination_opt"])
        self.task.load_state_dict(state["task_state"])
        self.prior.load_amp_state_dict(state["amp"])
        self.normalized = state["normalized"]
        self.initialization = {
            "kind": "macro_environment_warm_start",
            "source": str(path),
            "source_actor_updates": state["actor_updates"],
            "wm_optimizer_reset": True,
            "replay_reset": True,
            "physical_wm_weights_preserved": True,
        }

    def warm_start_actor(self, path):
        """Engineering bootstrap only: preserve first166 channels, zero new cols.

        The dropped blind/trust channels are fixed at zero (ideal measurements).
        Fresh optimizers, WM, replay and AMP remain explicitly fresh.
        """
        import hashlib
        from pathlib import Path

        saved = torch.load(path, map_location=self.device, weights_only=False)[
            "model_state_dict"
        ]
        active = [
            k
            for k, v in saved.items()
            if "normalizer" in k
            and v.numel() > 0
            and not k.endswith("num_batches_tracked")
        ]
        if active:
            raise ValueError(
                "Warm-start PPO must use identity observation normalization"
            )
        weights = {
            k.removeprefix("actor."): v.clone()
            for k, v in saved.items()
            if k.startswith("actor.")
        }
        if weights["0.weight"].shape[1] not in (167, 170):
            raise ValueError("Expected legacy 167/170-D Teacher policy")
        first = weights["0.weight"]
        weights["0.weight"] = torch.zeros(
            (first.shape[0], 179), device=self.device, dtype=first.dtype
        )
        weights["0.weight"][:, :166] = first[:, :166]
        self.actor.load_state_dict(weights, strict=True)
        self.initialization = {
            "kind": "ppo_actor_warm_start",
            "path": str(path),
            "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "source_obs_dim": first.shape[1],
            "dropped_reliability_channels": 0.0,
            "optimizer_replay_amp_restored": False,
        }

    def observation(self, b, s=None, previous=None, phase=None, fk_result=None):
        return self.bridge(
            b["state"] if s is None else s,
            b["previous_action"] if previous is None else previous,
            b["phase"] if phase is None else phase,
            b["yaw"],
            b["launched"],
            b["dimensions"],
            b["mass"],
            b.get("default_q"),
            b.get("default_dq"),
            fk_result=fk_result,
        )

    def critic_observation(
        self, b, s=None, previous=None, phase=None, fk_result=None, actor_result=None
    ):
        if self.privileged_bridge is None:
            return self.obs_norm(
                self.observation(b, s, previous, phase, fk_result)
                if actor_result is None
                else actor_result
            )
        value = self.privileged_bridge(
            b["state"] if s is None else s,
            b["previous_action"] if previous is None else previous,
            b["phase"] if phase is None else phase,
            b["yaw"],
            b["launched"],
            b["dimensions"],
            b["mass"],
            b.get("default_q"),
            b.get("default_dq"),
            fk_result=fk_result,
            actor_result=actor_result,
        )
        return self.critic_norm(value)

    def act(self, b, explore=False):
        raw = self.actor(self.obs_norm(self.observation(b)))
        if explore:
            raw = raw + self.cfg.exploration * torch.randn_like(raw)
        return raw

    def targets(self, b, raw, blend=None):
        return self.action_map(
            raw,
            b["blend"] if blend is None else blend,
            b["launched"],
            b["scale"],
            b["offset"],
            b["down"],
            b["up"],
            b["limits"],
        )

    def collect(self, adapter):
        rewards = []
        with torch.no_grad():
            for _ in range(self.cfg.collect_steps):
                before = adapter.snapshot()
                raw = self.act(before, explore=True)
                expected, blend = self.targets(before, raw)
                tr = adapter.step(raw, expected, blend)
                # Fail closed: infinities in hard-limit metadata are legal,
                # nonfinite observations/targets/labels are not.
                fields = (
                    "state",
                    "next_state",
                    "context",
                    "memory",
                    "next_memory",
                    "reward_terms",
                    "target",
                )
                finite = torch.stack(
                    [torch.isfinite(tr[k]).flatten(1).all(-1) for k in fields]
                ).all(0)
                tr["valid"] &= finite
                self.replay.add(tr)
                self.experience.update(tr)
                ok = tr["valid"] & ~tr["terminated"] & ~tr["timeout"]
                if bool(ok.any()):
                    self.prior.add_real(torch.stack((tr["amp"], tr["next_amp"]), 1)[ok])
                self.real_transitions += len(raw)
                rewards.append(tr["reward_terms"].sum(-1).mean())
        return {
            "real/task_reward": float(torch.stack(rewards).mean()),
            **self.experience.metrics(),
        }

    def sample(self, horizon=1, starts=False, validation=False, n=None):
        start = timestamp(self.device)
        try:
            if starts:
                return self.replay.sample_starts(n or self.cfg.batch, validation)
            if self.cfg.variable_model_windows and not validation:
                return self.replay.sample_variable_windows(
                    n or self.cfg.batch, horizon, validation
                )
            return self.replay.sample(
                n or self.cfg.batch,
                horizon,
                starts,
                validation=validation,
                stage_balanced=self.cfg.stage_balanced,
            )
        finally:
            self.replay_seconds += timestamp(self.device) - start

    def fit_normalizers(self):
        b = {k: v[0] for k, v in self.sample(n=self.cfg.normalizer_samples).items()}
        old_mean = self.obs_norm.mean.clone()
        old_std = self.obs_norm.std.clone()
        self.obs_norm.fit(self.observation(b))
        if self.privileged_bridge is not None:
            # Fit only training rows, never holdout environments.
            self.critic_norm.fit(
                self.privileged_bridge(
                    b["state"],
                    b["previous_action"],
                    b["phase"],
                    b["yaw"],
                    b["launched"],
                    b["dimensions"],
                    b["mass"],
                    b.get("default_q"),
                    b.get("default_dq"),
                )
            )
        if self.termination is not None:
            if "termination_features" not in b:
                raise ValueError(
                    "Replay lacks v3 physical termination features; collect a new baseline"
                )
            self.termination.inputs.fit(b["termination_features"])
        if self.initialization["kind"] == "ppo_actor_warm_start":
            # Preserve the warm policy when introducing normalization. Equality
            # holds in the unclipped domain; clipping can break this equality.
            with torch.no_grad():
                first = self.actor[0]
                old_weight = first.weight.clone()
                first.weight.mul_(self.obs_norm.std / old_std)
                first.bias.add_(
                    old_weight @ ((self.obs_norm.mean - old_mean) / old_std)
                )
        if self.cfg.condition_external_events:
            features = self.event_scheduler.codec.pre_features(
                b["event_state"], b["episode_steps"], self.task.dt
            )
            self.model.inputs.fit(
                self.model.features(b["state"], b["target"], b["context"], features)
            )
        elif self.model.state_only:
            self.model.inputs.fit(
                self.model.features(
                    b["state"], b["target"], b["context"], b["launched"]
                )
                if self.cfg.learn_external_events
                and not self.cfg.condition_external_events
                else self.model.features(b["state"], b["target"], b["context"])
            )
        else:
            self.model.inputs.fit(
                self.model.features(
                    b["state"],
                    b["target"],
                    b["context"],
                    b["memory"],
                    b["phase"],
                    b["next_blend"],
                    self.observation(b),
                    b["action"],
                )
            )
            self.model.reward_norm.fit(b["reward_terms"])
        self.normalized = True

    def fit_model(self):
        start = timestamp(self.device)
        if not self.normalized:
            self.fit_normalizers()
        probe = {}
        if self.cfg.joint_predicted_termination:
            for group in self.termination_opt.param_groups:
                group["lr"] = self.model_opt.param_groups[0]["lr"]
        if self._termination_probe_due:
            devices = (
                [torch.device(self.device).index or 0]
                if str(self.device).startswith("cuda")
                else []
            )
            with torch.random.fork_rng(devices=devices):
                row = {k: v[0] for k, v in self.sample(1, n=8).items()}
                ns = (
                    self.model.predict_physical(
                        row["state"],
                        row["target"],
                        row["context"],
                        self.event_scheduler.codec.pre_features(
                            row["event_state"], row["episode_steps"], self.task.dt
                        ),
                    )
                    if self.cfg.condition_external_events
                    else self.model.predict_all(
                        row["state"], row["target"], row["context"], row["launched"]
                    )[0]
                    if self.cfg.learn_external_events
                    else self.model(row["state"], row["target"], row["context"])
                )
                features = self.task.predicted_termination_features(
                    row, row["state"], ns, row["phase"]
                )
                bce = nn.functional.binary_cross_entropy_with_logits(
                    self.termination(features), row["terminated"].float()
                )
                parameters = tuple(self.model.parameters())
                grads = torch.autograd.grad(
                    self.cfg.term_loss_weight * bce, parameters, allow_unused=True
                )
                norm = (
                    torch.stack([g.square().sum() for g in grads if g is not None])
                    .sum()
                    .sqrt()
                )
                probe = {
                    "termination/probe_bce": float(bce.detach()),
                    "termination/probe_weighted_bce_wm_gradient_norm": float(norm),
                }
            self._termination_probe_due = False
        last = {}
        for _ in range(self.cfg.model_steps):
            seq = self.sample(self.cfg.model_window)
            s, m = seq["state"][0], seq["memory"][0]
            losses = []
            state_losses = []
            reward_losses = []
            termination_losses = []
            physics_losses, macro_losses, mode_losses = [], [], []
            mode = seq["launched"][0].float()
            mask = seq.get("loss_mask")

            def window_mean(values):
                values = torch.stack(values)
                if mask is not None:
                    return (
                        torch.where(mask, values, 0).sum(0) / mask.sum(0).clamp_min(1)
                    ).mean()
                return values.mean()

            for t in range(self.cfg.model_window):
                if self.model.state_only:
                    if mask is not None:
                        s = torch.where(mask[t, :, None], s, seq["state"][t])
                    if self.cfg.condition_external_events:
                        # Conditional supervised rollout: recorded event inputs
                        # align the stochastic branch with measured labels.
                        # Actor/free evaluation never consumes these future inputs.
                        pre = self.event_scheduler.codec.pre_features(
                            seq["event_state"][t], seq["episode_steps"][t], self.task.dt
                        )
                        physical = self._dynamics(
                            s, seq["target"][t], seq["context"][t], pre
                        )
                        ns = self._post(
                            physical, seq["event_targets"][t], seq["event_mask"][t]
                        )
                    elif self.cfg.learn_external_events:
                        if mask is not None:
                            mode = torch.where(
                                mask[t], mode, seq["launched"][t].float()
                            )
                        physical, ns, mode_logits = self._dynamics(
                            s, seq["target"][t], seq["context"][t], mode
                        )
                        mode = mode_logits.sigmoid()
                    else:
                        ns = self._dynamics(s, seq["target"][t], seq["context"][t])
                        physical = ns
                    squared = state_error(ns, seq["next_state"][t]).square()
                    sl = (
                        squared.sum(-1)
                        if self.cfg.state_loss_reduction == "sum"
                        else squared.mean(-1)
                    )
                    if self.cfg.learn_external_events:
                        macro_losses.append(sl)
                        pl = (
                            state_error(physical, seq["physical_next_state"][t])
                            .square()
                            .sum(-1)
                        )
                        physics_losses.append(pl)
                        ml = (
                            torch.zeros_like(pl)
                            if self.cfg.condition_external_events
                            else nn.functional.binary_cross_entropy_with_logits(
                                mode_logits,
                                seq["after_launched"][t].float(),
                                reduction="none",
                            )
                        )
                        mode_losses.append(ml)
                        sl = pl if self.cfg.deterministic_event_apply else sl + pl
                    if self.cfg.joint_predicted_termination:
                        row = {k: v[t] for k, v in seq.items()}
                        features = self.task.predicted_termination_features(
                            row,
                            s,
                            physical,
                            row["phase"],
                            initial=t == 0,
                            fk_before=self._fk(s),
                            fk_after=self._fk(physical),
                        )
                        dl = nn.functional.binary_cross_entropy_with_logits(
                            self.termination(features),
                            row["terminated"].float(),
                            reduction="none",
                        )
                        termination_losses.append(dl)
                        combined = sl + self.cfg.term_loss_weight * dl
                        losses.append(
                            combined + ml
                            if self.cfg.learn_external_events
                            else combined
                        )
                    else:
                        losses.append(sl)
                    state_losses.append(sl)
                    s = ns
                    continue
                ns, nm, r, d = self.model(
                    s,
                    seq["target"][t],
                    seq["context"][t],
                    m,
                    seq["phase"][t],
                    seq["next_blend"][t],
                    self.observation({k: v[t] for k, v in seq.items()}, s),
                    seq["action"][t],
                )
                sl = state_error(ns, seq["next_state"][t]).square().mean()
                rl = (
                    ((r - seq["reward_terms"][t]) / self.model.reward_norm.std)
                    .square()
                    .mean()
                )
                ml = (nm - seq["next_memory"][t]).square().mean()
                dl = nn.functional.binary_cross_entropy_with_logits(
                    d, seq["terminated"][t].float()
                )
                losses.append(sl + rl + 0.1 * ml + dl)
                state_losses.append(sl)
                reward_losses.append(rl)
                s, m = ns, nm
            loss = window_mean(losses)
            self.model_opt.zero_grad(set_to_none=True)
            if self.cfg.joint_predicted_termination:
                self.termination_opt.zero_grad(set_to_none=True)
            loss.backward()
            parameters = list(self.model.parameters())
            if self.cfg.joint_predicted_termination:
                parameters += list(self.termination.parameters())
            nn.utils.clip_grad_norm_(
                parameters,
                self.cfg.model_grad_clip,
                error_if_nonfinite=True,
            )
            self.model_opt.step()
            if self.cfg.joint_predicted_termination:
                self.termination_opt.step()
                self.termination_updates += 1
            self.model_updates += 1
            last = {
                "wm/loss": float(loss.detach()),
                "wm/state_loss": float(window_mean(state_losses).detach()),
                "wm/last_step_state_loss": float(sl.mean().detach()),
            }
            if physics_losses:
                last["wm/physical_state_loss"] = float(
                    window_mean(physics_losses).detach()
                )
                last["wm/post_control_state_loss"] = float(
                    window_mean(macro_losses).detach()
                )
                last["wm/post_control_in_objective"] = float(
                    not self.cfg.deterministic_event_apply
                )
                if not self.cfg.condition_external_events:
                    last["wm/launch_mode_bce"] = float(
                        window_mean(mode_losses).detach()
                    )
                last["wm/external_write_transition_fraction"] = (
                    float(seq["external_boundary"].float()[mask].mean())
                    if mask is not None
                    else float(seq["external_boundary"].float().mean())
                )
            if termination_losses:
                last["termination/train_bce"] = float(
                    window_mean(termination_losses).detach()
                )
                last["termination/bce_updates_wm"] = 1.0
            if mask is not None:
                last["wm/last_step_state_loss"] = float(
                    (sl.detach() * mask[-1]).sum() / mask[-1].sum().clamp_min(1)
                )
                last["wm/last_step_samples"] = int(mask[-1].sum())
                last["wm/effective_window_steps"] = float(mask.sum(0).float().mean())
                last["wm/prelaunch_window_fraction"] = float(
                    (~seq["launched"][0].bool()).float().mean()
                )
            if reward_losses:
                last["wm/reward_loss"] = float(
                    torch.stack(reward_losses).mean().detach()
                )
        if self.termination is not None and not self.cfg.joint_predicted_termination:
            for _ in range(self.cfg.model_steps):
                b = {k: v[0] for k, v in self.sample(1).items()}
                logits = self.termination(b["termination_features"].detach())
                loss = nn.functional.binary_cross_entropy_with_logits(
                    logits, b["terminated"].float()
                )
                self.termination_opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.termination.parameters(), 1, error_if_nonfinite=True
                )
                self.termination_opt.step()
                self.termination_updates += 1
            last["termination/train_bce"] = float(loss.detach())
        last.update(probe)
        end_model = timestamp(self.device)
        last["time/model_fit_seconds"] = end_model - start
        last["amp/discriminator_loss"] = self.prior.train_discriminator()
        last["time/amp_update_seconds"] = timestamp(self.device) - end_model
        last["amp/ramp"] = self.prior.ramp
        return last

    @torch.no_grad()
    def validate(self):
        if self.cfg.condition_external_events:
            from .events_v24.evaluation import evaluate

            return evaluate(self, batch=min(self.cfg.batch, 64))
        if self.cfg.learn_external_events:
            return self.validate_macro()
        if self.model.state_only:
            from .offline import evaluate

            batches = {}
            for h in (1, 4, 8, 16, 32):
                try:
                    batches[h] = self.sample(h, validation=True)
                except ValueError:
                    pass
            metrics = {
                f"validation/{h}/{key}": value
                for h, row in evaluate(self.model, None, batches).items()
                for key, value in row.items()
                if isinstance(value, (int, float))
            }
            if self.termination is not None:
                from .diagnostics import calibration

                b = {k: v[0] for k, v in self.sample(1, validation=True).items()}
                features = b["termination_features"]
                if self.cfg.joint_predicted_termination:
                    ns = self.model(b["state"], b["target"], b["context"])
                    features = self.task.predicted_termination_features(
                        b, b["state"], ns, b["phase"]
                    )
                metrics.update(calibration(self.termination(features), b["terminated"]))
            return metrics
        metrics = {}
        for h in (1, 4, 8, 16):
            try:
                seq = self.sample(h, validation=True)
            except ValueError:
                metrics[f"validation/{h}/available"] = 0.0
                continue
            s, m = seq["state"][0], seq["memory"][0]
            for t in range(h):
                s, m, r, _ = self.model(
                    s,
                    seq["target"][t],
                    seq["context"][t],
                    m,
                    seq["phase"][t],
                    seq["next_blend"][t],
                    self.observation({k: v[t] for k, v in seq.items()}, s),
                    seq["action"][t],
                )
            error = state_error(s, seq["next_state"][-1]).square().mean(-1).sqrt()
            metrics[f"validation/{h}/state_rmse"] = float(error.mean())
            # Four geometric bins are diagnostics, not verified contact labels.
            for stage in range(4):
                mask = seq["stage"][0] == stage
                if bool(mask.any()):
                    metrics[f"validation/{h}/stage{stage}_rmse"] = float(
                        error[mask].mean()
                    )
            per_term = (
                ((r - seq["reward_terms"][-1]) / self.model.reward_norm.std)
                .square()
                .mean(0)
                .sqrt()
            )
            metrics[f"validation/{h}/reward_rmse"] = float(per_term.mean())
            if h == 1:
                for name, value in zip(self.reward_names, per_term):
                    metrics[f"reward_error/{name}"] = float(value)
        return metrics

    @torch.no_grad()
    def validate_macro(self):
        from .diagnostics import calibration

        metrics = {}
        for h in (1, 4, 8, 16, 32):
            try:
                seq = self.replay.sample(self.cfg.batch, h, validation=True)
            except ValueError:
                continue
            s, mode = seq["state"][0], seq["launched"][0].float()
            errors, physical_errors, mode_errors = [], [], []
            for j in range(h):
                physical, ns, logits = self.model.predict_all(
                    s, seq["target"][j], seq["context"][j], mode
                )
                errors.append(state_error(ns, seq["next_state"][j]).square())
                physical_errors.append(
                    state_error(physical, seq["physical_next_state"][j]).square()
                )
                mode_errors.append(
                    nn.functional.binary_cross_entropy_with_logits(
                        logits, seq["after_launched"][j].float()
                    )
                )
                if h == 1:
                    row = {k: v[0] for k, v in seq.items()}
                    f = self.task.predicted_termination_features(
                        row, s, physical, row["phase"]
                    )
                    metrics.update(calibration(self.termination(f), row["terminated"]))
                s, mode = ns, logits.sigmoid()
            metrics[f"validation/{h}/state_rmse"] = float(
                torch.stack(errors).mean().sqrt()
            )
            metrics[f"validation/{h}/physical_state_rmse"] = float(
                torch.stack(physical_errors).mean().sqrt()
            )
            metrics[f"validation/{h}/launch_mode_bce"] = float(
                torch.stack(mode_errors).mean()
            )
        return metrics

    def basic_ready(self, metrics):
        if self.model.state_only:
            value = metrics.get("validation/8/state_rmse", float("inf"))
            return bool(np.isfinite(value) and value < self.cfg.max_state_rmse)
        return all(
            metrics.get(f"validation/{h}/state_rmse", float("inf"))
            < self.cfg.max_state_rmse
            and metrics.get(f"validation/{h}/reward_rmse", float("inf"))
            < self.cfg.max_reward_rmse
            for h in (1, 4, 8, 16)
        )

    def validation_due(self):
        rounds = self.model_updates // self.cfg.model_steps
        return (rounds - 1) % self.cfg.validation_every == 0

    def schedule_lr(self):
        if self.cfg.schedule != "joint_reproduction":
            return self.cfg.model_lr or self.cfg.lr
        import math

        step = min(self.iteration + 1, self.cfg.lr_decay_steps)
        ratio = self.cfg.lr_final_ratio + (1 - self.cfg.lr_final_ratio) * 0.5 * (
            1 + math.cos(math.pi * step / self.cfg.lr_decay_steps)
        )
        for opt, base in (
            (self.model_opt, self.cfg.model_lr or self.cfg.lr),
            (self.actor_opt, self.cfg.lr),
            (self.critic_opt, self.cfg.lr),
        ):
            for group in opt.param_groups:
                group["lr"] = base * ratio
        if self.cfg.joint_predicted_termination:
            for group in self.termination_opt.param_groups:
                group["lr"] = (self.cfg.model_lr or self.cfg.lr) * ratio
        return (self.cfg.model_lr or self.cfg.lr) * ratio

    def updates_due(self):
        if self.cfg.schedule != "joint_reproduction":
            return 1
        import math

        j = self.iteration + 1 - self.cfg.initial_collect_steps
        if j < 0:
            return 0
        rate = self.cfg.updates_per_env_step
        return math.floor((j + 1) * rate + 1e-9) - math.floor(j * rate + 1e-9)

    def ready(self, metrics, ranking_approved=False):
        if self.cfg.schedule == "joint_reproduction":
            return True
        # A real-simulator action-ranking report is a separate required gate.
        return (
            ranking_approved
            and self.iteration >= self.cfg.warmup_iterations
            and self.basic_ready(metrics)
        )

    def imagined_observations(self, b, s, previous, phase, poses):
        raw = self.observation(b, s, previous, phase, poses)
        critic = self.critic_observation(b, s, previous, phase, poses, raw)
        return raw, critic

    def transition(
        self,
        b,
        s,
        target,
        raw,
        previous,
        phase,
        hist,
        episode,
        event,
        poses,
        step,
        first,
    ):
        ns = self._dynamics(s, target, b["context"])
        next_poses = self._fk(ns)
        task = self._task_step(
            b,
            s,
            ns,
            raw,
            previous,
            phase,
            hist,
            episode,
            event,
            step,
            fk_before=poses,
            fk_after=next_poses,
            initial_step=first,
        )
        d = (
            self._termination_fn(task.termination_features).sigmoid()
            if self.termination is not None
            else task.terminated.float()
        )
        return ns, next_poses, task, d

    def imagine(self, start_batch=None):
        b = self.sample(starts=True) if start_batch is None else start_batch
        if self.model.state_only and self.task is None:
            raise RuntimeError("Pure dynamics needs an AnalyticTask for imagination")
        s, m, blend = b["state"], b["memory"], b["blend"]
        mode = b["launched"].float()
        event_state = b.get("event_state")
        event_pool = b.get("event_pool")
        common = b["common_step"] if self.cfg.condition_external_events else None
        if self.cfg.learn_external_events:
            b = {**b, "learn_external_events": True}
        if self.cfg.condition_external_events:
            b = {**b, "event_conditioned": True}
        previous, phase = b["previous_action"], b["phase"]
        observations = []
        critic_observations = []
        rewards = []
        dones = []
        terms = []
        styles = []
        validity = []
        truncations = []
        predicted_modes = []
        valid = torch.ones(len(s), dtype=torch.bool, device=s.device)
        hist = b.get("task_history")
        episode = b.get("episode_steps")
        event = b.get("next_external_event_seconds")
        if self._transition is not None and not self._transition_cache_ready:
            # Original rewards initialize constant body/joint indices and log
            # once. Do that outside fullgraph; discard this local view/history.
            with torch.no_grad():
                self.task.evaluate_step(
                    b, s, s, previous, previous, phase, hist, episode, event, 0
                )
            self._transition_cache_ready = True
        poses = self._fk(s)
        for t in range(self.cfg.horizon):
            with torch.profiler.record_function("imagine/observation"):
                raw_obs, critic_obs = self._observations(b, s, previous, phase, poses)
                critic_observations.append(critic_obs)
            obs = self.obs_norm(raw_obs)
            observations.append(obs)
            raw = self.actor(obs)
            target, blend = self.targets(b, raw, blend)
            if self.model.state_only:
                if self._transition is not None:
                    ns, next_poses, task, d = self._transition(
                        b,
                        s,
                        target,
                        raw,
                        previous,
                        phase,
                        hist,
                        episode,
                        event,
                        poses,
                        torch.tensor(t, device=s.device),
                        t == 0,
                    )
                    r, trunc = task.rewards, task.truncated
                    physical, physical_poses = ns, next_poses
                else:
                    with torch.profiler.record_function("imagine/dynamics"):
                        if self.cfg.condition_external_events:
                            pre = self.event_scheduler.codec.pre_features(
                                event_state, episode, self.task.dt
                            )
                            physical = self._dynamics(s, target, b["context"], pre)
                            ns = physical
                        elif self.cfg.learn_external_events:
                            physical, ns, mode_logits = self._dynamics(
                                s, target, b["context"], mode
                            )
                        else:
                            ns = self._dynamics(s, target, b["context"])
                            physical = ns
                    with torch.profiler.record_function("imagine/FK"):
                        physical_poses = self._fk(physical)
                        next_poses = (
                            self._fk(ns)
                            if self.cfg.learn_external_events
                            and not self.cfg.condition_external_events
                            else physical_poses
                        )
                    with torch.profiler.record_function(
                        "imagine/reward_and_termination"
                    ):
                        task = self.task.evaluate_step(
                            b,
                            s,
                            physical,
                            raw,
                            previous,
                            phase,
                            hist,
                            episode,
                            event,
                            t,
                            fk_before=poses,
                            fk_after=physical_poses,
                        )
                    r, d, trunc = task.rewards, task.terminated.float(), task.truncated
                    if self.termination is not None:
                        d = self.termination(task.termination_features).sigmoid()
                if self.cfg.condition_external_events:
                    from .events_v24.writes import WritePlan

                    active = valid & ~task.truncated & ~task.terminated
                    ids = active.nonzero().flatten()
                    plan = WritePlan(physical)
                    next_event, next_pool = event_state, event_pool
                    if len(ids):
                        event_b = {
                            **b,
                            "phase": task.phase,
                            "task_history": task.history,
                        }
                        event_b = {
                            k: (
                                v[ids]
                                if torch.is_tensor(v) and v.ndim and len(v) == len(s)
                                else v
                            )
                            for k, v in event_b.items()
                        }
                        pp = tuple(
                            {k: v[ids] for k, v in group.items()}
                            for group in physical_poses
                        )
                        child, ce, cp = self.event_scheduler.step(
                            event_b,
                            physical[ids],
                            pp,
                            event_state[ids],
                            event_pool[ids],
                            task.episode_steps[ids],
                            common[ids] + 1,
                        )
                        plan.merge(ids, child)
                        next_event = event_state.index_copy(0, ids, ce)
                        next_pool = event_pool.index_copy(0, ids, cp)
                    ns = self._post(physical, plan.targets, plan.mask)
                    next_poses = self._fk(ns)
                    event_state = torch.where(valid[:, None], next_event, event_state)
                    event_pool = torch.where(
                        valid[:, None, None], next_pool, event_pool
                    )
                    b = {**b, "launched": self.event_scheduler.codec.mode(event_state)}
                    history_values, _ = self.event_scheduler.codec.decode(event_state)
                    task.history = task.history.clone()
                    from .task import HISTORY_FIELDS

                    for key in ("_box_regrace", "_v38_cradle_ticks"):
                        task.history[:, HISTORY_FIELDS.index(key)] = history_values[
                            key
                        ].float()
                    common = torch.where(valid, common + 1, common)
                hist = torch.where(valid[:, None], task.history, hist)
                episode = torch.where(valid, task.episode_steps, episode)
                event = torch.where(valid, task.event_seconds, event)
                next_phase = task.phase
            else:
                ns, m, r, d = self.model(
                    s, target, b["context"], m, phase, blend, raw_obs, raw
                )
                d = torch.sigmoid(d)
                trunc = phase + 1 >= b["reference_length"]
                next_phase = (phase + 1).minimum(b["reference_length"] - 1)
            with torch.profiler.record_function("imagine/AMP"):
                style = self.prior.reward(
                    amp_features(s, self.bridge.fk, poses),
                    amp_features(
                        physical if self.model.state_only else ns,
                        self.bridge.fk,
                        physical_poses if self.model.state_only else None,
                    ),
                )
            rewards.append(
                self.prior._w_task * r.sum(-1)
                + self.prior._w_style * self.prior.ramp * style
            )
            dones.append(d)
            terms.append(r)
            styles.append(style)
            validity.append(valid)
            truncations.append(trunc)
            if self.model.state_only:
                poses = tuple(
                    {k: torch.where(valid[:, None], pred[k], old[k]) for k in old}
                    for old, pred in zip(poses, next_poses)
                )
            else:
                poses = self._fk(torch.where(valid[:, None], ns, s))
            s = torch.where(valid[:, None], ns, s)
            if (
                self.cfg.learn_external_events
                and not self.cfg.condition_external_events
            ):
                mode = torch.where(valid, mode_logits.sigmoid(), mode)
                predicted_modes.append(mode)
                b = {**b, "launched": mode.detach() > 0.5}
            if self.cfg.condition_external_events:
                predicted_modes.append(
                    self.event_scheduler.codec.mode(event_state).float()
                )
            previous = torch.where(valid[:, None], raw, previous)
            phase = torch.where(valid, next_phase, phase)
            valid = valid & ~trunc & ~(d.detach() >= 1)
        final_obs = self.observation(b, s, previous, phase, poses)
        observations.append(self.obs_norm(final_obs))
        critic_observations.append(
            self.critic_observation(b, s, previous, phase, poses, final_obs)
        )
        return {
            "predicted_launch_probability": torch.stack(predicted_modes)
            if predicted_modes
            else torch.empty(0, device=self.device),
            "obs": torch.stack(observations),
            "critic_obs": torch.stack(critic_observations),
            "reward": torch.stack(rewards),
            "done": torch.stack(dones),
            "terms": torch.stack(terms),
            "style": torch.stack(styles),
            "valid": torch.stack(validity),
            "truncated": torch.stack(truncations),
        }

    def update_actor(self):
        if (
            self.cfg.actor_microbatch
            or self.cfg.privileged_critic
            or self.cfg.learned_termination
        ):
            from .joint import update_actor

            return update_actor(self)
        metrics = {}
        imagine_seconds = actor_seconds = critic_seconds = 0.0
        for _ in range(self.cfg.actor_steps):
            with freeze(self.model, self.prior._disc, self.critic):
                start = timestamp(self.device)
                imagined = self.imagine()
                obs, reward, done, terms, styles = (
                    imagined["obs"],
                    imagined["reward"],
                    imagined["done"],
                    imagined["terms"],
                    imagined["style"],
                )
                after_imagine = timestamp(self.device)
                imagine_seconds += after_imagine - start
                survival = torch.ones_like(reward[0])
                objective = torch.zeros_like(survival)
                reward_weights = []
                for t in range(self.cfg.horizon):
                    reward_weight = self.cfg.gamma**t * survival
                    reward_weights.append(reward_weight)
                    objective = objective + reward_weight * reward[t]
                    continuation = survival * (1 - done[t])
                    objective = objective + self.cfg.gamma ** (
                        t + 1
                    ) * continuation * imagined["truncated"][t] * self.critic(
                        obs[t + 1]
                    ).squeeze(-1)
                    survival = continuation * (~imagined["truncated"][t])
                objective = (
                    objective
                    + self.cfg.gamma** self.cfg.horizon
                    * survival
                    * self.critic(obs[-1]).squeeze(-1)
                )
                loss = -objective.mean()
                # Audit task/style gradients on shared actor params; learned reward
                # gradient can be misleading even with good scalar prediction loss.
                params = list(self.actor.parameters())
                weights = torch.stack(reward_weights)
                contribution = lambda values: (weights * values).sum(0).mean()
                audits = [
                    ("task", self.prior._w_task * contribution(terms.sum(-1))),
                    (
                        "style",
                        self.prior._w_style * self.prior.ramp * contribution(styles),
                    ),
                ]
                audits.extend(
                    (f"reward/{name}", self.prior._w_task * contribution(terms[..., i]))
                    for i, name in enumerate(self.reward_names)
                )
                audit = (
                    self.cfg.gradient_audit_every > 0
                    and self.actor_updates % self.cfg.gradient_audit_every == 0
                )
                for name, value in audits if audit else []:
                    grads = (
                        torch.autograd.grad(
                            value, params, retain_graph=True, allow_unused=True
                        )
                        if value.requires_grad
                        else []
                    )
                    norms = [g.norm() for g in grads if g is not None]
                    metrics[f"gradient/{name}"] = (
                        float(torch.stack(norms).norm()) if norms else 0.0
                    )
                self.actor_opt.zero_grad(set_to_none=True)
                loss.backward()
                norm = nn.utils.clip_grad_norm_(params, 1, error_if_nonfinite=True)
                self.actor_opt.step()
            self.actor_updates += 1
            after_actor = timestamp(self.device)
            actor_seconds += after_actor - after_imagine
            with torch.no_grad():
                values = self.critic(obs.detach()).squeeze(-1)
                ret = values[-1]
                targets = []
                for t in reversed(range(self.cfg.horizon)):
                    future = torch.where(imagined["truncated"][t], values[t + 1], ret)
                    ret = reward[t].detach() + self.cfg.gamma * (
                        1 - done[t].detach()
                    ) * ((1 - self.cfg.lam) * values[t + 1] + self.cfg.lam * future)
                    targets.append(ret)
                target = torch.stack(targets[::-1])
            closs = (self.critic(obs[:-1].detach()).squeeze(-1) - target).square()
            closs = (closs * imagined["valid"]).sum() / imagined[
                "valid"
            ].sum().clamp_min(1)
            self.critic_opt.zero_grad(set_to_none=True)
            closs.backward()
            self.critic_opt.step()
            critic_seconds += timestamp(self.device) - after_actor
            metrics.update(
                {
                    "actor/loss": float(loss.detach()),
                    "actor/gradient_norm": float(norm),
                    "critic/loss": float(closs.detach()),
                    "imagination/reward": float(reward.detach().mean()),
                }
            )
        metrics.update(
            {
                "time/imagination_seconds": imagine_seconds,
                "time/actor_update_seconds": actor_seconds,
                "time/critic_update_seconds": critic_seconds,
            }
        )
        return metrics

    @property
    def learning_contract(self):
        result = {
            "critic_input_dim": self.critic_dim,
            "critic_layout": "reflex_privileged292_v1"
            if self.cfg.privileged_critic
            else "oracle_policy179_v1",
            "termination_geometry": "anchor_world3+gravity3+tracked_body_error_world3_signed_abs_v1"
            if self.termination is not None
            else None,
            "termination_feature_dim": self.task.termination_feature_dim
            if self.termination is not None
            else None,
        }

        if self.cfg.condition_external_events:
            result["event_scheduler"] = self.event_scheduler.specification()
        if self.cfg.learn_external_events:
            result["state_transition"] = (
                "learned_physics_and_post_control_with_next_mode_v1"
            )
        if self.cfg.deterministic_event_apply:
            result["state_transition"] = (
                "physical_nn_native_scheduler_deterministic_apply_v1"
            )
            result["event_supervision"] = self.cfg.event_supervision_protocol
        if self.cfg.joint_predicted_termination:
            result["termination_training"] = (
                "autoregressive_predicted_features_joint_bce_v1"
            )
            result["state_loss_reduction"] = self.cfg.state_loss_reduction
            result["term_loss_weight"] = self.cfg.term_loss_weight
            result["variable_model_windows"] = self.cfg.variable_model_windows
        return result

    def invalidate_contact_history(self, old_schema):
        # Physics/reward/termination labels are valid; only old sensor telemetry
        # was wrong. Keep model/optimizer/replay physics, mark contacts unknown.
        for key, value in (
            ("contact_known", False),
            ("box_contact", False),
            ("contact_phase", -1),
            ("start_contact_phase", -1),
        ):
            if key in self.replay.data:
                self.replay.data[key].fill_(value)
        from .experience import ExperienceMetrics

        self.experience = ExperienceMetrics(monitor_contacts=self.cfg.monitor_contacts)
        self.contact_filter_migration = {
            "from": old_schema,
            "to": self.contact_filter_schema,
            "invalidated_vector_steps": self.replay.count,
            "experience_totals_reset": True,
        }

    def save(self, path):
        state = {
            "version": 3
            if (self.cfg.privileged_critic or self.cfg.learned_termination)
            else 2,
            "event_rng": self.event_scheduler.rng
            if self.event_scheduler is not None
            else None,
            "model_kind": self.cfg.wm_kind,
            "sampling_policy": "all_valid_states_uniform_with_replacement_v2",
            "contact_filter_schema": self.contact_filter_schema,
            "contact_filter_migration": self.contact_filter_migration,
            "learning_contract": self.learning_contract,
            "config": asdict(self.cfg),
            "contract": self.contract,
            "reward_names": self.reward_names,
            "iteration": self.iteration,
            "real_transitions": self.real_transitions,
            "probe_transitions": self.probe_transitions,
            "actor_updates": self.actor_updates,
            "model_updates": self.model_updates,
            "last_diagnostic_seconds": self.last_diagnostic_seconds,
            "termination_updates": self.termination_updates,
            "initialization": self.initialization,
            "experience": self.experience.state_dict(),
            "normalized": self.normalized,
            "task_spec": self.task.specification() if self.task is not None else None,
            "task_state": self.task.state_dict() if self.task is not None else None,
            "critic_norm": self.critic_norm.state_dict(),
            "termination": self.termination.state_dict()
            if self.termination is not None
            else None,
            "termination_opt": self.termination_opt.state_dict()
            if self.termination_opt is not None
            else None,
            "replay": self.replay.state_dict(),
            "amp": self.prior.amp_state_dict(),
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
            "cuda_rng": torch.cuda.get_rng_state_all()
            if torch.cuda.is_initialized()
            else None,
        }
        for name in (
            "actor",
            "critic",
            "model",
            "obs_norm",
            "bridge",
            "action_map",
            "actor_opt",
            "critic_opt",
            "model_opt",
        ):
            state[name] = getattr(self, name).state_dict()
        # Atomic replacement prevents an interrupted save leaving a valid-looking ckpt.
        temporary = str(path) + ".tmp"
        torch.save(state, temporary)
        import os

        os.replace(temporary, path)

    def load(self, path, migrate_termination_training=False):
        state = torch.load(path, map_location=self.device, weights_only=False)
        saved_config = dict(state["config"])
        if state.get("version") == 1:
            saved_config.setdefault("wm_kind", "multitask")
        loaded_config = asdict(TrainConfig(**saved_config))
        requested_config = asdict(self.cfg)
        # Execution optimizations and optional telemetry do not change weights,
        # optimizers or task semantics. Contact metadata is migrated separately.
        for key in (
            "compile_dynamics",
            "compile_fk",
            "compile_backend",
            "compile_step",
            "monitor_contacts",
            "minimal_event_view",
        ):
            loaded_config.pop(key, None)
            requested_config.pop(key, None)
        migration = None
        if migrate_termination_training:
            if (
                state.get("version") != 3
                or not self.cfg.joint_predicted_termination
                or loaded_config["joint_predicted_termination"]
            ):
                raise ValueError(
                    "Explicit migration requires old separate-head v3 to joint predicted training"
                )
            migration = {
                "from": "separate_real_features",
                "to": "joint_predicted_features",
                "source": str(path),
                "optimizers_preserved": True,
            }
            for key in (
                "joint_predicted_termination",
                "variable_model_windows",
                "state_loss_reduction",
                "term_loss_weight",
            ):
                loaded_config.pop(key)
                requested_config.pop(key)
        if (
            state.get("version") not in (1, 2, 3)
            or state["contract"] != self.contract
            or loaded_config != requested_config
            or state["reward_names"] != self.reward_names
        ):
            raise ValueError(
                "Checkpoint model schema/config/physical contract mismatch; v1 multitask cannot silently resume pure dynamics"
            )
        requested_learning = dict(self.learning_contract)
        if migration:
            for key in (
                "termination_training",
                "variable_model_windows",
                "state_loss_reduction",
                "term_loss_weight",
            ):
                requested_learning.pop(key, None)
        if (
            state.get("version") == 3
            and state.get("learning_contract") != requested_learning
        ):
            raise ValueError("Checkpoint learning layout mismatch")
        if self.cfg.privileged_critic or self.cfg.learned_termination:
            if state.get("version") != 3 or "critic_norm" not in state:
                raise ValueError(
                    "Checkpoint schema mismatch: repaired baseline needs v3 critic and termination states"
                )
        if self.cfg.learned_termination and (
            state.get("termination") is None or state.get("termination_opt") is None
        ):
            raise ValueError("Checkpoint missing learned termination state/optimizer")
        task_spec = self.task.specification() if self.task is not None else None
        if state.get("task_spec") != task_spec:
            raise ValueError("Checkpoint analytic task contract mismatch")
        for name in (
            "actor",
            "critic",
            "model",
            "obs_norm",
            "bridge",
            "action_map",
            "actor_opt",
            "critic_opt",
            "model_opt",
        ):
            getattr(self, name).load_state_dict(state[name])
        if "critic_norm" in state:
            self.critic_norm.load_state_dict(state["critic_norm"])
        if self.termination is not None:
            if state.get("version") != 3 or state.get("termination") is None:
                raise ValueError(
                    "Repair baseline requires v3 termination/critic checkpoint; old policies cannot resume"
                )
            self.termination.load_state_dict(state["termination"])
            self.termination_opt.load_state_dict(state["termination_opt"])
        self.replay.load_state_dict(state["replay"])
        if migration:
            # Old adapter validity included launched, not only finite state.
            count = self.replay.count
            data = self.replay.data
            finite = torch.ones_like(data["valid"][:count])
            for key in (
                "state",
                "next_state",
                "context",
                "memory",
                "next_memory",
                "reward_terms",
                "target",
            ):
                finite &= torch.isfinite(data[key][:count]).flatten(2).all(-1)
            recovered = finite & ~data["valid"][:count]
            data["valid"][:count].copy_(finite)
            if "start_valid" in data:
                data["start_valid"][:count].copy_(
                    torch.isfinite(data["state"][:count]).all(-1)
                    & torch.isfinite(data["context"][:count]).all(-1)
                )
            migration["recovered_finite_transitions"] = int(recovered.sum())
        if self.task is not None:
            self.task.load_state_dict(state["task_state"])
        self.prior.load_amp_state_dict(state["amp"])
        # A new simulator starts on resume. Never join its first trajectory to
        # the final stored trajectory from the previous process.
        if self.replay.count:
            last = (self.replay.pointer - 1) % self.replay.capacity
            self.replay.data["boundary"][last].fill_(True)
        self.iteration, self.real_transitions = (
            state["iteration"],
            state["real_transitions"],
        )
        if "experience" in state:
            self.experience.load_state_dict(state["experience"])
        from .diagnostics import saved_diagnostic_duration

        self.last_diagnostic_seconds = saved_diagnostic_duration(state, path)
        if (
            self.contact_filter_schema is not None
            and state.get("contact_filter_schema") != self.contact_filter_schema
        ):
            self.invalidate_contact_history(state.get("contact_filter_schema"))
        self.normalized = state["normalized"]
        if self.event_scheduler is not None:
            self.event_scheduler.rng = state["event_rng"].cpu()
        self.initialization = dict(state.get("initialization", {"kind": "scratch"}))
        if migration:
            self.initialization["training_migration"] = migration
        self.probe_transitions = state["probe_transitions"]
        self.actor_updates = state["actor_updates"]
        self.model_updates = state.get("model_updates", 0)
        self.termination_updates = state.get("termination_updates", 0)
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"].cpu())
        if state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda_rng"]])
