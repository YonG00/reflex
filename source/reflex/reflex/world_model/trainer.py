"""Collect → fit joint WM / real AMP → imagine → pathwise actor / λ critic.

No PPO loss or likelihood ratios. All model/discriminator weights are frozen
during actor backprop; their derivatives with respect to inputs remain enabled.
"""

from __future__ import annotations
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import random
import numpy as np
import torch
from torch import nn
from .models import WorldModel, Standardizer, mlp
from .physics import amp_features, state_error
from .replay import Replay
from .timing import timestamp


@dataclass
class TrainConfig:
    capacity_steps: int = 1024
    collect_steps: int = 24
    model_steps: int = 64
    model_window: int = 4
    batch: int = 256
    horizon: int = 32
    actor_steps: int = 4
    warmup_iterations: int = 24
    gamma: float = 0.99
    lam: float = 0.95
    lr: float = 3e-4
    exploration: float = 0.15
    hidden: int = 256
    actor_hidden_dims: list[int] | None = None
    stage_balanced: bool = False
    # Normalized physical error / normalized per-term reward error. Provisional
    # acceptance limits, not literature-backed guarantees of safe imagination.
    max_state_rmse: float = 1.0
    max_reward_rmse: float = 1.0


@contextmanager
def freeze(*modules):
    params = [(p, p.requires_grad) for m in modules for p in m.parameters()]
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
    ):
        self.cfg, self.device = cfg or TrainConfig(), device
        c = self.cfg
        self.bridge, self.action_map, self.prior = (
            bridge.to(device),
            action_map.to(device),
            prior,
        )
        self.reward_names, self.contract = list(reward_names), contract
        self.model = WorldModel(
            context_dim,
            memory_dim,
            len(reward_names),
            hidden=c.hidden,
            com_indices=contract.get("com_indices"),
        ).to(device)
        self.obs_norm = Standardizer(179).to(device)
        self.actor = mlp(179, 29, c.hidden, layers=c.actor_hidden_dims).to(device)
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)
        self.critic = mlp(179, 1, c.hidden).to(device)
        self.model_opt = torch.optim.Adam(self.model.parameters(), lr=c.lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=c.lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=c.lr)
        self.replay = Replay(c.capacity_steps, device)
        self.iteration, self.real_transitions = 0, 0
        self.probe_transitions = 0
        self.actor_updates = 0
        self.replay_seconds = 0.0
        self.normalized = False
        self.initialization = {"kind": "scratch"}

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

    def observation(self, b, s=None, previous=None, phase=None):
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
        )

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
                ok = tr["valid"] & ~tr["terminated"] & ~tr["timeout"]
                if bool(ok.any()):
                    self.prior.add_real(torch.stack((tr["amp"], tr["next_amp"]), 1)[ok])
                self.real_transitions += len(raw)
                rewards.append(tr["reward_terms"].sum(-1).mean())
        return {"real/task_reward": float(torch.stack(rewards).mean())}

    def sample(self, horizon=1, starts=False, validation=False):
        start = timestamp(self.device)
        try:
            return self.replay.sample(
                self.cfg.batch,
                horizon,
                starts,
                validation=validation,
                stage_balanced=self.cfg.stage_balanced,
            )
        finally:
            self.replay_seconds += timestamp(self.device) - start

    def fit_normalizers(self):
        b = {k: v[0] for k, v in self.sample().items()}
        old_mean = self.obs_norm.mean.clone()
        old_std = self.obs_norm.std.clone()
        self.obs_norm.fit(self.observation(b))
        if self.initialization["kind"] == "ppo_actor_warm_start":
            # Preserve the warm policy when introducing normalization. Equality
            # holds in the unclipped domain; clipping is audited on real data.
            with torch.no_grad():
                first = self.actor[0]
                old_weight = first.weight.clone()
                first.weight.mul_(self.obs_norm.std / old_std)
                first.bias.add_(
                    old_weight @ ((self.obs_norm.mean - old_mean) / old_std)
                )
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
        last = {}
        for _ in range(self.cfg.model_steps):
            seq = self.sample(self.cfg.model_window)
            s, m = seq["state"][0], seq["memory"][0]
            losses = []
            for t in range(self.cfg.model_window):
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
                s, m = ns, nm
            loss = torch.stack(losses).mean()
            self.model_opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.model.parameters(), 10, error_if_nonfinite=True
            )
            self.model_opt.step()
            last = {
                "wm/loss": float(loss.detach()),
                "wm/state_loss": float(sl.detach()),
                "wm/reward_loss": float(rl.detach()),
            }
        end_model = timestamp(self.device)
        last["time/model_fit_seconds"] = end_model - start
        last["amp/discriminator_loss"] = self.prior.train_discriminator()
        last["time/amp_update_seconds"] = timestamp(self.device) - end_model
        last["amp/ramp"] = self.prior.ramp
        return last

    @torch.no_grad()
    def validate(self):
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

    def ready(self, metrics, ranking_approved=False):
        # A real-simulator action-ranking report is a separate required gate.
        return (
            ranking_approved
            and self.iteration >= self.cfg.warmup_iterations
            and all(
                metrics.get(f"validation/{h}/state_rmse", float("inf"))
                < self.cfg.max_state_rmse
                and metrics.get(f"validation/{h}/reward_rmse", float("inf"))
                < self.cfg.max_reward_rmse
                for h in (1, 4, 8, 16)
            )
        )

    def imagine(self):
        seq = self.sample(self.cfg.horizon, starts=True)
        b = {k: v[0] for k, v in seq.items()}
        s, m, blend = b["state"], b["memory"], b["blend"]
        previous, phase = b["previous_action"], b["phase"]
        obs_list = []
        rewards = []
        dones = []
        terms = []
        styles = []
        for t in range(self.cfg.horizon):
            raw_obs = self.observation(b, s, previous, phase + t)
            obs = self.obs_norm(raw_obs)
            obs_list.append(obs)
            raw = self.actor(obs)
            target, blend = self.targets(b, raw, blend)
            ns, m, r, d = self.model(
                s, target, b["context"], m, phase + t, blend, raw_obs, raw
            )
            style = self.prior.reward(
                amp_features(s, self.bridge.fk), amp_features(ns, self.bridge.fk)
            )
            rewards.append(
                self.prior._w_task * r.sum(-1)
                + self.prior._w_style * self.prior.ramp * style
            )
            dones.append(torch.sigmoid(d))
            terms.append(r)
            styles.append(style)
            s, previous = ns, raw
        obs_list.append(
            self.obs_norm(self.observation(b, s, previous, phase + self.cfg.horizon))
        )
        return (
            torch.stack(obs_list),
            torch.stack(rewards),
            torch.stack(dones),
            torch.stack(terms),
            torch.stack(styles),
        )

    def update_actor(self):
        metrics = {}
        imagine_seconds = actor_seconds = critic_seconds = 0.0
        for _ in range(self.cfg.actor_steps):
            with freeze(self.model, self.prior._disc, self.critic):
                start = timestamp(self.device)
                obs, reward, done, terms, styles = self.imagine()
                after_imagine = timestamp(self.device)
                imagine_seconds += after_imagine - start
                survival = torch.ones_like(reward[0])
                objective = torch.zeros_like(survival)
                for t in range(self.cfg.horizon):
                    objective = objective + self.cfg.gamma**t * survival * reward[t]
                    survival = survival * (1 - done[t])
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
                audits = [
                    ("task", self.prior._w_task * terms.sum(-1).mean()),
                    ("style", self.prior._w_style * self.prior.ramp * styles.mean()),
                ]
                audits.extend(
                    (f"reward/{name}", self.prior._w_task * terms[..., i].mean())
                    for i, name in enumerate(self.reward_names)
                )
                for name, value in audits:
                    grads = torch.autograd.grad(
                        value, params, retain_graph=True, allow_unused=True
                    )
                    metrics[f"gradient/{name}"] = float(
                        torch.stack([g.norm() for g in grads if g is not None]).norm()
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
                target = lambda_returns(
                    reward.detach(), done.detach(), values, self.cfg.gamma, self.cfg.lam
                )
            closs = (
                (self.critic(obs[:-1].detach()).squeeze(-1) - target).square().mean()
            )
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

    def save(self, path):
        state = {
            "version": 1,
            "config": asdict(self.cfg),
            "contract": self.contract,
            "reward_names": self.reward_names,
            "iteration": self.iteration,
            "real_transitions": self.real_transitions,
            "probe_transitions": self.probe_transitions,
            "actor_updates": self.actor_updates,
            "initialization": self.initialization,
            "normalized": self.normalized,
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

    def load(self, path):
        state = torch.load(path, map_location=self.device, weights_only=False)
        if (
            state.get("version") != 1
            or state["contract"] != self.contract
            or asdict(TrainConfig(**state["config"])) != asdict(self.cfg)
            or state["reward_names"] != self.reward_names
        ):
            raise ValueError("Checkpoint config / physical contract mismatch")
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
        self.replay.load_state_dict(state["replay"])
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
        self.normalized = state["normalized"]
        self.initialization = state.get("initialization", {"kind": "scratch"})
        self.probe_transitions = state["probe_transitions"]
        self.actor_updates = state["actor_updates"]
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"].cpu())
        if state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda_rng"]])
