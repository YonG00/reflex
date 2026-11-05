"""Memory-bounded Actor update with one optimizer step over the full start batch."""

import time
import torch
from torch import nn


def returns(imagined, critic, gamma):
    reward, done, obs = imagined["reward"], imagined["done"], imagined["obs"]
    survival = torch.ones_like(reward[0])
    result = torch.zeros_like(survival)
    for t in range(len(reward)):
        result = result + gamma**t * survival * reward[t]
        continuation = survival * (1 - done[t])
        result = result + gamma ** (t + 1) * continuation * imagined["truncated"][
            t
        ] * critic(obs[t + 1]).squeeze(-1)
        survival = continuation * ~imagined["truncated"][t]
    return result + gamma ** len(reward) * survival * critic(obs[-1]).squeeze(-1)


@torch.no_grad()
def critic_targets(t, imagined):
    values = t.critic(imagined["obs"].detach()).squeeze(-1)
    ret = values[-1]
    targets = []
    for j in reversed(range(t.cfg.horizon)):
        future = torch.where(imagined["truncated"][j], values[j + 1], ret)
        ret = imagined["reward"][j].detach() + t.cfg.gamma * (
            1 - imagined["done"][j].detach()
        ) * ((1 - t.cfg.lam) * values[j + 1] + t.cfg.lam * future)
        targets.append(ret)
    return torch.stack(targets[::-1])


def update_actor(t):
    from .trainer import freeze

    c = t.cfg
    if c.actor_steps != 1:
        raise ValueError("Chunked joint update requires actor_steps=1")
    if c.gradient_audit_every:
        raise ValueError(
            "Chunked Actor uses aggregate gradient norm; configure gradient_audit_every=0 explicitly"
        )
    begin = time.monotonic()
    n = c.actor_batch or c.batch
    starts = t.sample(starts=True, n=n)
    t.actor_opt.zero_grad(set_to_none=True)
    x, y = [], []
    loss_total = 0.0
    valid_steps = 0
    for offset in range(0, n, c.actor_microbatch):
        b = {k: v[offset : offset + c.actor_microbatch] for k, v in starts.items()}
        with freeze(t.model, t.prior._disc, t.critic):
            imagined = t.imagine(start_batch=b)
            loss = -returns(imagined, t.critic, c.gamma).mean()
            if c.actor_horizon_scale:
                loss = loss / c.horizon
            scaled = loss * len(b["state"]) / n
            scaled.backward()
            loss_total += float(scaled.detach())
            target = critic_targets(t, imagined)
            mask = imagined["valid"].reshape(-1)
            x.append(imagined["obs"][:-1].detach().reshape(-1, 179)[mask])
            y.append(target.reshape(-1)[mask])
            valid_steps += int(mask.sum())
        del imagined, scaled, loss, target
        if (offset + len(b["state"])) % 1024 == 0:
            print(
                f"Actor gradient trajectories {offset + len(b['state'])}/{n}",
                flush=True,
            )
    params = list(t.actor.parameters())
    norm = nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
    if c.actor_grad_clip_value:
        nn.utils.clip_grad_value_(params, c.actor_grad_clip_value)
    t.actor_opt.step()
    t.actor_updates += 1
    actor_seconds = time.monotonic() - begin
    x, y = torch.cat(x), torch.cat(y)
    if not len(y):
        raise RuntimeError("Imagination has no valid critic targets")
    losses = []
    for _ in range(c.critic_epochs):
        for idx in torch.tensor_split(
            torch.randperm(len(y), device=y.device), c.critic_minibatches
        ):
            if not len(idx):
                continue
            loss = (t.critic(x[idx]).squeeze(-1) - y[idx]).square().mean()
            t.critic_opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                t.critic.parameters(), 1.0, error_if_nonfinite=True
            )
            t.critic_opt.step()
            losses.append(float(loss.detach()))
    return {
        "actor/loss": loss_total,
        "actor/gradient_norm": float(norm),
        "critic/loss": sum(losses) / len(losses),
        "imagination/valid_fraction": valid_steps / (n * c.horizon),
        "imagination/start_batch": n,
        "time/actor_update_seconds": actor_seconds,
        "time/critic_update_seconds": time.monotonic() - begin - actor_seconds,
    }
