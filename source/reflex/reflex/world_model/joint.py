"""Memory-bounded Actor update with one optimizer step over the full start batch."""

import torch
from torch import nn

from .timing import timestamp


def returns(imagined, critic, gamma, values=None):
    reward, done, obs = (
        imagined["reward"],
        imagined["done"],
        imagined.get("critic_obs", imagined["obs"]),
    )
    survival = torch.ones_like(reward[0])
    result = torch.zeros_like(survival)
    for t in range(len(reward)):
        result = result + gamma**t * survival * reward[t]
        continuation = survival * (1 - done[t])
        result = result + gamma ** (t + 1) * continuation * imagined["truncated"][t] * (
            critic(obs[t + 1]).squeeze(-1) if values is None else values[t + 1]
        )
        survival = continuation * ~imagined["truncated"][t]
    return result + gamma ** len(reward) * survival * (
        critic(obs[-1]).squeeze(-1) if values is None else values[-1]
    )


@torch.no_grad()
def critic_targets(t, imagined, values=None):
    if values is None:
        values = t.critic(imagined.get("critic_obs", imagined["obs"]).detach()).squeeze(
            -1
        )
    else:
        values = values.detach()
    ret = values[-1]
    targets = []
    for j in reversed(range(t.cfg.horizon)):
        future = torch.where(imagined["truncated"][j], values[j + 1], ret)
        ret = imagined["reward"][j].detach() + t.cfg.gamma * (
            1 - imagined["done"][j].detach()
        ) * ((1 - t.cfg.lam) * values[j + 1] + t.cfg.lam * future)
        targets.append(ret)
    return torch.stack(targets[::-1])


def _update_actor(t):
    from .trainer import freeze

    c = t.cfg
    if not t.actor_warmup_complete():
        raise RuntimeError("Actor/Critic update blocked during WM-only warmup")
    if c.actor_steps != 1:
        raise ValueError("Chunked joint update requires actor_steps=1")
    if c.gradient_audit_every:
        raise ValueError(
            "Chunked Actor uses aggregate gradient norm; configure gradient_audit_every=0 explicitly"
        )
    begin = timestamp(t.device)
    if str(t.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    n = c.actor_batch or c.batch
    starts = t.sample(starts=True, n=n)
    t.actor_opt.zero_grad(set_to_none=True)
    x, y = [], []
    loss_total = 0.0
    valid_steps = 0
    for offset in range(0, n, c.actor_microbatch or n):
        b = {
            k: v[offset : offset + (c.actor_microbatch or n)] for k, v in starts.items()
        }
        with freeze(t.model, t.termination, t.prior._disc, t.critic):
            imagined = t.imagine(start_batch=b)
            values = None
            if c.batch_imagination_critic:
                observation = imagined.get("critic_obs", imagined["obs"])
                values = t.critic(observation.flatten(0, 1)).reshape(
                    observation.shape[:2]
                )
            loss = -returns(imagined, t.critic, c.gamma, values).mean()
            if c.actor_horizon_scale:
                loss = loss / c.horizon
            scaled = loss * len(b["state"]) / n
            with torch.profiler.record_function("actor/BPTT"):
                scaled.backward()
            loss_total += float(scaled.detach())
            target = critic_targets(t, imagined, values)
            mask = imagined["valid"].reshape(-1)
            x.append(
                imagined.get("critic_obs", imagined["obs"])[:-1]
                .detach()
                .reshape(-1, t.critic_dim)[mask]
            )
            y.append(target.reshape(-1)[mask])
            valid_steps += int(mask.sum())
        del imagined, scaled, loss, target, values
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
    actor_seconds = timestamp(t.device) - begin
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
            losses.append(loss.detach().double())
    critic_end = timestamp(t.device)
    result = {
        "actor/loss": loss_total,
        "actor/gradient_norm": float(norm),
        "critic/loss": float(torch.stack(losses).mean()),
        "imagination/valid_fraction": valid_steps / (n * c.horizon),
        "imagination/start_batch": n,
        "time/actor_update_seconds": actor_seconds,
        "time/critic_update_seconds": critic_end - begin - actor_seconds,
    }

    if "launched" in starts:
        result["imagination/start_waiting_fraction"] = float(
            (~starts["launched"].bool()).float().mean()
        )
        result["imagination/start_launched_fraction"] = float(
            starts["launched"].float().mean()
        )
    speed = starts["state"][:, 78:81].norm(dim=-1)
    result["imagination/start_box_stationary_fraction"] = float(
        (speed <= 0.1).float().mean()
    )
    result["imagination/start_box_moving_fraction"] = float(
        (speed > 0.1).float().mean()
    )
    if "stage" in starts:
        result.update(
            {
                f"imagination/start_geometric_stage_{i}": int(v)
                for i, v in enumerate(torch.bincount(starts["stage"], minlength=4))
            }
        )
    if t.cfg.monitor_contacts and "start_contact_phase" in starts:
        result.update(
            {
                f"imagination/start_contact_phase_{i}": int(
                    (starts["start_contact_phase"] == i).sum()
                )
                for i in (-1, 0, 1, 2)
            }
        )
    phase_bin = (
        (starts["phase"].float() / starts["reference_length"] * 10).long().clamp(0, 9)
    )
    result.update(
        {
            f"imagination/start_reference_phase_bin_{i}": int(v)
            for i, v in enumerate(torch.bincount(phase_bin, minlength=10))
        }
    )
    if str(t.device).startswith("cuda"):
        result["gpu/actor_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        result["gpu/actor_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    return result


def update_actor(t):
    if not t.profile_next_actor:
        return _update_actor(t)
    from pathlib import Path

    activities = [torch.profiler.ProfilerActivity.CPU]
    if str(t.device).startswith("cuda"):
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    if t.profile_dir is None:
        raise ValueError("Profiling requires an output directory")
    t.profile_next_actor = False
    with torch.profiler.profile(
        activities=activities, record_shapes=True, profile_memory=True
    ) as profiler:
        result = _update_actor(t)
    path = Path(t.profile_dir)
    path.mkdir(parents=True, exist_ok=True)
    profiler.export_chrome_trace(str(path / f"actor_{t.actor_updates:06d}.json"))
    (path / f"actor_{t.actor_updates:06d}.txt").write_text(
        profiler.key_averages().table(sort_by="self_cpu_time_total", row_limit=40)
    )
    return result
