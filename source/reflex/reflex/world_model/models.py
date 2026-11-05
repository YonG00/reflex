"""Joint residual dynamics and per-term learned reward / termination / memory heads."""

import torch
from torch import nn
from .physics import canonical, integrate


def mlp(inputs, outputs, hidden=256):
    return nn.Sequential(
        nn.Linear(inputs, hidden),
        nn.ELU(),
        nn.Linear(hidden, hidden),
        nn.ELU(),
        nn.Linear(hidden, outputs),
    )


class Standardizer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))

    @torch.no_grad()
    def fit(self, x):
        self.mean.copy_(x.mean(0))
        self.std.copy_(x.std(0, unbiased=False).clamp_min(0.01))

    def forward(self, x):
        return ((x - self.mean) / self.std).clamp(-20, 20)


class WorldModel(nn.Module):
    def __init__(
        self, context_dim, memory_dim, rewards, dt=0.02, hidden=256, com_indices=None
    ):
        super().__init__()
        self.dt, self.memory_dim, self.rewards = dt, memory_dim, rewards
        self.com_indices = com_indices
        # Canonical physical features 86 + target29 + ξ + reward memory + phase/blend2.
        n = 86 + 29 + context_dim + memory_dim + 2
        self.inputs = Standardizer(n)
        self.reward_norm = Standardizer(rewards)
        self.net = mlp(n, 82 + memory_dim + rewards + 1, hidden)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def features(self, s, target, context, memory, phase, blend):
        return torch.cat(
            (
                canonical(s),
                target,
                context,
                memory,
                phase[..., None].float() * self.dt,
                blend,
            ),
            -1,
        )

    def forward(self, s, target, context, memory, phase, blend):
        y = self.net(
            self.inputs(self.features(s, target, context, memory, phase, blend))
        )
        delta, dm, r, done = y.split((82, self.memory_dim, self.rewards, 1), -1)
        if self.com_indices is None:
            ns = integrate(s, delta, self.dt)
        else:
            root, box = self.com_indices
            ns = integrate(
                s,
                delta,
                self.dt,
                context[..., root : root + 3],
                context[..., box : box + 3],
            )
        reward = r * self.reward_norm.std + self.reward_norm.mean
        return ns, memory + dm, reward, done.squeeze(-1)
