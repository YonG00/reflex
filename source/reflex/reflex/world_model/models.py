"""Joint residual dynamics and per-term learned reward / termination / memory heads."""

import torch
from torch import nn
from .physics import canonical, integrate


def mlp(inputs, outputs, hidden=256, layers=None, activation=nn.ELU):
    sizes = [inputs, *(layers or [hidden, hidden]), outputs]
    modules = []
    for i, (a, b) in enumerate(zip(sizes, sizes[1:])):
        modules.append(nn.Linear(a, b))
        if i < len(sizes) - 2:
            modules.append(activation())
    return nn.Sequential(*modules)


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
    """Legacy v1 multitask model, retained only for explicit old-data controls."""

    state_only = False

    def __init__(
        self,
        context_dim,
        memory_dim,
        rewards,
        dt=0.02,
        hidden=256,
        com_indices=None,
        hidden_layers=None,
        activation=nn.ELU,
    ):
        super().__init__()
        self.dt, self.memory_dim, self.rewards = dt, memory_dim, rewards
        self.com_indices = com_indices
        # Reward conditioning must include relative reference pose, previous raw
        # action (in observation), and current raw action: wrist locks make the
        # raw → processed target map noninvertible.
        n = 86 + 29 + context_dim + memory_dim + 2 + 179 + 29
        self.inputs = Standardizer(n)
        self.reward_norm = Standardizer(rewards)
        self.net = mlp(
            n,
            82 + memory_dim + rewards + 1,
            hidden,
            layers=hidden_layers,
            activation=activation,
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def features(self, s, target, context, memory, phase, blend, observation, raw):
        return torch.cat(
            (
                canonical(s),
                target,
                context,
                memory,
                phase[..., None].float() * self.dt,
                blend,
                observation,
                raw,
            ),
            -1,
        )

    def forward(self, s, target, context, memory, phase, blend, observation, raw):
        y = self.net(
            self.inputs(
                self.features(
                    s, target, context, memory, phase, blend, observation, raw
                )
            )
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


class StateWorldModel(nn.Module):
    """V2 physical dynamics only. No learned reward, history or termination head.

    The state-conditioned wrist reflex/PD/PhysX macro plant consumes processed
    targets. Raw action history and reference/reward history are task inputs,
    not inputs to this autonomous post-launch physical transition.
    """

    state_only = True

    def __init__(
        self,
        context_dim,
        dt=0.02,
        hidden_layers=(2048, 2048, 2048),
        com_indices=None,
        activation=nn.SiLU,
    ):
        super().__init__()
        self.dt, self.com_indices = dt, com_indices
        self.inputs = Standardizer(86 + 29 + context_dim)
        self.net = mlp(
            86 + 29 + context_dim, 82, layers=hidden_layers, activation=activation
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def features(self, state, target, context):
        return torch.cat((canonical(state), target, context), -1)

    def forward(self, state, target, context):
        delta = self.net(self.inputs(self.features(state, target, context)))
        if self.com_indices is None:
            return integrate(state, delta, self.dt)
        root, box = self.com_indices
        return integrate(
            state,
            delta,
            self.dt,
            context[..., root : root + 3],
            context[..., box : box + 3],
        )

    step = forward
