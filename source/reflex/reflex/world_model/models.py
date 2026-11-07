"""Pure physical residual dynamics and explicitly retained legacy multitask model."""

import torch
from torch import nn

from .physics import canonical, integrate


class PrecisionMLP(nn.Sequential):
    """Optional MLP-only autocast; public state and parameters remain FP32."""

    precision = "fp32"

    def forward(self, x):
        if self.precision == "bf16":
            with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16):
                y = super().forward(x)
            return y.float()
        return super().forward(x)


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


class TerminationHead(nn.Module):
    """Independent P(physical terminal | predicted state, reference geometry)."""

    def __init__(self, feature_dim, hidden=(64, 64)):
        super().__init__()
        self.inputs = Standardizer(feature_dim)
        self.net = mlp(2 * feature_dim, 1, layers=hidden, activation=nn.SiLU)

    def forward(self, features):
        x = self.inputs(features)
        return self.net(torch.cat((x, x.abs()), -1)).squeeze(-1)


class MacroWorldModel(StateWorldModel):
    """One learned macro step, including external control writes.

    Shared trunk predicts pre-control physics, post-control state and next
    launched mode. No simulator control/event callback runs in imagination.
    """

    def __init__(self, context_dim, **kwargs):
        super().__init__(context_dim, **kwargs)
        first, last = self.net[0], self.net[-1]
        self.inputs = Standardizer(first.in_features + 1)
        self.net[0] = nn.Linear(first.in_features + 1, first.out_features)
        self.net[-1] = nn.Linear(last.in_features, 165)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def features(self, state, target, context, mode=None):
        if mode is None:
            mode = torch.zeros_like(state[..., 0])
        return torch.cat(
            (canonical(state), target, context, mode.float()[..., None]), -1
        )

    def predict_all(self, state, target, context, mode=None):
        values = self.net(self.inputs(self.features(state, target, context, mode)))
        if self.com_indices is None:
            physical = integrate(state, values[..., :82], self.dt)
        else:
            root, box = self.com_indices
            physical = integrate(
                state,
                values[..., :82],
                self.dt,
                context[..., root : root + 3],
                context[..., box : box + 3],
            )
        # dt=0 applies learned instantaneous corrections, without another
        # analytic physics step or any hand-coded launch/hover rule.
        post_control = integrate(physical, values[..., 82:164], 0.0)
        return physical, post_control, values[..., 164]

    def forward(self, state, target, context, mode=None):
        return self.predict_all(state, target, context, mode)[1]

    step = forward

    def warm_start_physical(self, old):
        current = self.state_dict()
        last = str(len(self.net) - 1)
        for key, value in old.items():
            if key == "net.0.weight":
                current[key].zero_()
                current[key][:, :-1].copy_(value)
            elif key in ("inputs.mean", "inputs.std"):
                current[key][:-1].copy_(value)
            elif key == "net." + last + ".weight":
                current[key].zero_()
                current[key][:82].copy_(value)
            elif key == "net." + last + ".bias":
                current[key].zero_()
                current[key][:82].copy_(value)
            else:
                current[key].copy_(value)
        self.load_state_dict(current)


class ConditionedWorldModel(StateWorldModel):
    """V24 causal split: predict physics, then condition on issued events.

    Post-control commands cannot reach the physical encoder. The decoder learns
    their effects; neither scheduler nor write plan substitutes its state output.
    """

    pre_dim = 4
    condition_dim = 166

    def __init__(self, context_dim, **kwargs):
        super().__init__(context_dim, **kwargs)
        old = self.net[0]
        self.net[0] = nn.Linear(old.in_features + self.pre_dim, old.out_features)
        self.inputs = Standardizer(old.in_features + self.pre_dim)
        self.post = mlp(
            86 + self.condition_dim, 82, layers=[512, 512], activation=nn.SiLU
        )
        nn.init.zeros_(self.post[-1].weight)
        nn.init.zeros_(self.post[-1].bias)

    def features(self, state, target, context, event):
        return torch.cat((canonical(state), target, context, event), -1)

    def predict_physical(self, state, target, context, event):
        delta = self.net(self.inputs(self.features(state, target, context, event)))
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

    def predict_post(self, physical, targets, mask):
        from .events_v24.writes import event_condition

        condition = event_condition(physical, targets, mask)
        delta = self.post(torch.cat((canonical(physical), condition), -1))
        return integrate(physical, delta, 0.0)

    def forward(self, state, target, context, event):
        return self.predict_physical(state, target, context, event)

    def warm_start_physical(self, old):
        current = self.state_dict()
        last = str(len(self.net) - 1)
        base = self.net[0].in_features - self.pre_dim
        old_mode = old["net.0.weight"].shape[1] == base + 1
        for key, value in old.items():
            if key not in current or key.startswith("post."):
                continue
            if key == "net.0.weight":
                current[key].zero_()
                if value.shape == current[key].shape:
                    current[key].copy_(value)
                else:
                    current[key][:, :base].copy_(value[:, :base])
                if old_mode:
                    current[key][:, base].copy_(value[:, base])
            elif key in ("inputs.mean", "inputs.std"):
                if value.shape == current[key].shape:
                    current[key].copy_(value)
                else:
                    current[key][:base].copy_(value[:base])
                if old_mode:
                    current[key][base].copy_(value[base])
            elif key in ("net." + last + ".weight", "net." + last + ".bias"):
                current[key].copy_(value[:82])
            else:
                current[key].copy_(value)
        self.load_state_dict(current)


class DeterministicEventWorldModel(ConditionedWorldModel):
    """V24 physical network with exact native command application."""

    def __init__(self, context_dim, **kwargs):
        super().__init__(context_dim, **kwargs)
        del self.post

    def predict_post(self, physical, targets, mask):
        # Native plans already use local positions and world-frame velocities.
        # Do not integrate/normalize untouched quaternion or velocity components.
        return torch.where(mask, targets, physical)
