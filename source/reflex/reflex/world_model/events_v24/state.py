"""Controller state codec. Stores only information available at the current step."""

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class Field:
    name: str
    shape: tuple
    dtype: torch.dtype

    @property
    def width(self):
        import math

        return math.prod(self.shape) if self.shape else 1


class EventStateCodec:
    base = {
        "_box_p0": (3,),
        "_box_q0": (4,),
        "_box_v0": (3,),
        "_box_omega": (3,),
        "_box_launch_step": (),
        "_box_launched": (),
        "_box_aim_err": (),
        "_box_catch_h": (),
        "_box_tmult": (),
        "_box_jitter": (),
        "_box_lib_mask": (),
        "_balloon_radius_per_env": (),
        "_box_half_extents": (3,),
        "_v36_walk_v": (2,),
        "_v36_walk_anchor": (2,),
        "_v38_aim_off": (2,),
        "_box_regrace": (),
        "_v38_cradle_ticks": (),
        "_hold_init_pending": (),
        "_hold_init_active": (),
    }
    booleans = {
        "_box_launched",
        "_box_lib_mask",
        "_hold_init_pending",
        "_hold_init_active",
    }
    integers = {
        "_box_launch_step",
        "_box_regrace",
        "_v38_cradle_ticks",
        "_box_model_idx",
    }

    def __init__(self, env, terms):
        shapes = dict(self.base)
        for name in ("_box_model_idx", "_box_lib_arr_dy", "_box_lib_tgt_x"):
            if hasattr(env, name):
                shapes[name] = ()
        if hasattr(env, "_box_lib_arrvec_w"):
            shapes["_box_lib_arrvec_w"] = (2,)
        self.fields = [
            Field(
                k,
                shape,
                torch.bool
                if k in self.booleans
                else torch.long
                if k in self.integers
                else torch.float32,
            )
            for k, shape in shapes.items()
        ]
        self.slices = {}
        offset = 0
        for field in self.fields:
            self.slices[field.name] = slice(offset, offset + field.width)
            offset += field.width
        self.control_width = offset
        self.timer_count = len(terms)
        self.width = offset + len(terms)

    def specification(self):
        return {
            "version": 1,
            "fields": [(f.name, list(f.shape), str(f.dtype)) for f in self.fields],
            "timer_count": self.timer_count,
        }

    def capture(self, env, timers):
        n, device = env.num_envs, env.device
        values = []
        for field in self.fields:
            v = getattr(env, field.name, None)
            if v is None:
                if field.name == "_v36_walk_anchor":
                    v = env._box_p0[:, :2]
                else:
                    v = torch.zeros((n, *field.shape), device=device, dtype=field.dtype)
            values.append(v.reshape(n, field.width).float())
        values.extend(t.expand(n).reshape(n, 1) for t in timers)
        return torch.cat(values, -1).clone()

    def decode(self, packed):
        n = len(packed)
        values = {
            f.name: packed[:, self.slices[f.name]]
            .reshape(n, *f.shape)
            .to(f.dtype)
            .clone()
            for f in self.fields
        }
        return values, packed[:, self.control_width :].clone()

    def encode(self, values, timers):
        return torch.cat(
            [
                *(
                    values[f.name].reshape(len(timers), f.width).float()
                    for f in self.fields
                ),
                timers,
            ],
            -1,
        )

    def mode(self, state):
        return state[:, self.slices["_box_launched"]].squeeze(-1).bool()

    def pre_features(self, state, episode_steps, dt):
        mode = self.mode(state).float()
        remaining = (
            state[:, self.slices["_box_launch_step"]].squeeze(-1) - episode_steps
        ) * dt
        pending = state[:, self.slices["_hold_init_pending"]].squeeze(-1)
        ticks = state[:, self.slices["_v38_cradle_ticks"]].squeeze(-1) / 75
        return torch.stack((mode, remaining, pending, ticks), -1)
