"""Snapshot benchmark mutations; retain real replay sampling and fixed RNG."""

import copy
import hashlib
import random

import numpy as np
import torch


def digest(tensors):
    h = hashlib.sha256()
    for key, value in sorted(tensors.items()):
        h.update(f"{key}:{value.dtype}:{tuple(value.shape)}".encode())
        h.update(
            value.detach()
            .cpu()
            .contiguous()
            .reshape(-1)
            .view(torch.uint8)
            .numpy()
            .tobytes()
        )
    return h.hexdigest()


class Snapshot:
    names = ("actor", "critic", "actor_opt", "critic_opt", "obs_norm", "critic_norm")

    def __init__(self, trainer):
        self.t = trainer
        self.weights = {
            k: copy.deepcopy(getattr(trainer, k).state_dict()) for k in self.names
        }
        self.gradients = {
            k: [
                None if p.grad is None else p.grad.clone()
                for p in getattr(trainer, k).parameters()
            ]
            for k in ("actor", "critic")
        }
        self.rng = (
            random.getstate(),
            np.random.get_state(),
            torch.get_rng_state(),
            torch.cuda.get_rng_state_all()
            if str(trainer.device).startswith("cuda")
            else None,
        )
        self.scheduler_rng = trainer.event_scheduler.rng.clone()
        self.attrs = {
            k: copy.deepcopy(getattr(trainer, k))
            for k in (
                "actor_updates",
                "profile_next_actor",
                "replay_seconds",
                "_transition_cache_ready",
            )
            if hasattr(trainer, k)
        }
        self.static = copy.deepcopy(trainer.event_scheduler.static)
        self.task_cache = copy.deepcopy(trainer.task._static_cache)
        self.index_cache = dict(trainer.replay._index_cache)

    def reset(self):
        for name, value in self.weights.items():
            getattr(self.t, name).load_state_dict(copy.deepcopy(value))
        for name, gradients in self.gradients.items():
            for p, g in zip(getattr(self.t, name).parameters(), gradients):
                p.grad = None if g is None else g.clone()
        self.t.actor_updates = self.attrs["actor_updates"]
        self.t.profile_next_actor = False
        random.setstate(self.rng[0])
        np.random.set_state(self.rng[1])
        torch.set_rng_state(self.rng[2])
        if self.rng[3] is not None:
            torch.cuda.set_rng_state_all(self.rng[3])
        self.t.event_scheduler.rng = self.scheduler_rng.clone()

    def restore(self):
        self.reset()
        for name, value in self.attrs.items():
            setattr(self.t, name, value)
        self.t.event_scheduler.static = self.static
        self.t.task._static_cache = self.task_cache
        self.t.replay._index_cache.clear()
        self.t.replay._index_cache.update(self.index_cache)


def parameter_result(trainer):
    result = {}
    for name in ("actor", "critic"):
        result.update(
            {
                f"{name}/{k}": v.detach().cpu().clone()
                for k, v in getattr(trainer, name).state_dict().items()
            }
        )
        result.update(
            {
                f"{name}/grad/{i}": p.grad.detach().cpu().clone()
                for i, p in enumerate(getattr(trainer, name).parameters())
                if p.grad is not None
            }
        )
    return result
