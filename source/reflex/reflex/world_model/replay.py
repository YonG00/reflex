"""Time-major ring of vectorized transitions, with explicit external-event edges."""

import torch


class Replay:
    def __init__(self, capacity_steps, device="cpu"):
        self.capacity, self.device = capacity_steps, device
        self.data, self.count, self.pointer = {}, 0, 0

    def add(self, transition):
        if not self.data:
            self.data = {
                k: torch.empty(
                    (self.capacity, *v.shape), dtype=v.dtype, device=self.device
                )
                for k, v in transition.items()
            }
        if transition.keys() != self.data.keys():
            raise ValueError("Replay transition schema changed")
        for k, v in transition.items():
            self.data[k][self.pointer].copy_(v.detach())
        self.pointer = (self.pointer + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def ordered(self, key):
        t = self.data[key]
        if self.count < self.capacity:
            return t[: self.count]
        return torch.cat((t[self.pointer :], t[: self.pointer]))

    def indices(self, horizon, starts=False):
        if self.count < horizon:
            return torch.empty((0, 2), dtype=torch.long, device=self.device)
        valid = self.ordered("valid").clone()
        for offset in range(horizon):
            shifted = torch.roll(self.ordered("valid"), -offset, 0)
            valid &= shifted
            if offset < horizon - 1:
                valid &= ~torch.roll(
                    self.ordered("boundary")
                    | self.ordered("terminated")
                    | self.ordered("timeout"),
                    -offset,
                    0,
                )
        if horizon > 1:
            valid[-horizon + 1 :] = False
        if starts:
            valid &= self.ordered("launched").bool()
            valid &= self.ordered("phase") + horizon < self.ordered("reference_length")
        return valid.nonzero()

    def sample(
        self, n, horizon=1, starts=False, validation=False, stage_balanced=False
    ):
        choices = self.indices(horizon, starts)
        # Entire environment IDs are held out. Never split adjacent frames from
        # the same physical episode between fit and validation.
        choices = choices[(choices[:, 1] % 5 == 0) == validation]
        if len(choices) == 0:
            raise ValueError("No valid replay windows: gather more post-launch data")
        if stage_balanced and not validation:
            labels = self.ordered("stage")[choices[:, 0], choices[:, 1]]
            pools = [choices[labels == stage] for stage in range(4)]
            weights = torch.tensor([0.35, 0.25, 0.2, 0.2], device=self.device)
            weights *= torch.tensor([len(p) > 0 for p in pools], device=self.device)
            stages = torch.multinomial(weights, n, replacement=True)
            selected = []
            for stage, pool in enumerate(pools):
                count = int((stages == stage).sum())
                if count:
                    selected.append(
                        pool[torch.randint(len(pool), (count,), device=self.device)]
                    )
            chosen = torch.cat(selected)
            chosen = chosen[torch.randperm(n, device=self.device)]
        else:
            chosen = choices[torch.randint(len(choices), (n,), device=self.device)]
        t, e = chosen.unbind(-1)
        offsets = torch.arange(horizon, device=self.device)[:, None]
        start = self.pointer if self.count == self.capacity else 0
        physical = (t[None, :] + offsets + start) % self.capacity
        return {k: value[physical, e[None, :]] for k, value in self.data.items()}

    def start_indices(self, validation=False):
        """Only the snapshot must be valid; no real future is inspected.

        Reference availability, timeouts and future event handling belong to
        imagined rollout masks, not to the recorded future trajectory filter.
        """
        if not self.count:
            return torch.empty((0, 2), dtype=torch.long, device=self.device)
        valid = (
            self.ordered("start_valid" if "start_valid" in self.data else "valid")
            & self.ordered("launched").bool()
        )
        if "phase" in self.data:
            valid &= (self.ordered("phase") >= 0) & (
                self.ordered("phase") < self.ordered("reference_length")
            )
        ix = valid.nonzero()
        return ix[(ix[:, 1] % 5 == 0) == validation]

    def sample_starts(self, n, validation=False):
        choices = self.start_indices(validation)
        if not len(choices):
            raise ValueError("No valid post-launch start snapshots")
        t, e = choices[torch.randint(len(choices), (n,), device=self.device)].unbind(-1)
        start = self.pointer if self.count == self.capacity else 0
        physical = (t + start) % self.capacity
        return {k: v[physical, e] for k, v in self.data.items()}

    def state_dict(self):
        return {
            "capacity": self.capacity,
            "count": self.count,
            "pointer": self.pointer,
            "data": {
                k: v[: self.count].cpu() if self.count < self.capacity else v.cpu()
                for k, v in self.data.items()
            },
        }

    def load_state_dict(self, state):
        if state["capacity"] != self.capacity:
            raise ValueError("Replay capacity mismatch")
        self.count, self.pointer = state["count"], state["pointer"]
        if (
            not 0 <= self.count <= self.capacity
            or not 0 <= self.pointer < self.capacity
        ):
            raise ValueError("Invalid replay indices")
        self.data = {}
        for k, v in state["data"].items():
            dst = torch.empty(
                (self.capacity, *v.shape[1:]), dtype=v.dtype, device=self.device
            )
            dst[: len(v)].copy_(v.to(self.device))
            self.data[k] = dst
