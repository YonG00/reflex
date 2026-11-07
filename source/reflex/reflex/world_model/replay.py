"""Time-major ring of vectorized transitions, with explicit external-event edges."""

import torch


class Replay:
    def __init__(
        self, capacity_steps, device="cpu", global_fields=(), cache_indices=True
    ):
        self.capacity, self.device = capacity_steps, device
        self.global_fields = set(global_fields)
        self.data, self.count, self.pointer = {}, 0, 0
        self.cache_indices = cache_indices
        self._index_cache = {}
        self._cache_signature = None

    def _cached(self, key, build):
        # Tensor versions also catch explicit boundary edits during resume/tests.
        fields = (
            "valid",
            "start_valid",
            "boundary",
            "terminated",
            "timeout",
            "phase",
            "reference_length",
        )
        signature = (
            self.count,
            self.pointer,
            tuple(
                (k, id(self.data[k]), self.data[k]._version)
                for k in fields
                if k in self.data
            ),
        )
        if signature != self._cache_signature:
            self._index_cache.clear()
            self._cache_signature = signature
        if not self.cache_indices:
            return build()
        if key not in self._index_cache:
            self._index_cache[key] = build()
        return self._index_cache[key]

    def memory_report(self, capacity_steps=None):
        """Exact schema storage, excluding allocator overhead and index caches."""
        capacity = self.capacity if capacity_steps is None else capacity_steps
        fields = {k: v[0].numel() * v.element_size() for k, v in self.data.items()}
        per_step = sum(fields.values())
        # Count unique additional storage. Ordered views before ring wrap alias
        # Replay payload and must not be reported as new GPU allocations.
        payload_storage = {v.untyped_storage().data_ptr() for v in self.data.values()}
        cache_storage = {
            v.untyped_storage().data_ptr(): v.untyped_storage().nbytes()
            for v in self._index_cache.values()
        }
        return {
            "capacity_steps": capacity,
            "bytes_per_vector_step": per_step,
            "payload_bytes": per_step * capacity,
            "field_bytes_per_vector_step": fields,
            "index_cache_bytes": sum(
                n for ptr, n in cache_storage.items() if ptr not in payload_storage
            ),
        }

    def _choices(self, validation, variable=False):
        def build():
            choices = self.indices(1)
            choices = choices[(choices[:, 1] % 5 == 0) == validation]
            if variable and "phase" in self.data:
                phase = self.ordered("phase")[choices[:, 0], choices[:, 1]]
                length = self.ordered("reference_length")[choices[:, 0], choices[:, 1]]
                choices = choices[(phase >= 0) & (phase < length)]
            return choices

        return self._cached(("choices", validation, variable), build)

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
        return self._cached(
            ("windows", horizon, starts), lambda: self._indices(horizon, starts)
        )

    def _indices(self, horizon, starts=False):
        if self.count < horizon:
            return torch.empty((0, 2), dtype=torch.long, device=self.device)
        ordered_valid = self._cached(("ordered_valid",), lambda: self.ordered("valid"))
        edge = (
            self._cached(
                ("ordered_edge",),
                lambda: (
                    self.ordered("boundary")
                    | self.ordered("terminated")
                    | self.ordered("timeout")
                ),
            )
            if horizon > 1
            else None
        )
        valid = ordered_valid.clone()
        for offset in range(horizon):
            shifted = torch.roll(ordered_valid, -offset, 0)
            valid &= shifted
            if offset < horizon - 1:
                valid &= ~torch.roll(
                    edge,
                    -offset,
                    0,
                )
        if horizon > 1:
            valid[-horizon + 1 :] = False
        if starts:
            valid &= self.ordered("phase") >= 0
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
            raise ValueError("No valid replay windows: gather more valid data")
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
        return {
            k: (
                value[physical]
                if k in self.global_fields
                else value[physical, e[None, :]]
            )
            for k, value in self.data.items()
        }

    def sample_variable_windows(self, n, horizon, validation=False):
        """Up to H transitions, stopping only at recorded supervision boundaries.

        Normal V24 external writes remain inside windows; padding carries no loss.
        """
        choices = self._choices(validation, variable=True)
        if not len(choices):
            raise ValueError("No valid variable-length windows")
        t0, e = choices[torch.randint(len(choices), (n,), device=self.device)].unbind(
            -1
        )
        times = []
        masks = []
        alive = torch.ones(n, dtype=torch.bool, device=self.device)
        valid = self._cached(("ordered_valid",), lambda: self.ordered("valid"))
        edge = self._cached(
            ("ordered_edge",),
            lambda: (
                self.ordered("boundary")
                | self.ordered("terminated")
                | self.ordered("timeout")
            ),
        )
        for j in range(horizon):
            raw = t0 + j
            ix = raw.clamp_max(self.count - 1)
            alive = alive & (raw < self.count) & valid[ix, e]
            times.append(torch.where(alive, ix, t0))
            masks.append(alive)
            alive = alive & ~edge[ix, e]
        times = torch.stack(times)
        start = self.pointer if self.count == self.capacity else 0
        physical = (times + start) % self.capacity
        result = {
            k: (
                v[physical]
                if k in self.global_fields
                else v[physical, e[None, :].expand_as(times)]
            )
            for k, v in self.data.items()
        }
        result["loss_mask"] = torch.stack(masks)
        return result

    def start_indices(self, validation=False):
        return self._cached(
            ("starts", validation), lambda: self._start_indices(validation)
        )

    def _start_indices(self, validation=False):
        """Only the snapshot must be valid; no real future is inspected.

        Reference availability, timeouts and future event handling belong to
        imagined rollout masks, not to the recorded future trajectory filter.
        """
        if not self.count:
            return torch.empty((0, 2), dtype=torch.long, device=self.device)
        valid = self.ordered(
            "start_valid" if "start_valid" in self.data else "valid"
        ).clone()
        if "phase" in self.data:
            valid &= (self.ordered("phase") >= 0) & (
                self.ordered("phase") < self.ordered("reference_length")
            )
        ix = valid.nonzero()
        return ix[(ix[:, 1] % 5 == 0) == validation]

    def sample_starts(self, n, validation=False):
        choices = self.start_indices(validation)
        if not len(choices):
            raise ValueError("No valid imagination start snapshots")
        t, e = choices[torch.randint(len(choices), (n,), device=self.device)].unbind(-1)
        start = self.pointer if self.count == self.capacity else 0
        physical = (t + start) % self.capacity
        return {
            k: (v[physical] if k in self.global_fields else v[physical, e])
            for k, v in self.data.items()
        }

    def state_dict(self):
        return {
            "global_fields": sorted(self.global_fields),
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
        if set(state.get("global_fields", [])) != self.global_fields:
            raise ValueError("Replay global metadata schema mismatch")
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
