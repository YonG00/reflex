"""Independent stage-1 WM projection and lossless physical-context Replay codec.

Projection is neural conditioning only. Analytical consumers recover the exact
observed full context, including mass-dependent inertias and all COM frames.
"""

from dataclasses import dataclass

import torch

from .replay import Replay


@dataclass(frozen=True)
class ContextSpec:
    schema: tuple
    body_names: tuple
    torso: str = "torso_link"
    profile: str = "stage1_dr_v1"

    def __post_init__(self):
        if self.profile != "stage1_dr_v1" or self.torso not in self.body_names:
            raise ValueError("Unsupported compact context profile/body layout")
        widths = dict(self.schema)
        n = len(self.body_names)
        required = {
            "box.dimensions": 3,
            "box.masses": 1,
            "box.inertias": 9,
            "box.coms": 7,
            "box.material_properties": 3,
            "robot.masses": n,
            "robot.inertias": 9 * n,
            "robot.coms": 7 * n,
            **{
                "robot." + p: 29
                for p in (
                    "joint_stiffness",
                    "joint_damping",
                    "joint_armature",
                    "joint_friction_coeff",
                    "joint_effort_limits",
                )
            },
        }
        if set(widths) != {*required, "robot.material_properties"} or len(
            widths
        ) != len(self.schema):
            raise ValueError("Unknown/duplicate physical context fields")
        if any(widths[k] != v for k, v in required.items()):
            raise ValueError("Physical context dimensions differ from G1 layout")
        if (
            widths["robot.material_properties"] <= 0
            or widths["robot.material_properties"] % 3
        ):
            raise ValueError(
                "Contact material layout must retain every native shape triplet"
            )

    @property
    def slices(self):
        result, offset = {}, 0
        for name, width in self.schema:
            result[name] = (offset, offset + width)
            offset += width
        return result

    @property
    def full_dim(self):
        return sum(w for _, w in self.schema)

    @property
    def groups(self):
        groups = [
            (k, list(range(*self.slices[k])))
            for k in (
                "box.dimensions",
                "box.masses",
                "box.material_properties",
                "robot.masses",
            )
        ]
        start = self.slices["robot.coms"][0] + 7 * self.body_names.index(self.torso)
        groups.append(("robot.torso_com_xyz", list(range(start, start + 3))))
        groups.append(
            (
                "robot.material_properties",
                list(range(*self.slices["robot.material_properties"])),
            )
        )
        return groups

    @property
    def columns(self):
        return [i for _, cols in self.groups for i in cols]

    @property
    def omitted(self):
        chosen = set(self.columns)
        return [i for i in range(self.full_dim) if i not in chosen]

    @property
    def wm_dim(self):
        return len(self.columns)

    def select(self, context):
        if context.shape[-1] != self.full_dim:
            raise ValueError("Expected full analytical physical context")
        return context[..., self.columns]

    def specification(self):
        return {
            "version": 1,
            "profile": self.profile,
            "schema": [list(x) for x in self.schema],
            "body_names": list(self.body_names),
            "torso": self.torso,
            "full_dim": self.full_dim,
            "wm_dim": self.wm_dim,
            "wm_groups": [[name, len(cols)] for name, cols in self.groups],
            "wm_columns": self.columns,
            "omitted_columns": self.omitted,
        }

    @classmethod
    def from_contract(cls, contract):
        return cls(
            tuple((k, int(w)) for k, w in contract["context_schema"]),
            tuple(contract["body_names"]),
        )


class CompactContextReplay(Replay):
    """Ring stores wm_context + physics IDs; a version bank stores other columns.

    Every omitted-field change creates an immutable version, never an inferred
    value. Old samples remain correct across reset/mass/COM/controller changes.
    """

    supports_physical_codec = True

    def __init__(
        self, capacity_steps, spec, device="cpu", global_fields=(), cache_indices=True
    ):
        super().__init__(capacity_steps, device, global_fields, cache_indices)
        self.spec = spec
        self.bank = None
        self.used = 0
        self.last = self.last_ids = None
        self.gc_threshold = 0

    def _append(self, values):
        n = len(values)
        needed = self.used + n
        if self.bank is None or needed > len(self.bank):
            capacity = max(
                needed, 2 * len(self.bank) if self.bank is not None else 2 * n
            )
            bank = values.new_empty(capacity, len(self.spec.omitted))
            if self.bank is not None:
                bank[: self.used].copy_(self.bank[: self.used])
            self.bank = bank
        ids = torch.arange(self.used, needed, device=values.device)
        self.bank[self.used : needed].copy_(values)
        self.used = needed
        return ids

    def add(self, transition):
        if "context" not in transition or "physics_id" in transition:
            raise ValueError(
                "Compact Replay accepts explicit complete physical transitions"
            )
        full = transition["context"].detach()
        if (
            full.ndim != 2
            or full.shape[-1] != self.spec.full_dim
            or full.dtype != torch.float32
        ):
            raise ValueError("Compact physical context must be [env,full_dim] FP32")
        other = full[:, self.spec.omitted].contiguous()
        if self.last is None:
            self.last_ids = self._append(other)
            self.last = other.clone()
            self.gc_threshold = max(16, 4 * len(full))
        else:
            if other.shape != self.last.shape:
                raise ValueError("Environment count changed inside Replay")
            # Bit-level comparison preserves signed zero, NaN payloads and inf.
            changed = (
                (other.view(torch.int32) != self.last.view(torch.int32))
                .any(-1)
                .nonzero()
                .flatten()
            )
            if len(changed):
                self.last_ids = self.last_ids.index_copy(
                    0, changed, self._append(other[changed])
                )
                self.last[changed] = other[changed]
        encoded = {
            k: v for k, v in transition.items() if k not in ("context", "wm_context")
        }
        encoded.update(wm_context=self.spec.select(full), physics_id=self.last_ids)
        super().add(encoded)
        if self.used > self.gc_threshold:
            self._compact_bank()

    def decode(self, batch):
        ids, wm = batch["physics_id"], batch["wm_context"]
        if wm.shape[-1] != self.spec.wm_dim:
            raise ValueError("Compact WM dimension mismatch")
        full = wm.new_empty(*ids.shape, self.spec.full_dim)
        full[..., self.spec.columns] = wm
        full[..., self.spec.omitted] = self.bank[ids]
        return {**batch, "context": full}

    def sample(self, *args, **kwargs):
        return self.decode(super().sample(*args, **kwargs))

    def sample_variable_windows(self, *args, **kwargs):
        return self.decode(super().sample_variable_windows(*args, **kwargs))

    def sample_starts(self, *args, **kwargs):
        return self.decode(super().sample_starts(*args, **kwargs))

    def ordered(self, key):
        if key == "context":
            return self.decode(
                {
                    "physics_id": super().ordered("physics_id"),
                    "wm_context": super().ordered("wm_context"),
                }
            )["context"]
        return super().ordered(key)

    def _compact_bank(self):
        if not self.data:
            return
        keep = torch.unique(
            torch.cat((self.data["physics_id"][: self.count].flatten(), self.last_ids))
        )
        mapping = torch.full(
            (self.used,), -1, device=self.bank.device, dtype=torch.long
        )
        mapping[keep] = torch.arange(len(keep), device=keep.device)
        self.bank = self.bank[keep].clone()
        self.used = len(keep)
        self.gc_threshold = max(16, 4 * len(self.last_ids), 2 * self.used)
        self.data["physics_id"][: self.count] = mapping[
            self.data["physics_id"][: self.count]
        ]
        self.last_ids = mapping[self.last_ids]

    def memory_report(self, capacity_steps=None):
        report = super().memory_report(capacity_steps)
        report.update(
            physical_bank_used_bytes=0
            if self.bank is None
            else self.used * self.bank.shape[1] * 4,
            physical_bank_allocated_bytes=0
            if self.bank is None
            else self.bank.numel() * 4,
            physical_bank_versions=self.used,
            last_physical_bytes=0
            if self.last is None
            else self.last.numel() * 4 + self.last_ids.numel() * 8,
        )
        return report

    def state_dict(self):
        state = super().state_dict()
        state["physical_context_codec"] = {
            "spec": self.spec.specification(),
            "used": self.used,
            "bank": None if self.bank is None else self.bank[: self.used].cpu(),
            "last": None if self.last is None else self.last.cpu(),
            "last_ids": None if self.last_ids is None else self.last_ids.cpu(),
        }
        return state

    def load_state_dict(self, state):
        codec = state.get("physical_context_codec")
        if codec is None or codec["spec"] != self.spec.specification():
            raise ValueError(
                "Compact Replay schema mismatch; no implicit old Replay migration"
            )
        super().load_state_dict(state)
        self.used = codec["used"]
        self.gc_threshold = max(16, 2 * self.used)
        self.bank = (
            None if codec["bank"] is None else codec["bank"].to(self.device).clone()
        )
        self.last = (
            None if codec["last"] is None else codec["last"].to(self.device).clone()
        )
        self.last_ids = (
            None
            if codec["last_ids"] is None
            else codec["last_ids"].to(self.device).clone()
        )
        if self.data and (
            (self.data["physics_id"][: self.count] < 0).any()
            or (self.data["physics_id"][: self.count] >= self.used).any()
        ):
            raise ValueError("Invalid physical-context version reference")


def audit_randomization(env):
    """Fail closed on independent DR not covered by the stage-1 projection."""
    from .task import primitive

    allowed = {
        "randomize_rigid_body_material",
        "randomize_rigid_body_mass",
        "randomize_rigid_body_com",
        "randomize_joint_default_pos",
    }
    rows = []
    for mode, names in env.event_manager.active_terms.items():
        for name in names:
            term = env.event_manager.get_term_cfg(name)
            function = getattr(term.func, "__name__", type(term.func).__name__)
            params = term.params
            dr = (
                function.startswith("randomize_")
                or "mass_dr" in function
                or "randomize" in name
            )
            if dr and function not in allowed:
                raise ValueError(f"Uncovered domain randomization: {name}/{function}")
            asset = params.get("asset_cfg")
            asset_name = getattr(asset, "name", None)
            if dr and asset_name not in ("robot", "balloon"):
                raise ValueError(f"Uncovered randomized asset: {asset_name}")
            if function == "randomize_rigid_body_com":
                bodies = getattr(asset, "body_names", None)
                if bodies not in ("torso_link", ["torso_link"], ("torso_link",)):
                    raise ValueError(
                        "Compact WM only includes independently randomized torso COM"
                    )
            rows.append(
                {
                    "name": name,
                    "mode": mode,
                    "function": function,
                    "parameters": primitive(params),
                    "conditioning": "wm_context"
                    if dr and function != "randomize_joint_default_pos"
                    else "state_target_or_native_event",
                }
            )
    return {
        "profile": "stage1_dr_v1",
        "events": rows,
        "inertia": "mass randomization can recompute inertia; observed full tensors remain lossless in bank",
        "controller": "effective target already includes randomized default pose; gains not independently randomized",
    }


class FixedContextGuard:
    """Catch runtime drift in omitted non-inertia fields, rather than hide it."""

    def __init__(self, spec):
        inertia = set(range(*spec.slices["box.inertias"])) | set(
            range(*spec.slices["robot.inertias"])
        )
        self.columns = [i for i in spec.omitted if i not in inertia]
        self.previous = None

    def check(self, full):
        current = full[:, self.columns].contiguous()
        if self.previous is not None and not torch.equal(
            current.view(torch.int32), self.previous.view(torch.int32)
        ):
            raise ValueError(
                "Omitted fixed physical parameters changed; extend audited WM context profile"
            )
        self.previous = current.detach().clone()
