"""Real experience accounting. Unknown contact evidence is never called zero contact."""

import torch


def pair_contact(sensor, n, device):
    unknown = torch.zeros(n, dtype=torch.bool, device=device)
    if sensor is None:
        return unknown, unknown
    cfg = sensor.cfg
    if (
        not cfg.prim_path.endswith("/Balloon")
        or not cfg.filter_prim_paths_expr
        or not all("/Robot/" in p for p in cfg.filter_prim_paths_expr)
    ):
        raise ValueError(
            "Contact statistics require a box-only sensor filtered to robot bodies"
        )
    matrix = sensor.data.force_matrix_w
    if (
        matrix is None
        or matrix.ndim != 4
        or matrix.shape[0] != n
        or matrix.shape[1] != 1
        or matrix.shape[2] < 1
        or matrix.shape[-1] != 3
    ):
        raise ValueError(
            "Unexpected box/robot filtered force matrix; cannot report contacts"
        )
    history = getattr(sensor.data, "force_matrix_w_history", None)
    if getattr(cfg, "history_length", 0) >= 4:
        if history is None or history.ndim != 5 or history.shape[1] < 4:
            raise ValueError("Missing physics-substep contact history")
        values = history[:, :4]
    else:
        values = matrix
    if not bool(torch.isfinite(values).all()):
        raise ValueError("Nonfinite pair contact forces")
    return (values.norm(dim=-1) > 1.0).flatten(1).any(-1), torch.ones_like(unknown)


def clean2_intercept_candidate(box_pos, box_velocity, torso_pos, radius):
    # Exact CLEAN2 per_throw_eval criterion; persistence >=5 handled per throw.
    return (
        ((box_pos - torso_pos).norm(dim=-1) < radius.clamp_min(0.05) + 0.45)
        & (box_pos[..., 2] > 0.30)
        & (box_velocity.norm(dim=-1) < 1.0)
    )


class ExperienceMetrics:
    def __init__(self):
        self.reward = self.length = self.success = self.contact_prev = None
        self.completed = self.return_sum = self.length_sum = self.launches = (
            self.contact_edges
        ) = self.contact_steps = self.successes = 0
        self.contact_known_steps = self.steps = 0
        self.reasons = None
        self.loose_throws = self.loose_measured_throws = 0
        self.interception_throws = 0
        self.intercept_persist = self.throw_intercepted = None
        self.throw_loose = self.throw_loose_known = None
        self.active_throw = self.throw_contact = self.throw_success = (
            self.throw_measured
        ) = None
        self.completed_throws = self.measured_throws = self.intercepted_throws = (
            self.held_throws
        ) = 0

    def update(self, tr):
        r = tr["reward_terms"].sum(-1)
        if self.reward is None:
            self.reward = torch.zeros_like(r)
            self.length = torch.zeros_like(r)
            self.success = torch.zeros_like(r, dtype=torch.bool)
            self.contact_prev = torch.zeros_like(self.success)
        if self.active_throw is None:
            self.active_throw = torch.zeros_like(self.success)
            self.throw_contact = torch.zeros_like(self.success)
            self.throw_success = torch.zeros_like(self.success)
            self.throw_measured = torch.ones_like(self.success)
            self.throw_loose = torch.zeros_like(self.success)
            self.throw_loose_known = torch.ones_like(self.success)
            self.intercept_persist = torch.zeros_like(self.length)
            self.throw_intercepted = torch.zeros_like(self.success)
        self.reward += r
        self.length += 1
        success = tr.get("strict_success", torch.zeros_like(self.success))
        self.success |= success
        done = tr["terminated"] | tr["timeout"]
        self.completed += int(done.sum())
        self.return_sum += float(self.reward[done].sum())
        self.length_sum += float(self.length[done].sum())
        self.successes += int((self.success & done).sum())
        known = tr.get("contact_known", torch.zeros_like(done))
        contact = tr.get("box_contact", torch.zeros_like(done)) & known
        self.contact_known_steps += int(known.sum())
        self.steps += len(r)
        self.contact_steps += int(contact.sum())
        self.contact_edges += int((contact & ~self.contact_prev).sum())
        launch = tr.get("launch_event", torch.zeros_like(done))
        self.launches += int(launch.sum())
        self.throw_contact |= self.active_throw & contact
        self.throw_success |= self.active_throw & success
        candidate = tr.get("intercept_candidate", torch.zeros_like(done))
        self.intercept_persist = torch.where(
            self.active_throw & candidate, self.intercept_persist + 1, 0
        )
        self.throw_intercepted |= self.active_throw & (self.intercept_persist >= 5)
        self.throw_loose |= self.active_throw & tr.get(
            "loose_catch", torch.zeros_like(done)
        )
        self.throw_loose_known &= ~self.active_throw | tr.get(
            "loose_catch_known", torch.zeros_like(done)
        )
        self.throw_measured &= ~self.active_throw | known
        end = self.active_throw & (
            done | launch | ~tr.get("after_launched", torch.ones_like(done))
        )
        self.completed_throws += int(end.sum())
        self.interception_throws += int((end & self.throw_intercepted).sum())
        self.loose_measured_throws += int((end & self.throw_loose_known).sum())
        self.loose_throws += int(
            (end & self.throw_loose_known & self.throw_loose).sum()
        )
        self.measured_throws += int((end & self.throw_measured).sum())
        self.intercepted_throws += int(
            (end & self.throw_measured & self.throw_contact).sum()
        )
        self.held_throws += int((end & self.throw_success).sum())
        self.active_throw[end] = False
        self.throw_contact[end] = False
        self.throw_success[end] = False
        self.throw_measured[end] = True
        self.throw_loose[end] = False
        self.intercept_persist[end | launch] = 0
        self.throw_intercepted[end | launch] = False
        self.throw_loose_known[end] = True
        self.active_throw |= launch & ~done
        if "termination_reasons" in tr and bool(
            tr.get("termination_reasons_known", torch.zeros_like(done)).all()
        ):
            counts = tr["termination_reasons"].sum(0)
            self.reasons = counts if self.reasons is None else self.reasons + counts
        self.reward[done] = 0
        self.length[done] = 0
        self.success[done] = False
        self.contact_prev = contact & ~done

    def metrics(self):
        result = {
            "experience/completed_episodes": self.completed,
            "experience/launches": self.launches,
            "experience/contact_measured_fraction": self.contact_known_steps
            / max(1, self.steps),
            "experience/completed_strict_successes": self.successes,
            "experience/completed_observed_throws": self.completed_throws,
            "experience/contact_measured_throws": self.measured_throws,
            "experience/termination_reasons_measured": float(self.reasons is not None),
        }
        if self.completed:
            result.update(
                {
                    "experience/episode_return": self.return_sum / self.completed,
                    "experience/episode_length": self.length_sum / self.completed,
                    "experience/completed_strict_success_rate": self.successes
                    / self.completed,
                }
            )
        if self.contact_known_steps:
            result.update(
                {
                    "experience/box_contact_edges": self.contact_edges,
                    "experience/box_contact_steps": self.contact_steps,
                }
            )
        if self.completed_throws:
            result["experience/clean2_interception_per_completed_throw"] = (
                self.interception_throws / self.completed_throws
            )
            result["experience/strict_hold_per_completed_throw"] = (
                self.held_throws / self.completed_throws
            )
        if self.measured_throws:
            result["experience/box_robot_contact_per_measured_throw"] = (
                self.intercepted_throws / self.measured_throws
            )
        if self.loose_measured_throws:
            result["experience/original_loose_catch_per_measured_throw"] = (
                self.loose_throws / self.loose_measured_throws
            )
        if self.reasons is not None:
            result.update(
                {
                    f"experience/termination_reason_{i}": int(n)
                    for i, n in enumerate(self.reasons)
                }
            )
        return result

    def state_dict(self):
        # Physics is not restored: save completed totals, never join open episodes.
        return {
            k: v
            for k, v in vars(self).items()
            if isinstance(v, (int, float)) or k == "reasons"
        }

    def load_state_dict(self, state):
        for key, value in state.items():
            if key in vars(self) and (
                isinstance(value, (int, float)) or key == "reasons"
            ):
                setattr(self, key, value)
