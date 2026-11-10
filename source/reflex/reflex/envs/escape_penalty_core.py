"""One-shot per-throw failure, independent of Isaac Lab and policy architecture."""
from dataclasses import dataclass
import math
import torch


@dataclass(frozen=True)
class EscapeConfig:
    amount: float = 100.0
    distance: float = 1.0
    radial_speed: float = 1.0
    persistence: float = 0.15
    contact_force: float = 1.0  # contact detection only, NOT a force penalty

    def __post_init__(self):
        for name, value in vars(self).items():
            if not math.isfinite(value) or value < 0 or (name != 'amount' and value == 0):
                raise ValueError(f'Invalid escape {name}: {value}')


class EscapeState:
    def __init__(self, n, device):
        self.n = n
        self.contacted = torch.zeros(n, dtype=torch.bool, device=device)
        self.escaped = self.contacted.clone()
        self.started = self.contacted.clone()
        self.ever_contacted = self.contacted.clone()
        self.caught = self.contacted.clone()
        self.hold_init = self.contacted.clone()
        self.observed = self.contacted.clone()
        self.dwell = torch.zeros(n, device=device)
        self.catch_time = self.dwell.clone()
        self.throw_id = torch.zeros(n, dtype=torch.long, device=device)
        self.counts = {k: self.dwell.new_zeros(()) for k in (
            'throws', 'contacts', 'escapes', 'catches', 'closed_throws',
            'no_contact_terminated', 'contact_terminated_before_escape',
            'pending_escape_terminated', 'caught_after_escape', 'terminated_without_recorded_launch')}
        self.last_step, self.cached = None, None

    def throw_reset(self, ids, hold_init=False, terminated=None):
        # Called by the existing real reset/rethrow/hold-init callbacks.
        active = self.started[ids]
        self.counts['closed_throws'] += active.sum()
        if terminated is not None:
            self.counts['terminated_without_recorded_launch'] += (
                self.observed[ids] & ~active & terminated).sum()
            self.counts['no_contact_terminated'] += (active & terminated & ~self.ever_contacted[ids]).sum()
            self.counts['contact_terminated_before_escape'] += (
                active & terminated & self.ever_contacted[ids] & ~self.escaped[ids]).sum()
            self.counts['pending_escape_terminated'] += (
                active & terminated & (self.dwell[ids] > 0) & ~self.escaped[ids]).sum()
        for value in (self.contacted, self.escaped, self.started, self.ever_contacted,
                      self.caught, self.hold_init, self.observed, self.dwell, self.catch_time):
            value[ids] = 0
        self.hold_init[ids] = hold_init
        self.throw_id[ids] += 1
        if self.cached is not None:
            for value in self.cached.values():
                value[ids] = 0

    def episode_reset(self, ids, terminated=None):
        self.throw_reset(ids, terminated=terminated)

    def update(self, *, step, dt, launched, contact, position_b, velocity_b,
               strict_now, cfg):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError('dt must be positive')
        if step == self.last_step:
            return self.cached
        self.observed.fill_(True)
        active = launched & ~self.hold_init
        start = active & ~self.started
        self.started |= start
        self.counts['throws'] += start.sum()
        contact_now = active & contact
        self.contacted |= contact_now
        first_throw_contact = self.contacted & ~self.ever_contacted
        self.ever_contacted |= self.contacted
        self.counts['contacts'] += first_throw_contact.sum()
        radius = position_b.norm(dim=-1)
        outward_speed = (position_b * velocity_b).sum(-1) / radius.clamp_min(1e-6)
        candidate = active & self.contacted & ~self.escaped & (
            radius > cfg.distance) & (outward_speed > cfg.radial_speed)
        self.dwell = torch.where(candidate, self.dwell + dt, 0.)
        new_escape = candidate & (self.dwell >= cfg.persistence - 1e-6)
        self.escaped |= new_escape
        self.counts['escapes'] += new_escape.sum()
        self.catch_time = torch.where(active & strict_now, self.catch_time + dt, 0.)
        new_catch = active & ~self.caught & (self.catch_time >= 0.6 - 1e-6)
        self.caught |= new_catch
        self.counts['catches'] += new_catch.sum()
        self.counts['caught_after_escape'] += (new_catch & self.escaped).sum()
        self.cached = dict(event=-cfg.amount * new_escape.to(radius.dtype),
            new_escape=new_escape, candidate=candidate, contact_now=contact_now,
            radius=radius, radial_speed=outward_speed, dwell=self.dwell.clone())
        self.last_step = step
        return self.cached
