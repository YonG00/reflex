"""Tensor-only impact bookkeeping; no simulator events, RNG or model changes."""
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class ImpactBufferConfig:
    w_pre: float = 0.2
    w_damp: float = 0.5
    w_escape: float = 1.0
    window: float = 0.5
    ramp: float = 0.1
    pre_range: float = 0.8
    speed_scale: float = 8.0
    escape_scale: float = 0.6
    control_radius: float = 0.65
    free_displacement: float = 0.08
    contact_force: float = 1.0

    def __post_init__(self):
        for name in ('w_pre', 'w_damp', 'w_escape', 'free_displacement', 'control_radius'):
            if not 0 <= getattr(self, name) < float('inf'):
                raise ValueError(name)
        for name in ('window', 'ramp', 'pre_range', 'speed_scale', 'escape_scale', 'contact_force'):
            if not 0 < getattr(self, name) < float('inf'):
                raise ValueError(name)


class ImpactBufferState:
    """One first contact per throw, including partial resets and same-step reuse.

    Position/velocity inputs are expressed in the live torso frame. ``gap`` is
    the box-surface to nearest forearm/hand gap, not centre-to-centre distance.
    ``arm_speed`` is full box-vs-contact-region relative speed; ``approach`` is
    its positive closing component. Filtered contact impulses are supplied by
    the simulator adapter, never inferred from proximity or total body forces.
    """
    def __init__(self, n, device, dtype=torch.float32):
        self.n, self.device = n, device
        self.contacted = torch.zeros(n, dtype=torch.bool, device=device)
        self.started = self.contacted.clone()
        self.suppressed = self.contacted.clone()
        self.strict = self.contacted.clone()
        self.near_torso = self.contacted.clone()
        self.age = torch.zeros(n, device=device, dtype=dtype)
        self.first_pos = torch.zeros(n, 3, device=device, dtype=dtype)
        self.first_speed = self.age.clone()
        self.previous_speed = self.age.clone()
        self.previous_valid = self.contacted.clone()
        self.max_escape = self.age.clone()
        self.strict_time = self.age.clone()
        self.counts = {k: torch.zeros((), device=device, dtype=dtype) for k in
            ('throws', 'contacts', 'strict_success', 'closed_throws', 'missed',
             'near_missed', 'early_terminated', 'post_window_truncated')}
        self.max_escape_seen = self.age.new_zeros(())
        self.last_step, self.cached = None, None

    def reset(self, ids, *, hold_init=False, terminated=None):
        active = self.started[ids] & ~self.suppressed[ids]
        self.counts['closed_throws'] += active.sum()
        self.counts['missed'] += (active & ~self.contacted[ids]).sum()
        self.counts['near_missed'] += (active & self.near_torso[ids] & ~self.contacted[ids]).sum()
        if hasattr(self, 'cfg_window'):
            self.counts['post_window_truncated'] += (
                active & self.contacted[ids] & (self.age[ids] < self.cfg_window)).sum()
        if terminated is not None:
            self.counts['early_terminated'] += (active & terminated).sum()
        for value in (self.contacted, self.started, self.strict, self.near_torso,
                      self.previous_valid):
            value[ids] = False
        for value in (self.age, self.first_pos, self.first_speed, self.previous_speed,
                      self.max_escape, self.strict_time):
            value[ids] = 0
        self.suppressed[ids] = hold_init
        # A reset after reward evaluation must not make unrelated environments
        # advance twice if diagnostics read this function again in the same step.
        if self.cached is not None:
            for value in self.cached.values():
                value[ids] = 0

    def update(self, *, step, dt, launched, contact, position_b, velocity_b,
               gap, approach, arm_speed, strict_now, cfg):
        if self.last_step == step:
            return self.cached
        self.cfg_window = cfg.window
        active = launched & ~self.suppressed
        start = active & ~self.started
        self.counts['throws'] += start.sum()
        self.started |= active
        self.near_torso |= active & (position_b.norm(dim=-1) < 1.2)
        # Increment only a previously contacted throw; first contact starts at t=0.
        self.age += (self.contacted & active).to(self.age.dtype) * dt
        first = active & ~self.contacted & contact
        self.first_pos[first] = position_b[first]
        # Previous control sample approximates PRE-impact speed; current speed
        # can already have dropped after the impulse. Both are logged separately.
        self.first_speed[first] = torch.where(self.previous_valid, self.previous_speed, arm_speed)[first]
        self.age[first] = 0
        self.contacted |= first
        self.counts['contacts'] += first.sum()
        self.previous_speed.copy_(arm_speed)
        self.previous_valid.copy_(active & ~self.contacted)
        pre_gate = active & ~self.contacted & (gap < cfg.pre_range) & (approach > 0)
        proximity = (1 - gap / cfg.pre_range).clamp(0, 1)
        pre = torch.where(pre_gate, proximity * (approach / cfg.speed_scale).square(), 0.)
        post_gate = active & self.contacted & (self.age <= cfg.window + 1e-6)
        speed = velocity_b.norm(dim=-1)
        damp = torch.where(post_gate, (self.age / cfg.ramp).clamp(0, 1) *
                           (speed / cfg.speed_scale).square(), 0.)
        # Radially outward motion only: sideways/inward absorption is free.
        # A rigid translation/rotation of robot+box leaves this invariant.
        radius = position_b.norm(dim=-1)
        baseline = self.first_pos.norm(dim=-1)
        outward = (radius - baseline).clamp_min(0)
        allowance = torch.maximum(baseline + cfg.free_displacement,
                                   torch.full_like(baseline, cfg.control_radius))
        escaped = (radius - allowance).clamp_min(0)
        escape = torch.where(post_gate, (escaped / cfg.escape_scale).square(), 0.)
        self.max_escape = torch.maximum(self.max_escape, torch.where(post_gate, outward, 0.))
        self.max_escape_seen = torch.maximum(self.max_escape_seen, self.max_escape.max())
        self.strict_time = torch.where(active & strict_now, self.strict_time + dt, 0.)
        success = active & (self.strict_time >= 0.6 - 1e-6) & ~self.strict
        self.strict |= success
        self.counts['strict_success'] += success.sum()
        parts = torch.stack((-cfg.w_pre * pre, -cfg.w_damp * damp, -cfg.w_escape * escape), -1)
        total = parts.sum(-1)
        self.cached = dict(total=total, parts=parts, approach=approach, pre_gate=pre_gate,
            first=first, first_speed=self.first_speed.clone(), post_speed=speed,
            post_gate=post_gate, max_escape=self.max_escape.clone(), active=active,
            contact_age=self.age.clone(), contact=contact)
        self.last_step = step
        return self.cached
