"""Bounded intake/retention windows; no fixed receiving pose or action changes."""
from dataclasses import dataclass
import math
import torch


@dataclass(frozen=True)
class BufferConfig:
    intake_budget: float = .6
    control_window: float = 2.
    contact_grace: float = .3
    control_radius: float = .9
    escape_radius: float = 1.2
    outward_limit: float = 2.
    stable_delay: float = .5
    restore_time: float = 1.

    def __post_init__(self):
        for k,v in vars(self).items():
            if not math.isfinite(v) or v<=0:raise ValueError(k)
        if self.escape_radius<=self.control_radius:raise ValueError('escape_radius')


class BufferState:
    def __init__(self,n,device):
        self.started=torch.zeros(n,dtype=torch.bool,device=device)
        self.contacted=self.started.clone();self.raw_contacted=self.started.clone();self.grounded=self.started.clone()
        self.lost=self.started.clone();self.settled=self.started.clone()
        self.hold_init=self.started.clone();self.caught=self.started.clone()
        self.age=torch.zeros(n,device=device)
        self.since_contact=torch.full((n,),100.,device=device)
        self.intake_used=self.age.clone();self.stable_time=self.age.clone()
        self.settle_age=self.age.clone();self.control_time=self.age.clone()
        self.strict_time=self.age.clone()
        self.continuous_control=self.age.clone();self.max_control=self.age.clone()
        self.counts={k:self.age.new_zeros(()) for k in ['throws','physical_contacts','contacts','catches','control_seconds']}
        self.last_step=None;self.cached=None

    def reset(self,ids,hold_init=False):
        for x in (self.started,self.contacted,self.raw_contacted,self.grounded,self.lost,self.settled,
                  self.hold_init,self.caught,self.age,self.intake_used,self.stable_time,
                  self.settle_age,self.control_time,self.strict_time,self.continuous_control,self.max_control):x[ids]=0
        self.since_contact[ids]=100.
        self.hold_init[ids]=hold_init
        if self.cached is not None:
            for x in self.cached.values():x[ids]=0

    def update(self,*,step,dt,launched,contact,ground_contact,box_bottom,safe,
               radius,radial_speed,relative_speed,incoming,strict,cfg):
        if not math.isfinite(dt) or dt<=0:raise ValueError("dt")
        if self.last_step==step:return self.cached
        active=launched & ~self.hold_init
        new_throw=active & ~self.started
        self.counts['throws']+=new_throw.sum();self.started|=active
        raw_hit=active & contact
        self.counts['physical_contacts']+=(raw_hit & ~self.raw_contacted).sum()
        self.raw_contacted|=raw_hit
        hit=raw_hit & safe & ~ground_contact & (box_bottom>.08) & (radius<cfg.escape_radius)
        new_contact=hit & ~self.contacted
        self.counts['contacts']+=new_contact.sum();self.contacted|=hit
        self.age+=((self.contacted & ~new_contact).float()*dt)
        self.since_contact=torch.where(hit,0.,self.since_contact+dt)
        self.grounded|=active & (ground_contact | (box_bottom<=.08))
        self.lost|=self.contacted & (radius>cfg.escape_radius)
        healthy=active & safe & ~self.grounded & ~self.lost
        pre=healthy & ~self.contacted & incoming & (self.intake_used<cfg.intake_budget-1e-6)
        self.intake_used+=pre.float()*dt
        control=healthy & self.contacted & (radius<cfg.control_radius) & (
            self.since_contact<=cfg.contact_grace+1e-6) & (radial_speed<cfg.outward_limit) & (
            self.age+dt<=cfg.control_window+1e-6)
        stable=control & (radius<.65) & (relative_speed<.5)
        self.stable_time=torch.where(stable,self.stable_time+dt,0.)
        newly_settled=stable & (self.stable_time>=cfg.stable_delay-1e-6) & ~self.settled
        self.settled|=newly_settled
        self.settle_age+=(self.settled & ~newly_settled).float()*dt
        restore=(self.settle_age/cfg.restore_time).clamp(0,1)
        transition=pre | control
        relax=torch.where(transition,1.-restore,0.)
        # Reward survival of CONTROL, not just contact history. Bounded per throw,
        # with a short detach grace and a handoff to the original stable rewards.
        proximity=(1.-radius/cfg.control_radius).clamp(0,1)
        ramp=(self.age/.1).clamp(0,1)
        retain=torch.where(control,(.5+.5*proximity)*ramp*(1.-restore),0.)
        self.continuous_control=torch.where(control,self.continuous_control+dt,0.)
        self.max_control=torch.maximum(self.max_control,self.continuous_control)
        self.control_time+=control.float()*dt
        self.counts['control_seconds']+=control.sum()*dt
        self.strict_time=torch.where(active & strict,self.strict_time+dt,0.)
        caught=active & ~self.caught & (self.strict_time>=.6-1e-6)
        self.caught|=caught;self.counts['catches']+=caught.sum()
        self.cached=dict(intake_gate=pre,retain=retain,control=control,relax=relax,
                         restore=restore,safe=safe,contact=hit,stable=stable)
        self.last_step=step
        return self.cached
