"""C extension: holding depends on physical control, never elapsed catch time."""
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class HoldConfig:
    enter_seconds: float = .12
    enter_radius: float = .75
    keep_radius: float = .9
    enter_speed: float = 1.
    keep_speed: float = 2.
    contact_grace: float = .3
    release_seconds: float = .25


class PersistentHoldState:
    def __init__(self,n,device):
        self.held=torch.zeros(n,dtype=torch.bool,device=device)
        self.entered=self.held.clone();self.ever=self.held.clone()
        self.recent=torch.full((n,),100.,device=device)
        self.ready=torch.zeros(n,device=device);self.mix=self.ready.clone()
        self.duration=self.ready.clone();self.continuous=self.ready.clone();self.longest=self.ready.clone()
        self.generation=torch.zeros(n,dtype=torch.long,device=device)
        self.last_step=None;self.cached=None
        self.totals={k:self.ready.new_zeros(()) for k in ('hold_entries','holding_throws','hold_seconds','lost_after_hold')}

    def reset(self,ids):
        for v in (self.held,self.entered,self.ever,self.ready,self.mix,self.duration,self.continuous,self.longest):v[ids]=0
        self.recent[ids]=100.;self.generation[ids]+=1
        if self.cached is not None:
            for v in self.cached.values():v[ids]=0

    def update(self,*,step,dt,active,safe,grounded,lost,arm_contact,radius,relative_speed,angular_speed,cfg=HoldConfig()):
        if self.last_step==step:return self.cached
        healthy=active&safe&~grounded&~lost
        hit=healthy&arm_contact
        self.recent=torch.where(hit,0.,self.recent+dt)
        # Entry needs sustained real support, not a passing chest collision.
        candidate=healthy&(self.recent<=.08+1e-6)&(radius<cfg.enter_radius)&(
            relative_speed<cfg.enter_speed)&(angular_speed<2.)
        self.ready=torch.where(candidate,self.ready+dt,0.)
        keep=healthy&(self.recent<=cfg.contact_grace+1e-6)&(radius<cfg.keep_radius)&(
            relative_speed<cfg.keep_speed)&(angular_speed<6.)
        previous=self.held.clone()
        self.held=keep&(self.held|(self.ready>=cfg.enter_seconds-1e-6))
        self.entered=self.held&~previous
        self.totals['hold_entries']+=self.entered.sum()
        self.totals['holding_throws']+=(self.held&~self.ever).sum()
        self.totals['lost_after_hold']+=(previous&~self.held).sum()
        self.ever|=self.held
        # This interpolation is triggered by acquisition/loss, not catch age.
        self.mix=torch.where(self.held,(self.mix+dt/cfg.enter_seconds).clamp(max=1.),
                             (self.mix-dt/cfg.release_seconds).clamp_min(0.))
        self.mix=torch.where(healthy,self.mix,0.)
        self.duration+=self.held.float()*dt
        self.continuous=torch.where(self.held,self.continuous+dt,0.)
        self.longest=torch.maximum(self.longest,self.continuous)
        self.totals['hold_seconds']+=self.held.sum()*dt
        proximity=(1-radius/cfg.keep_radius).clamp(0,1)
        slow=torch.exp(-(relative_speed/.5).square()-(angular_speed/2.).square())
        reward=torch.where(self.held,(.5+.5*proximity)*slow*(self.continuous/.1).clamp(max=1.),0.)
        self.cached=dict(held=self.held,mix=self.mix,reward=reward,entered=self.entered)
        self.last_step=step
        return self.cached
