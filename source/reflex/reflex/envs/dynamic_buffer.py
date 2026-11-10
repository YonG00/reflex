"""B only: bounded TTC pre-window and first-contact post-window, pure tensors."""
import torch


class DynamicBufferState:
    def __init__(self,n,device):
        self.pre=torch.zeros(n,device=device);self.post=self.pre.clone()
        self.seen=torch.zeros(n,dtype=torch.bool,device=device)
        self.previous_valid=self.seen.clone();self.previous_velocity=torch.zeros(n,3,device=device)
        self.previous_front=self.seen.clone();self.incident=torch.zeros(n,3,device=device)
        self.front=self.seen.clone();self.last_step=None;self.cached=None
        self.totals={k:self.pre.new_zeros(()) for k in ('pre_seconds','post_seconds','pre_samples','post_samples',
            'lean_nonzero','yield_nonzero','drop_nonzero','lean_sum','yield_sum','drop_sum','first_contacts')}

    def reset(self,ids):
        for v in (self.pre,self.post,self.seen,self.previous_valid,self.previous_velocity,
                  self.previous_front,self.incident,self.front):v[ids]=0
        if self.cached is not None:
            for v in self.cached.values():v[ids]=0

    def update(self,*,step,dt,active,safe,grounded,lost,contacted,age,ttc,miss,
               closing,box_velocity,front,root_velocity,torso_omega_y):
        if self.last_step==step:return self.cached
        first=contacted&~self.seen
        source=torch.where(self.previous_valid[:,None],self.previous_velocity,box_velocity)
        self.incident[first]=source[first]
        self.front[first]=torch.where(self.previous_valid,self.previous_front,front)[first]
        self.seen|=contacted
        healthy=active&safe&~grounded&~lost
        pre=healthy&~contacted&(ttc>0)&(ttc<=.5)&(miss<.75)&(closing>.5)&(
            box_velocity.norm(dim=-1)>1.)&(self.pre+dt<=.5+1e-6)
        post=healthy&contacted&(age<.35-1e-6)&(self.post+dt<=.35+1e-6)
        self.pre+=pre.float()*dt;self.post+=post.float()*dt
        v=torch.where(contacted[:,None],self.incident,box_velocity)
        frontal=torch.where(contacted,self.front,front)
        direction=v[:,:2]/v[:,:2].norm(dim=-1,keepdim=True).clamp_min(1e-6)
        lean=torch.where(frontal,(-torso_omega_y/1.2).clamp(0,1),0.)
        yielding=((root_velocity[:,:2]*direction).sum(-1)/.75).clamp(0,1)
        dropping=(-root_velocity[:,2]/.5).clamp(0,1)
        gate=pre|post
        raw=torch.where(gate,.35*lean+.40*yielding+.25*dropping,0.)
        for key,value in [('pre_seconds',pre.sum()*dt),('post_seconds',post.sum()*dt),
            ('pre_samples',pre.sum()),('post_samples',post.sum()),('first_contacts',first.sum()),
            ('lean_nonzero',((lean>0)&gate).sum()),('yield_nonzero',((yielding>0)&gate).sum()),
            ('drop_nonzero',((dropping>0)&gate).sum()),('lean_sum',(lean*gate).sum()),
            ('yield_sum',(yielding*gate).sum()),('drop_sum',(dropping*gate).sum())]:self.totals[key]+=value
        self.previous_velocity.copy_(box_velocity);self.previous_front.copy_(front)
        self.previous_valid.copy_(active&~contacted)
        self.cached=dict(raw=raw,pre_gate=pre,post_gate=post,lean=lean*gate,yielding=yielding*gate,
                         dropping=dropping*gate,first=first)
        self.last_step=step
        return self.cached
