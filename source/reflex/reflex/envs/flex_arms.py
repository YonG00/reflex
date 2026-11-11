"""C only: bounded authority ramp and receiving-phase posture relaxation."""
import torch

_INNER={}


def phase_mask(env,update=False):
    if update:
        from reflex.envs.buffer_control import context
        context(env)
    state=getattr(env,'_buffer_state',None)
    if state is None or state.cached is None:return torch.zeros(env.num_envs,dtype=torch.bool,device=env.device)
    from reflex.envs.escape_penalty import torso_relative_kinematics
    pos,_=torso_relative_kinematics(env)
    controlled=state.contacted&(state.since_contact<=.3)&(pos.norm(dim=-1)<.9)
    return env._box_launched&~state.hold_init&state.cached['safe']&~state.grounded&~state.lost&(
        state.cached['intake_gate']|controlled)


class ShoulderRamp:
    """Pure ramp: 5s startup plus 0.25s per receiving-window entrance."""
    def __init__(self,n,device):
        self.age=torch.zeros(n,device=device);self.last_step=None;self.cached=None

    def update(self,step,dt,active):
        if step==self.last_step:return self.cached
        self.age=torch.where(active,self.age+dt,0.)
        fraction=min(1.,max(0.,step*dt/5.))*(self.age/.25).clamp(0,1)
        self.cached=torch.where(active,.02+.18*fraction,.02)
        self.last_step=step
        return self.cached


def _factor(env):return torch.where(phase_mask(env,True),.1,1.)


def _call(name,env,params):
    # Preserve the complete original trust/reactive wrapper parameter chain.
    return _INNER[name](env,**{k:v for k,v in params.items() if v is not None})*_factor(env)


def cradle_pose(env,asset_cfg=None,inner=None,hold_sec=None,sigma=None,
                 z_low=None,z_high=None,near_max=None,deadzone=None):
    return _call('cradle_pose',env,dict(asset_cfg=asset_cfg,inner=inner,hold_sec=hold_sec,
        sigma=sigma,z_low=z_low,z_high=z_high,near_max=near_max,deadzone=deadzone))


def cradle_ready(env,asset_cfg=None,inner=None,hold_sec=None,sigma=None,
                 z_low=None,z_high=None,near_max=None,deadzone=None):
    return _call('cradle_ready',env,dict(asset_cfg=asset_cfg,inner=inner,hold_sec=hold_sec,
        sigma=sigma,z_low=z_low,z_high=z_high,near_max=near_max,deadzone=deadzone))


def cradle_ready_dev(env,asset_cfg=None,inner=None,hold_sec=None,sigma=None,
                 z_low=None,z_high=None,near_max=None,deadzone=None):
    return _call('cradle_ready_dev',env,dict(asset_cfg=asset_cfg,inner=inner,hold_sec=hold_sec,
        sigma=sigma,z_low=z_low,z_high=z_high,near_max=near_max,deadzone=deadzone))


def box_on_bed(env,asset_cfg=None,inner=None,hold_sec=None,sigma=None,
                 z_low=None,z_high=None,near_max=None,deadzone=None):
    return _call('box_on_bed',env,dict(asset_cfg=asset_cfg,inner=inner,hold_sec=hold_sec,
        sigma=sigma,z_low=z_low,z_high=z_high,near_max=near_max,deadzone=deadzone))



def configure(cfg):
    from reflex.envs.flex_arms_action import FlexArmsAction
    from reflex.envs.gimbal_action import GimbalJointPositionAction
    if cfg.actions.joint_pos.class_type is not GimbalJointPositionAction:
        raise ValueError('C expects the original S3 gimbal action; refusing to replace an unknown action class')
    cfg.actions.joint_pos.class_type=FlexArmsAction
    for name in ('cradle_pose','cradle_ready','cradle_ready_dev','box_on_bed'):
        term=getattr(cfg.rewards,name,None)
        if term is not None:
            wrapper=globals()[name]
            if term.func is not wrapper:_INNER[name]=term.func
            term.func=wrapper
