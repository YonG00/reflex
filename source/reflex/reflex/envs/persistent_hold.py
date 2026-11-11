"""Opt-in continuous hold, upper-body tracking gate and achieved arm base."""
import os
import torch
from reflex.envs.persistent_hold_core import PersistentHoldState

_ORIGINALS={}


def enabled():return os.environ.get('BOX_PERSISTENT_HOLD','0')=='1'


def state(env):
    if not hasattr(env,'_persistent_hold'):env._persistent_hold=PersistentHoldState(env.num_envs,env.device)
    return env._persistent_hold


def mix(env,update=True):
    if update:
        from reflex.envs.buffer_control import context
        context(env)
    return state(env).mix


def body_tracking(env,command_name,std,persistent_key,body_names=None):
    original=_ORIGINALS[persistent_key]
    full=original(env,command_name=command_name,std=std,body_names=body_names)
    command=env.command_manager.get_term(command_name)
    names=[n for n in command.cfg.body_names if (body_names is None or n in body_names)
           and any(k in n for k in ('hip','knee','ankle','foot'))]
    lower=original(env,command_name=command_name,std=std,body_names=names) if names else torch.zeros_like(full)
    m=mix(env)
    return (1-m)*full+m*lower


def anchor_tracking(env,command_name,std,persistent_key):
    return _ORIGINALS[persistent_key](env,command_name=command_name,std=std)*(1-mix(env))


def configure(cfg):
    for name in ('motion_body_pos','motion_body_ori','motion_body_lin_vel','motion_body_ang_vel',
                 'motion_global_anchor_pos','motion_global_anchor_ori'):
        term=getattr(cfg.rewards,name)
        wrapper=body_tracking if name.startswith('motion_body_') else anchor_tracking
        if term.func is not wrapper:_ORIGINALS[name]=term.func
        term.func=wrapper;term.params={**term.params,'persistent_key':name}
    print('[persistent-hold] controlled holding has no age limit or scheduled pose restoration; '
          'upper-body/torso reference rewards gated, leg tracking/AMP/safety/terminations retained',flush=True)


def arm_reference(action,base,delta,cols):
    """Replace only the arm BASE, preserving policy residuals and native callbacks.

    Entry base = current joints minus current residual, so its initial target
    starts at the achieved pose. It stays local to this throw until real loss.
    """
    env=action._env;s=state(env)
    if not hasattr(action,'_hold_arm_anchor'):
        action._hold_arm_anchor=torch.zeros_like(base.expand(env.num_envs,-1))
        action._hold_arm_seen=torch.zeros_like(s.held)
        action._hold_arm_generation=s.generation.clone()
    fresh=action._hold_arm_generation!=s.generation
    action._hold_arm_seen[fresh]=False
    action._hold_arm_generation.copy_(s.generation)
    entering=s.held&~action._hold_arm_seen
    if not hasattr(action,'_hold_arm_joints'):
        ids=list(range(len(action._asset.joint_names))) if isinstance(action._joint_ids,slice) else list(action._joint_ids)
        action._hold_arm_joints=[ids[c] for c in cols.tolist()]
    current=action._asset.data.joint_pos[:,action._hold_arm_joints]
    action._hold_arm_anchor[entering]=(current-delta)[entering]
    action._hold_arm_seen|=s.held
    action._hold_arm_seen&=(s.mix>0)
    m=s.mix[:,None]
    env._hold_arm_base_weight=s.mix
    return (1-m)*base+m*action._hold_arm_anchor
