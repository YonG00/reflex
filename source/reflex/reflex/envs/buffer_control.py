"""Positive buffering ablation: lean freedom -> intake -> transition retention."""
import json
import os
from pathlib import Path
import torch
from reflex.envs.buffer_control_core import BufferConfig,BufferState


def enabled():return int(os.environ.get('BOX_BUFFER_STAGE','0'))>0


def _state(env):
    if not hasattr(env,'_buffer_state'):env._buffer_state=BufferState(env.num_envs,env.device)
    return env._buffer_state


def buffer_throw_reset(env,ids,hold_init=False):
    state=_state(env)
    if getattr(state,'log_path',''):
        indices=torch.arange(env.num_envs,device=env.device)[ids]
        rows=indices[state.started[indices]].tolist()
        if rows:
            with Path(state.log_path).open('a') as f:
                for i in rows:
                    f.write(json.dumps(dict(event='throw_end',step=int(env.common_step_counter),env_id=i,
                        controlled_seconds=float(state.control_time[i]),longest_control_seconds=float(state.max_control[i]),contacted=bool(state.contacted[i]),
                        caught=bool(state.caught[i]),grounded=bool(state.grounded[i]),lost=bool(state.lost[i]),
                        episode_step=int(env.episode_length_buf[i])))+'\n')
    if hasattr(env,'_persistent_hold'):
        hold=env._persistent_hold
        if getattr(state,'log_path',''):
            indices=torch.arange(env.num_envs,device=env.device)[ids]
            with Path(state.log_path).open('a') as f:
                for j in indices[hold.ever[indices]].tolist():
                    f.write(json.dumps(dict(event='persistent_hold_throw_end',step=int(env.common_step_counter),env_id=j,
                        hold_seconds=float(hold.duration[j]),longest_hold_seconds=float(hold.longest[j])))+'\n')
        hold.reset(ids)
    state.reset(ids,hold_init)
    if hasattr(env,'_dynamic_buffer_state'):env._dynamic_buffer_state.reset(ids)
    sensor=env.scene.sensors.get('buffer_box_contacts')
    if sensor is not None:sensor.clear_history(ids)


def context(env):
    state=_state(env);step=int(env.common_step_counter)
    if state.last_step==step:return state.cached
    path=os.environ.get('BOX_BUFFER_LOG_PATH',f'outputs/buffer_control/buffer_{os.getpid()}.jsonl')
    if getattr(state,'log_path',None)!=path:
        Path(path).parent.mkdir(parents=True,exist_ok=True);state.log_path=path
    from reflex.envs.escape_penalty import torso_relative_kinematics
    from reflex.envs.base.chest_v5_terms import compute_strict_components
    robot,box=env.scene['robot'],env.scene['balloon'];data=robot.data
    if not hasattr(env,'_buffer_ids'):
        env._buffer_ids=dict(torso=list(robot.body_names).index('torso_link'),
            knees=[i for i,n in enumerate(robot.joint_names) if 'knee' in n])
    i=env._buffer_ids['torso'];bp,bv=box.data.root_pos_w,box.data.root_lin_vel_w
    rp,rv=data.root_pos_w,data.root_lin_vel_w
    pos,vel=torso_relative_kinematics(env);radius=pos.norm(dim=-1)
    radial=(pos*vel).sum(-1)/radius.clamp_min(1e-6)
    q=data.body_quat_w[:,i];w,x,y,z=q.unbind(-1)
    up=1.-2*(x*x+y*y)
    pitch=torch.atan2(2*(w*y-x*z),up.clamp_min(1e-4))
    ground=env.scene.env_origins[:,2]
    root_height=rp[:,2]-ground;torso_height=data.body_pos_w[:,i,2]-ground
    safe=(root_height>.45)&(torso_height>.55)&(up>.25)
    bq=box.data.root_quat_w;bw,bx,by,bz=bq.unbind(-1)
    row=torch.stack((2*(bx*bz-bw*by),2*(by*bz+bw*bx),1-2*(bx*bx+by*by)),-1)
    bottom=bp[:,2]-ground-(row.abs()*env._box_half_extents).sum(-1)
    sensor=env.scene['buffer_box_contacts'];_=sensor.data
    forces=sensor.pair_force_history.norm(dim=-1)
    nr=sensor.robot_filter_count
    contact=(forces[...,:nr]>1.).flatten(1).any(-1)
    floor=(forces[...,nr:]>1.).flatten(1).any(-1)
    delta=bp-rp;dist=delta.norm(dim=-1);rel=bv-rv
    closing=(-(delta*rel).sum(-1)/dist.clamp_min(1e-6)).clamp_min(0)
    tca=(-(delta*rel).sum(-1)/rel.square().sum(-1).clamp_min(1e-6)).clamp(0,.6)
    miss=(delta+tca[:,None]*rel).norm(dim=-1)
    incoming=(miss<.75)&(dist<1.2)&(closing>.5)&(bv.norm(dim=-1)>1.)&(
        dist/closing.clamp_min(.01)<.6)&(bottom>.08)
    cfg=BufferConfig()
    out=state.update(step=step,dt=env.step_dt,launched=env._box_launched,
        contact=contact,ground_contact=floor,box_bottom=bottom,safe=safe,radius=radius,
        radial_speed=radial,relative_speed=vel.norm(dim=-1),incoming=incoming,
        strict=compute_strict_components(env)['strict_all'],cfg=cfg)
    from reflex.stage3_experiments import experiment, weights
    group=experiment()
    if group=='B':
        from reflex.envs.dynamic_buffer import DynamicBufferState
        from isaaclab.utils.math import quat_apply_inverse
        if not hasattr(env,'_dynamic_buffer_state'):
            env._dynamic_buffer_state=DynamicBufferState(env.num_envs,env.device)
        dynamic=env._dynamic_buffer_state
        # Unclipped time to closest approach: do not inherit the old <0.9m gate.
        ttc=-(delta*rel).sum(-1)/rel.square().sum(-1).clamp_min(1e-6)
        predicted=(delta+ttc[:,None]*rel).norm(dim=-1)
        local_delta=quat_apply_inverse(q,delta)
        front=(local_delta[:,0]>0)&(local_delta[:,0]>local_delta[:,1].abs())
        angular=quat_apply_inverse(q,data.body_ang_vel_w[:,i])[:,1]
        dyn=dynamic.update(step=step,dt=env.step_dt,active=env._box_launched&~state.hold_init,
            safe=safe,grounded=state.grounded,lost=state.lost,contacted=state.contacted,
            age=state.age,ttc=ttc,miss=predicted,closing=closing,
            box_velocity=bv,front=front,root_velocity=rv,torso_omega_y=angular)
        intake=dyn['raw']
        out['relax']=torch.where(out['control']|dyn['pre_gate']|dyn['post_gate'],1.-out['restore'],0.)
        first=dyn['first'].nonzero().flatten().tolist()
        if first:
            with Path(path).open('a') as f:
                for j in first:
                    f.write(json.dumps(dict(event='dynamic_first_contact',step=step,env_id=j,
                        pitch=float(pitch[j]),torso_omega=data.body_ang_vel_w[j,i].tolist(),
                        pelvis_velocity=rv[j].tolist(),knees=data.joint_pos[j,env._buffer_ids['knees']].tolist()))+'\n')
    else:
        # Original formula and gate are retained bit-for-bit for S3/A/C.
        from reflex.envs.throw import compliant_intake_reward
        intake=torch.where(out['intake_gate'],compliant_intake_reward(env),0.)
    if os.environ.get('BOX_PERSISTENT_HOLD','0')=='1':
        from reflex.envs.persistent_hold import state as hold_state
        hold=hold_state(env)
        names=sensor.robot_filter_names
        arms=[j for j,n in enumerate(names) if any(k in n for k in ('wrist','elbow','forearm','hand'))]
        if not arms:raise RuntimeError('persistent hold requires explicit real arm contact filters')
        arm_contact=(forces[...,arms]>1.).flatten(1).any(-1)
        angular=(box.data.root_ang_vel_w-data.body_ang_vel_w[:,i]).norm(dim=-1)
        held=hold.update(step=step,dt=env.step_dt,active=env._box_launched&~state.hold_init,
            safe=safe,grounded=state.grounded,lost=state.lost,arm_contact=arm_contact,
            radius=radius,relative_speed=vel.norm(dim=-1),angular_speed=angular)
        # No catch-age deadline can restore posture while control remains valid.
        out['relax']=torch.maximum(out['relax'],held['mix'])
        out['restore']=out['restore']*(1-held['mix'])
        out['retain']=torch.where(held['held'],held['reward'],out['retain'])
        out['persistent_held']=held['held']
    out['intake']=intake
    every=int(os.environ.get('BOX_BUFFER_LOG_EVERY','50'))
    if every>0 and step%every==0:
        denom=state.counts['throws'].clamp_min(1)
        stage=int(os.environ.get('BOX_BUFFER_STAGE','0'))
        metrics=dict(stage=stage,raw_contact_rate=float(state.counts['physical_contacts']/denom),contact_rate=float(state.counts['contacts']/denom),
            catch_success_rate=float(state.counts['catches']/denom),
            mean_control_seconds=float(state.counts['control_seconds']/denom),
            intake_raw_mean=float(intake.mean()),retain_raw_mean=float(out['retain'].mean()),
            intake_reward_mean=float(intake.mean())*weights()[0] if stage>=2 else 0.,
            retain_reward_mean=float(out['retain'].mean())*weights()[1] if stage>=3 else 0.,
            back_lean_degrees=float((-pitch).clamp_min(0).mean()*180/torch.pi),
            knee_flexion=float(data.joint_pos[:,env._buffer_ids['knees']].abs().mean()),
            max_back_lean_degrees=float((-pitch).clamp_min(0).max()*180/torch.pi),
            floor_contacts=int(floor.sum()),grounded_throws=int(state.grounded.sum()),
            intake_active=int((intake>0).sum()),retain_active=int((out['retain']>0).sum()),
            invalid_positive_intake=int(((intake>0)&(~safe|floor|(bottom<=.08))).sum()),
            invalid_positive_retain=int(((out['retain']>0)&(~safe|floor|(bottom<=.08))).sum()),
            mean_longest_control_seconds=float((state.max_control*state.contacted).sum()/state.contacted.sum().clamp_min(1)),
            root_ang_speed=float(getattr(data,'root_ang_vel_w',torch.zeros_like(rv)).norm(dim=-1).mean()),
            root_height=float(root_height.mean()),safe_fraction=float(safe.float().mean()),
            **{k:float(v) for k,v in state.counts.items()})
        if os.environ.get('BOX_PERSISTENT_HOLD','0')=='1':
            metrics.update(persistent_hold_fraction=float(hold.held.float().mean()),
                persistent_hold_mix=float(hold.mix.mean()),persistent_hold_max_mix=float(hold.mix.max()),
                persistent_hold_raw_mean=float(held['reward'].mean()),
                persistent_hold_max_seconds=float(hold.longest.max()),
                persistent_reference_scale_mean=float((1-hold.mix).mean()),
                persistent_arm_base_weight=float(getattr(env,'_hold_arm_base_weight',rp.new_zeros(env.num_envs)).mean()),
                **{f'persistent_{k}':float(v) for k,v in hold.totals.items()})
        metrics['experiment']=group
        if group=='B':
            total=(dynamic.totals['pre_samples']+dynamic.totals['post_samples']).clamp_min(1)
            metrics.update({f'dynamic_{k}':float(v) for k,v in dynamic.totals.items()})
            metrics.update(dynamic_pre_seconds_per_throw=float(dynamic.totals['pre_seconds']/denom),
                dynamic_post_seconds_per_throw=float(dynamic.totals['post_seconds']/denom),
                dynamic_pre_active_fraction=float(dynamic.totals['pre_samples']/((step+1)*env.num_envs)),
                dynamic_post_active_fraction=float(dynamic.totals['post_samples']/((step+1)*env.num_envs)))
            for key in ('lean','yield','drop'):
                metrics[f'dynamic_{key}_nonzero_fraction']=float(dynamic.totals[key+'_nonzero']/total)
                metrics[f'dynamic_{key}_active_mean']=float(dynamic.totals[key+'_sum']/total)
        if group=='C':
            from reflex.envs.flex_arms import phase_mask
            phase=phase_mask(env)
            metrics['flex_active_fraction']=float(phase.float().mean())
            metrics['flex_shoulder_scale_mean']=float(getattr(env,'_flex_scale',rp.new_full((env.num_envs,),.02)).mean())
            metrics['flex_target_delta_max']=float(getattr(env,'_flex_target_delta_max',rp.new_zeros(())))
            metrics['flex_target_clip_max']=float(getattr(env,'_flex_target_clip',rp.new_zeros(env.num_envs)).max())
            for key in ('shoulder','elbow'):
                ids=[j for j,n in enumerate(robot.joint_names) if key in n]
                if ids:
                    angles=data.joint_pos[:,ids]
                    low=getattr(env,'_flex_'+key+'_low',angles.clone())
                    high=getattr(env,'_flex_'+key+'_high',angles.clone())
                    low=torch.minimum(low,angles);high=torch.maximum(high,angles)
                    setattr(env,'_flex_'+key+'_low',low);setattr(env,'_flex_'+key+'_high',high)
                    metrics['flex_'+key+'_range_mean']=float((high-low).mean())
            for name in ('cradle_pose','cradle_ready','cradle_ready_dev','box_on_bed'):
                term=getattr(env.cfg.rewards,name,None)
                if term is not None:metrics['flex_'+name+'_effective_weight_mean']=float(torch.where(phase,.1,1.).mean())*term.weight
        env.extras.setdefault('log',{}).update({f'Buffer/{k}':v for k,v in metrics.items() if isinstance(v,(int,float))})
        with Path(path).open('a') as f:f.write(json.dumps(dict(step=step,**metrics))+'\n')
        print('[buffer_control] '+json.dumps(dict(step=step,**metrics)),flush=True)
    return out


def phase_upright_gate(env,original):
    c=context(env)
    return c['safe'].float()*(c['relax']+(1.-c['relax'])*original)


def buffer_torso_asym(env,fwd_max=.21,back_max=.31,lat_max=.35):
    from reflex.envs.throw import torso_asym_penalty
    c=context(env)
    front_side=torso_asym_penalty(env,fwd_max=fwd_max,back_max=torch.pi,lat_max=lat_max)
    original=torso_asym_penalty(env,fwd_max=fwd_max,back_max=back_max,lat_max=lat_max)
    return front_side+c['restore']*(original-front_side)


def buffer_upright_hold(env,asset_cfg=None,target_gbx=-.15,sigma=.2):
    from reflex.envs.throw import upright_hold_reward
    from isaaclab.managers import SceneEntityCfg
    c=context(env)
    base=upright_hold_reward(env,asset_cfg or SceneEntityCfg('balloon'),target_gbx,sigma)
    return base*c['safe']*(1.-c['relax'])


def buffer_compliant_intake(env):return context(env)['intake']


def retain_reward(env):return context(env)['retain']
