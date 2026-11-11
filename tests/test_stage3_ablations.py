"""CPU contracts; actual dynamics/learning require the prepared GPU regression."""
import ast
import os
from pathlib import Path
import subprocess
import sys
import types
import pytest
import torch
from test_buffer_control import load, modules, ENVS, ROOT

experiments=load('_s3_flags',ENVS.parent/'stage3_experiments.py')
dynamic=load('_s3_dynamic',ENVS/'dynamic_buffer.py')
flex=load('_s3_flex',ENVS/'flex_arms.py')
physical=load('_s3_physical',ENVS.parent/'eval/physical_hold.py')

@pytest.fixture(autouse=True)
def flags(monkeypatch):
    for key in experiments.FLAGS.values():monkeypatch.delenv(key,raising=False)
    monkeypatch.delenv('BOX_BUFFER_INTAKE_WEIGHT',raising=False)
    monkeypatch.delenv('BOX_BUFFER_RETAIN_WEIGHT',raising=False)
    monkeypatch.setitem(sys.modules,'reflex.stage3_experiments',experiments)
    monkeypatch.setitem(sys.modules,'reflex.envs.dynamic_buffer',dynamic)
    monkeypatch.setitem(sys.modules,'reflex.envs.flex_arms',flex)

@pytest.mark.parametrize('group,expected',[('S3',(.75,.5)),('A',(.75,5.)),('B',(6.,.5)),('C',(.75,.5))])
def test_independent_parameters(group,expected):
    env={'BOX_BUFFER_STAGE':'3'}
    if group!='S3':env[experiments.FLAGS[group]]='1'
    assert experiments.experiment(env)==group and experiments.weights(env)==expected

@pytest.mark.parametrize('env',[{'BOX_RETAIN_X10':'1','BOX_DYNAMIC_BUFFER':'1','BOX_BUFFER_STAGE':'3'},
    {'BOX_FLEX_ARMS':'1','BOX_BUFFER_STAGE':'2'},{'BOX_RETAIN_X10':'yes'},
    {'BOX_RETAIN_X10':'1','BOX_BUFFER_STAGE':'3','BOX_BUFFER_RETAIN_WEIGHT':'3'}])
def test_invalid_experiment_is_rejected(env):
    with pytest.raises(ValueError):experiments.weights(env)


def inputs(step=0,**kw):
    t=torch.ones(2,dtype=torch.bool);z=torch.zeros(2,dtype=torch.bool)
    d=dict(step=step,dt=.02,active=t,safe=t,grounded=z,lost=z,contacted=z,
        age=torch.zeros(2),ttc=torch.full((2,),.3),miss=torch.zeros(2),closing=torch.ones(2)*3,
        box_velocity=torch.tensor([[-3.,0,0.],[-3.,0,0.]]),front=t,
        root_velocity=torch.tensor([[-.75,0,-.5],[-.75,0,-.5]]),torso_omega_y=torch.full((2,),-1.2))
    d.update(kw);return d


def test_b_pre_budget_cache_partial_reset_and_hold_init_equivalent():
    s=dynamic.DynamicBufferState(2,'cpu')
    for step in range(80):
        args=inputs(step,active=torch.tensor([True,False]))
        out=s.update(**args);before=s.pre.clone();assert s.update(**args) is out
        assert torch.equal(s.pre,before)
    assert torch.allclose(s.pre,torch.tensor([.5,0.]),atol=1e-6)
    assert float(s.totals['pre_seconds'])==pytest.approx(.5)
    s.reset(torch.tensor([0]));assert s.update(**inputs(80))['raw'][0]==1

@pytest.mark.parametrize('kw',[{'safe':torch.zeros(2,dtype=torch.bool)},
    {'grounded':torch.ones(2,dtype=torch.bool)},{'lost':torch.ones(2,dtype=torch.bool)},
    {'ttc':torch.ones(2)*.6},{'ttc':torch.ones(2)*-.1},{'miss':torch.ones(2)},
    {'closing':torch.zeros(2)},{'active':torch.zeros(2,dtype=torch.bool)}])
def test_b_never_rewards_invalid_window(kw):
    assert not dynamic.DynamicBufferState(2,'cpu').update(**inputs(**kw))['raw'].any()


def test_b_post_uses_first_incident_direction_and_does_not_refresh_on_detach():
    s=dynamic.DynamicBufferState(2,'cpu');s.update(**inputs())
    for step in range(1,35):
        out=s.update(**inputs(step,contacted=torch.ones(2,dtype=torch.bool),age=torch.ones(2)*(step-1)*.02,
            box_velocity=torch.ones(2,3)*3,ttc=torch.ones(2)*10))
        if step==1:assert torch.allclose(out['raw'],torch.ones(2))
    assert s.post.max()<=.35 and not out['post_gate'].any() and s.totals['first_contacts']==2
    assert s.totals['post_seconds']==pytest.approx(.68,abs=1e-5)


def test_b_side_motion_allowed_wrong_direction_not_rewarded():
    s=dynamic.DynamicBufferState(2,'cpu')
    out=s.update(**inputs(front=torch.zeros(2,dtype=torch.bool),
        box_velocity=torch.tensor([[0.,3.,0.],[0.,3.,0.]]),
        root_velocity=torch.tensor([[0.,.75,0.],[0.,-.75,0.]])))
    assert torch.allclose(out['raw'],torch.tensor([.4,0.])) and out['lean'].sum()==0


def test_c_ramp_bounded_waiting_unchanged_and_same_step_cached():
    s=flex.ShoulderRamp(2,'cpu')
    assert (s.update(0,.02,torch.ones(2,dtype=torch.bool))==.02).all()
    for step in range(1,300):out=s.update(step,.02,torch.tensor([True,False]))
    assert torch.allclose(out,torch.tensor([.2,.02]))
    assert s.update(299,.02,torch.zeros(2,dtype=torch.bool)) is out
    assert (s.update(300,.02,torch.zeros(2,dtype=torch.bool))==.02).all()
    assert s.update(301,.02,torch.tensor([True,False]))[0]<.04


def test_c_action_keeps_elbow_and_waiting_targets_and_limits(monkeypatch):
    class Base:
        def __init__(self,cfg,env):
            self._env=env;self._asset=env.robot;self._joint_ids=[0,1,2]
            self._scale=torch.tensor([[.02,.35,.5],[.02,.35,.5]])
        def process_actions(self,a):self._processed_actions=a*self._scale+1.
    monkeypatch.setitem(sys.modules,'reflex.envs.gimbal_action',types.SimpleNamespace(GimbalJointPositionAction=Base))
    monkeypatch.setattr(flex,'phase_mask',lambda env:torch.tensor([True,False]))
    action=load('_s3_flex_action',ENVS/'flex_arms_action.py')
    limits=torch.tensor([[[-1.,1.1],[-2.,2.],[-2.,2.]]]).repeat(2,1,1)
    env=types.SimpleNamespace(num_envs=2,device='cpu',step_dt=.02,common_step_counter=300,
        robot=types.SimpleNamespace(joint_names=['left_shoulder_pitch_joint','left_elbow_joint','hip'],
            data=types.SimpleNamespace(joint_pos_limits=limits)))
    a=action.FlexArmsAction(None,env);v=torch.ones(2,3)*2
    for step in range(300,320):env.common_step_counter=step;a.process_actions(v)
    assert torch.allclose(a._processed_actions[:,1:],v[:,1:]*torch.tensor([.35,.5])+1)
    assert torch.equal(a._processed_actions[1],v[1]*torch.tensor([.02,.35,.5])+1)
    assert a._processed_actions[0,0]==pytest.approx(1.1)
    assert torch.equal(a._scale,torch.tensor([[.02,.35,.5],[.02,.35,.5]]))


def hold_args(**kw):
    d=dict(dt=.02,active=torch.ones(2,dtype=torch.bool),safe=torch.ones(2,dtype=torch.bool),
        arm_contact=torch.ones(2,dtype=torch.bool),forbidden=torch.zeros(2,dtype=torch.bool),
        ground_contact=torch.zeros(2,dtype=torch.bool),bottom=torch.ones(2)*.5,radius=torch.ones(2)*.5,
        relative_speed=torch.ones(2)*.2,angular_speed=torch.zeros(2))
    d.update(kw);return d


def test_shared_physical_hold_accepts_any_supported_pose_and_rethrow_resets():
    s=physical.PhysicalHoldState(2,'cpu')
    for _ in range(30):s.update(**hold_args())
    assert s.success.all() and float(s.longest.min())==pytest.approx(.6)
    s.reset(torch.tensor([1]));assert s.success.tolist()==[True,False]

@pytest.mark.parametrize('kw',[{'arm_contact':torch.zeros(2,dtype=torch.bool)},
    {'safe':torch.zeros(2,dtype=torch.bool)},{'ground_contact':torch.ones(2,dtype=torch.bool)},
    {'bottom':torch.zeros(2)},{'forbidden':torch.ones(2,dtype=torch.bool)},
    {'radius':torch.ones(2)},{'relative_speed':torch.ones(2)}, {'angular_speed':torch.ones(2)*3},
    {'active':torch.zeros(2,dtype=torch.bool)}])
def test_physical_hold_rejects_torso_only_floor_fallen_fast_or_forbidden(kw):
    s=physical.PhysicalHoldState(2,'cpu')
    for _ in range(60):s.update(**hold_args(**kw))
    assert not s.success.any()


def fake_env(monkeypatch):
    def inverse(q,v):
        xyz=-q[:,1:];w=q[:,:1]
        return v+2*(w*torch.cross(xyz,v,dim=-1)+torch.cross(xyz,torch.cross(xyz,v,dim=-1),dim=-1))
    monkeypatch.setitem(sys.modules,'isaaclab.utils.math',types.SimpleNamespace(quat_apply_inverse=inverse))
    core=load('_s3_esc_core',ENVS/'escape_penalty_core.py')
    monkeypatch.setitem(sys.modules,'reflex.envs.escape_penalty_core',core)
    monkeypatch.setitem(sys.modules,'reflex.envs.escape_penalty',load('_s3_esc_math',ENVS/'escape_penalty.py'))
    monkeypatch.setitem(sys.modules,'reflex.envs.base.chest_v5_terms',types.SimpleNamespace(compute_strict_components=lambda env:{'strict_all':torch.zeros(2,dtype=torch.bool)}))
    class Scene(dict):
        @property
        def sensors(self):return self
    data=types.SimpleNamespace(root_pos_w=torch.tensor([[0.,0,.8],[0.,0,.8]]),root_lin_vel_w=torch.tensor([[-.2,0,-.2],[-.2,0,-.2]]),
        body_pos_w=torch.tensor([[[0.,0,1.1]],[[0.,0,1.1]]]),body_lin_vel_w=torch.zeros(2,1,3),
        body_ang_vel_w=torch.tensor([[[0.,-1.,0]],[[0.,-1.,0]]]),body_quat_w=torch.tensor([[[1.,0,0,0]],[[1.,0,0,0]]]),joint_pos=torch.ones(2,2))
    bd=types.SimpleNamespace(root_pos_w=torch.tensor([[.7,0.,1.1],[.7,0.,1.1]]),root_lin_vel_w=torch.tensor([[-3.,0,0],[-3.,0,0]]),root_quat_w=data.body_quat_w[:,0].clone())
    hist=torch.zeros(2,4,1,2,3)
    sensor=types.SimpleNamespace(data=object(),robot_filter_count=1,pair_force_history=hist,clear_history=lambda ids:hist.__setitem__(ids,0))
    scene=Scene(robot=types.SimpleNamespace(data=data,body_names=['torso_link'],joint_names=['left_knee_joint','right_knee_joint']),
        balloon=types.SimpleNamespace(data=bd),buffer_box_contacts=sensor);scene.env_origins=torch.zeros(2,3)
    return types.SimpleNamespace(scene=scene,num_envs=2,device='cpu',step_dt=.02,common_step_counter=0,
        _box_launched=torch.ones(2,dtype=torch.bool),_box_half_extents=torch.ones(2,3)*.1,extras={},episode_length_buf=torch.zeros(2,dtype=torch.long))


def test_all_off_zero_difference_against_fixed_s3_adapter(modules,monkeypatch,tmp_path):
    mod,_=modules
    original=subprocess.check_output(['git','show','4138039:source/reflex/reflex/envs/buffer_control.py'],cwd=ROOT,text=True)
    path=tmp_path/'s3_original.py';path.write_text(original);old=load('_s3_original',path)
    left=fake_env(monkeypatch);right=fake_env(monkeypatch)
    for step in range(130):
        for env in (left,right):
            env.common_step_counter=step
            h=env.scene['buffer_box_contacts'].pair_force_history;h.zero_()
            if 5<=step<70:h[...,0,0]=3
            if step==80:h[0,...,1,0]=3
        a=old.context(left);b=mod.context(right)
        assert a.keys()==b.keys()
        for k in a:assert torch.equal(a[k],b[k]),(step,k)
        for k in left._buffer_state.counts:assert torch.equal(left._buffer_state.counts[k],right._buffer_state.counts[k])
        if step in (85,110):
            for adapter,env in ((old,left),(mod,right)):adapter.buffer_throw_reset(env,torch.tensor([0]),hold_init=step==110)
    # Core, observation, events and algorithm remain byte-identical to baseline.
    for rel in ('envs/buffer_control_core.py','teacher/train.py'):
        tracked='source/reflex/reflex/'+rel
        assert (ROOT/tracked).read_bytes()==subprocess.check_output(['git','show','4138039:'+tracked],cwd=ROOT)


def test_b_adapter_bypasses_old_point_nine_gate_and_resets(modules,monkeypatch):
    mod,_=modules;monkeypatch.setenv('BOX_DYNAMIC_BUFFER','1')
    env=fake_env(monkeypatch);env.scene['balloon'].data.root_pos_w[:,0]=1.3
    out=mod.context(env)
    assert not out['intake_gate'].any() and (out['intake']>0).all()
    assert env.extras['log']['Buffer/intake_reward_mean']==pytest.approx(float(out['intake'].mean())*6)
    mod.buffer_throw_reset(env,torch.tensor([0]),True);env.common_step_counter+=1
    assert mod.context(env)['intake'][0]==0 and env._dynamic_buffer_state.pre[0]==0


def test_a_adapter_only_changes_logged_weight(modules,monkeypatch):
    mod,_=modules;monkeypatch.setenv('BOX_RETAIN_X10','1');env=fake_env(monkeypatch)
    env.scene['buffer_box_contacts'].pair_force_history[...,0,0]=3
    mod.context(env);env.common_step_counter=1;out=mod.context(env)
    assert env.extras['log']['Buffer/retain_reward_mean']==pytest.approx(float(out['retain'].mean())*5)

@pytest.mark.parametrize('group,expected',[('S3',(.75,.5)),('A',(.75,5.)),('B',(6.,.5)),('C',(.75,.5))])
def test_real_config_wires_only_selected_experiment(modules,monkeypatch,group,expected):
    mod,_=modules
    class Term:
        def __init__(self,**kw):self.__dict__.update(kw)
    class Gimbal:pass
    class Flex:pass
    monkeypatch.setitem(sys.modules,'isaaclab.managers',types.SimpleNamespace(RewardTermCfg=Term))
    monkeypatch.setitem(sys.modules,'isaaclab.sensors',types.SimpleNamespace(ContactSensorCfg=Term))
    monkeypatch.setitem(sys.modules,'isaaclab.utils',types.SimpleNamespace(configclass=lambda cls:cls))
    monkeypatch.setitem(sys.modules,'reflex.envs',types.SimpleNamespace(buffer_control=mod))
    monkeypatch.setitem(sys.modules,'reflex.envs.escape_penalty_sensor',types.SimpleNamespace(EscapeRobotContactSensor=object))
    monkeypatch.setitem(sys.modules,'reflex.envs.gimbal_action',types.SimpleNamespace(GimbalJointPositionAction=Gimbal))
    monkeypatch.setitem(sys.modules,'reflex.envs.flex_arms_action',types.SimpleNamespace(FlexArmsAction=Flex))
    if group!='S3':monkeypatch.setenv(experiments.FLAGS[group],'1')
    old=lambda env:None
    cfg=types.SimpleNamespace(scene=types.SimpleNamespace(balloon=types.SimpleNamespace(prim_path='Balloon',spawn=types.SimpleNamespace()),
        robot=types.SimpleNamespace(prim_path='Robot'),terrain=types.SimpleNamespace(prim_path='/World/ground')),
        rewards=types.SimpleNamespace(torso_asym=Term(func=old,weight=-.5),upright_hold=Term(func=old,weight=1.5),
            compliant_intake=Term(func=old,weight=0),hold_still=Term(func=old,weight=1),
            **{k:Term(func=old,weight=2,params={}) for k in ('cradle_pose','cradle_ready','cradle_ready_dev','box_on_bed')}),
        actions=types.SimpleNamespace(joint_pos=Term(class_type=Gimbal,scale='unchanged')),
        events=types.SimpleNamespace(hover='hover',launch='launch'),decimation=4)
    load('_s3_real_config_'+group,ENVS/'buffer_control_config.py').configure_buffer_control(cfg)
    assert (cfg.rewards.compliant_intake.weight,cfg.rewards.retain_reward.weight)==expected
    assert cfg.actions.joint_pos.class_type is (Flex if group=='C' else Gimbal)
    assert cfg.actions.joint_pos.scale=='unchanged' and cfg.rewards.hold_still.func is old
    assert vars(cfg.events)==dict(hover='hover',launch='launch')
    assert cfg.rewards.cradle_pose.weight==2
    assert (cfg.rewards.cradle_pose.func is old)==(group!='C')


def test_c_pose_factors_preserve_waiting_and_hold_velocity_part(monkeypatch):
    monkeypatch.setattr(flex,'phase_mask',lambda env,update=False:torch.tensor([True,False]))
    for name in ('cradle_pose','cradle_ready','box_on_bed'):
        monkeypatch.setitem(flex._INNER,name,lambda env,asset_cfg=None:torch.ones(2)*2)
        assert torch.allclose(getattr(flex,name)(None),torch.tensor([.2,2.]))
    source=ast.parse((ENVS/'throw.py').read_text())
    code=ast.Module(body=[n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='hold_still_reward'],type_ignores=[])
    class Entity:
        def __init__(self,name):self.name=name
    namespace=dict(torch=torch,os=os,SceneEntityCfg=Entity,_get_posture_ids=lambda env:{'arm_joints':[0,1]})
    exec(compile(code,'hold_still_reward','exec'),namespace)
    robot=types.SimpleNamespace(data=types.SimpleNamespace(root_pos_w=torch.zeros(2,3),root_lin_vel_w=torch.zeros(2,3),joint_vel=torch.ones(2,2)*4))
    box=types.SimpleNamespace(data=types.SimpleNamespace(root_pos_w=torch.ones(2,3)*.1,root_lin_vel_w=torch.zeros(2,3)))
    env=types.SimpleNamespace(scene={'robot':robot,'balloon':box})
    old=namespace['hold_still_reward'](env)
    monkeypatch.setenv('BOX_FLEX_ARMS','1');new=namespace['hold_still_reward'](env)
    assert new[0]==1 and new[1]==old[1]
    box.data.root_lin_vel_w[0,0]=.2
    assert float(namespace['hold_still_reward'](env)[0])==pytest.approx(float(torch.exp(torch.tensor(-.2/.25))))


def test_physical_detach_ground_latch_and_nonconstant_contact():
    s=physical.PhysicalHoldState(2,'cpu');s.update(**hold_args())
    for _ in range(35):s.update(**hold_args(arm_contact=torch.zeros(2,dtype=torch.bool)))
    assert not s.success.any() and float(s.control.max())<=.14+1e-5
    s.update(**hold_args(ground_contact=torch.ones(2,dtype=torch.bool)))
    for _ in range(40):s.update(**hold_args())
    assert not s.success.any()


def test_evaluation_adapter_shared_first_throw_alignment_and_rethrow(modules,monkeypatch):
    env=fake_env(monkeypatch);bd=env.scene['balloon'].data
    bd.root_ang_vel_w=torch.zeros(2,3);bd.root_lin_vel_w.zero_();bd.root_pos_w[:,0]=.4
    sensor=env.scene['buffer_box_contacts'];sensor.robot_filter_names=['left_wrist_yaw_link','right_wrist_yaw_link']
    sensor.robot_filter_count=2;sensor.pair_force_history=torch.zeros(2,4,1,3,3)
    sensor.pair_force_history[...,0,0]=3
    evaluator=physical.PhysicalHoldEvaluator(env)
    evaluator.update()  # Existing launched/HOLD_INIT state is adopted, never scored.
    assert evaluator.index.sum()==0
    env._box_launched.zero_();evaluator.update();env._box_launched.fill_(True)
    for _ in range(30):evaluator.update()
    assert evaluator.state.success.all()
    env._box_launched.zero_();evaluator.update();env._box_launched.fill_(True);evaluator.update()
    rows=[dict(env=j,idx=k) for j in range(2) for k in (1,2)]
    evaluator.finish(rows)
    assert all(r['physical_hold_success']==(r['idx']==1) for r in rows)
    assert all('arm_contact_seconds' in r for r in rows)


def test_checkpoint_and_finished_run_acceptance_tools(tmp_path):
    checkpoint=tmp_path/'model_599.pt'
    torch.save(dict(iter=599,optimizer_state_dict={},model_state_dict={
        'actor.0.weight':torch.zeros(2,183),'critic.0.weight':torch.zeros(2,305)}),checkpoint)
    cmd=[sys.executable,str(ROOT/'scripts/s3_checkpoint_check.py'),str(checkpoint)]
    assert subprocess.run(cmd,capture_output=True).returncode==0
    torch.save(dict(iter=599,model_state_dict={'actor.0.weight':torch.zeros(2,170),'critic.0.weight':torch.zeros(2,292)}),checkpoint)
    assert subprocess.run(cmd,capture_output=True).returncode!=0
    log=tmp_path/'train.log';diag=tmp_path/'metrics.jsonl'
    log.write_text('Learning iteration 599/611\nLearning iteration 600/611\n')
    diag.write_text('{"experiment":"A","invalid_positive_intake":0,"invalid_positive_retain":0}\n')
    cmd=[sys.executable,str(ROOT/'scripts/check_s3_run.py'),str(log),str(diag),'2']
    assert subprocess.run(cmd,capture_output=True).returncode==0
    log.write_text('Learning iteration 599/611\n')
    assert subprocess.run(cmd,capture_output=True).returncode!=0


def test_shell_switches_do_not_silently_stack():
    env=dict(os.environ,ROOT=str(ROOT),GROUP='A',ISAACLAB_PY=sys.executable,PYTHONPATH=str(ROOT/'source/reflex'),BOX_DYNAMIC_BUFFER='1')
    p=subprocess.run(['bash','-c','source "$ROOT/scripts/s3_experiment_env.sh"'],env=env,capture_output=True,text=True)
    assert p.returncode!=0 and 'independent' in p.stderr


def test_summary_uses_matched_first_throws_not_rethrows(tmp_path):
    import json
    for group in ('S3','A','B','C'):
        for seed in (8,9):
            rows=[dict(env=i,idx=1,caught=i==0,caught_loose=i<2,physical_hold_success=i<3,
                physical_contact=True,fell=False,physical_ground=False,physical_escape=False,
                control_seconds=.5,longest_control_seconds=.4) for i in range(320)]
            rows.append(dict(rows[0],idx=2,caught=True))
            (tmp_path/f'{group}_s{seed}.json').write_text(json.dumps(dict(throw_log=rows)))
    cmd=[sys.executable,str(ROOT/'scripts/summarize_s3_ablations.py'),str(tmp_path)]
    p=subprocess.run(cmd,capture_output=True,text=True);assert p.returncode==0,p.stderr
    result=json.loads(p.stdout)
    assert result['A'][0]['strict']==1/320 and result['A'][0]['rethrows']==1
    (tmp_path/'C_s9.json').unlink()
    assert subprocess.run(cmd,capture_output=True).returncode!=0


def test_shell_baseline_rejects_inherited_experiment():
    env=dict(os.environ,ROOT=str(ROOT),GROUP='S3',ISAACLAB_PY=sys.executable,
        PYTHONPATH=str(ROOT/'source/reflex'),BOX_RETAIN_X10='1')
    p=subprocess.run(['bash','-c','source "$ROOT/scripts/s3_experiment_env.sh"'],env=env,capture_output=True,text=True)
    assert p.returncode!=0 and 'requested S3' in p.stderr


def test_c_preserves_original_trust_wrapper_inner_parameters(monkeypatch):
    import inspect
    monkeypatch.setattr(flex,'phase_mask',lambda env,update=False:torch.tensor([True,False]))
    for name in ('cradle_pose','cradle_ready','cradle_ready_dev','box_on_bed'):
        seen={}
        def original(env,inner,asset_cfg=None,sigma=None):
            seen.update(inner=inner,asset_cfg=asset_cfg,sigma=sigma)
            return torch.ones(2)
        monkeypatch.setitem(flex._INNER,name,original)
        fn=getattr(flex,name)
        # Match real RewardManager parameter validation, with the legacy inner key.
        inspect.signature(fn).bind(None,inner='on_bed',asset_cfg='balloon',sigma=.2)
        assert torch.allclose(fn(None,inner='on_bed',asset_cfg='balloon',sigma=.2),torch.tensor([.1,1.]))
        assert seen==dict(inner='on_bed',asset_cfg='balloon',sigma=.2)
