"""State persistence, safe loss, reference gradients and achieved-pose action tests."""
import importlib.util
from pathlib import Path
import sys
import types
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1];ENVS=ROOT/'source/reflex/reflex/envs'


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m


core=load('_persistent_core',ENVS/'persistent_hold_core.py')


def inputs(step=0,**kw):
    yes=torch.ones(2,dtype=torch.bool);no=torch.zeros_like(yes)
    d=dict(step=step,dt=.02,active=yes,safe=yes,grounded=no,lost=no,arm_contact=yes,
        radius=torch.ones(2)*.5,relative_speed=torch.ones(2)*.1,angular_speed=torch.zeros(2))
    d.update(kw);return d


def test_holding_survives_ten_seconds_without_postcatch_timer_recovery():
    s=core.PersistentHoldState(2,'cpu')
    for step in range(600):out=s.update(**inputs(step))
    assert s.held.all() and (s.mix==1).all()
    assert float(s.longest.min())>11. and (out['reward']>0).all()
    assert s.totals['hold_entries']==2 and s.totals['holding_throws']==2
    duration=s.duration.clone();assert s.update(**inputs(599)) is out
    assert torch.equal(s.duration,duration)


@pytest.mark.parametrize('kw',[{'active':torch.zeros(2,dtype=torch.bool)},
    {'safe':torch.zeros(2,dtype=torch.bool)},{'grounded':torch.ones(2,dtype=torch.bool)},
    {'lost':torch.ones(2,dtype=torch.bool)},{'arm_contact':torch.zeros(2,dtype=torch.bool)},
    {'relative_speed':torch.ones(2)*3},{'radius':torch.ones(2)}, {'angular_speed':torch.ones(2)*7}])
def test_hover_hold_init_chest_only_floor_fallen_fast_box_cannot_enter(kw):
    s=core.PersistentHoldState(2,'cpu')
    for step in range(30):out=s.update(**inputs(step,**kw))
    assert not s.held.any() and out['reward'].sum()==0


def test_real_loss_and_reset_end_reward_and_detach_does_not_refresh_budget():
    s=core.PersistentHoldState(2,'cpu')
    for step in range(40):s.update(**inputs(step))
    # Brief contact separation is allowed while the box remains controlled.
    for step in range(40,48):out=s.update(**inputs(step,arm_contact=torch.zeros(2,dtype=torch.bool)))
    assert out['held'].all()
    for step in range(48,60):out=s.update(**inputs(step,arm_contact=torch.zeros(2,dtype=torch.bool)))
    assert not out['held'].any() and out['reward'].sum()==0
    for step in range(60,80):s.update(**inputs(step))
    assert s.held.all()
    s.reset(torch.tensor([0]));assert not s.held[0] and s.held[1] and s.generation[0]==1
    # A reset in the same step cannot double-account another environment.
    before=s.duration[1].clone();s.update(**inputs(79));assert s.duration[1]==before
    out=s.update(**inputs(80,active=torch.tensor([False,True])))
    assert out['reward'][0]==0 and out['reward'][1]>0


@pytest.mark.parametrize('kw',[{'safe':torch.zeros(2,dtype=torch.bool)},
    {'grounded':torch.ones(2,dtype=torch.bool)},{'lost':torch.ones(2,dtype=torch.bool)}])
def test_real_floor_escape_or_fall_immediately_ends_relaxation(kw):
    s=core.PersistentHoldState(2,'cpu')
    for step in range(20):s.update(**inputs(step))
    out=s.update(**inputs(20,**kw))
    assert out['reward'].sum()==0 and s.mix.sum()==0 and not s.held.any()


def test_low_relative_speed_is_rewarded_more_without_forcing_reference_pose():
    s=core.PersistentHoldState(2,'cpu')
    for step in range(20):s.update(**inputs(step))
    out=s.update(**inputs(20,relative_speed=torch.tensor([.05,1.5])))
    assert out['held'].all() and out['reward'][0]>100*out['reward'][1]


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.setitem(sys.modules,'reflex.envs.persistent_hold_core',core)
    return load('_persistent_adapter',ENVS/'persistent_hold.py')


def test_tracking_removes_only_upper_reference_while_preserving_lower_gradients(adapter,monkeypatch):
    class Command:
        cfg=types.SimpleNamespace(body_names=['torso_link','left_elbow_link','left_knee_link'])
    env=types.SimpleNamespace(command_manager=types.SimpleNamespace(get_term=lambda name:Command()))
    errors=torch.tensor([[3.,4.,.2],[3.,4.,.2]],requires_grad=True)
    def original(env,command_name,std,body_names=None):
        names=Command.cfg.body_names;ids=list(range(3)) if body_names is None else [names.index(n) for n in body_names]
        return torch.exp(-errors[:,ids].mean(-1)/std)
    adapter._ORIGINALS['pos']=original
    monkeypatch.setattr(adapter,'mix',lambda env:torch.tensor([1.,0.]))
    r=adapter.body_tracking(env,'motion',1.,'pos')
    assert torch.allclose(r,torch.tensor([torch.exp(torch.tensor(-.2)),torch.exp(torch.tensor(-2.4))]))
    r.sum().backward()
    assert errors.grad[0,:2].abs().sum()==0 and errors.grad[0,2]!=0
    assert errors.grad[1,:2].abs().sum()>0
    adapter._ORIGINALS['anchor']=lambda env,command_name,std:torch.ones(2)*.4
    assert torch.equal(adapter.anchor_tracking(env,'motion',.3,'anchor'),torch.tensor([1.,.4]))


def test_achieved_arm_base_does_not_return_to_ready_and_policy_stays_free(adapter):
    env=types.SimpleNamespace(num_envs=2,device='cpu')
    s=adapter.state(env);s.held[:]=True;s.mix[:]=1
    action=types.SimpleNamespace(_env=env,_joint_ids=[0,1],
        _asset=types.SimpleNamespace(joint_names=['shoulder','elbow'],data=types.SimpleNamespace(joint_pos=torch.tensor([[.7,.8],[.3,.4]]))))
    cols=torch.tensor([0,1]);delta=torch.tensor([[.1,.2],[.1,.2]])
    initial=adapter.arm_reference(action,torch.ones(2,2)*-2,delta,cols)
    assert torch.allclose(initial+delta,action._asset.data.joint_pos)
    # A native down/raised reference change no longer moves a held target.
    assert torch.equal(adapter.arm_reference(action,torch.ones(2,2)*5,delta,cols),initial)
    changed=delta+.05
    assert torch.allclose(adapter.arm_reference(action,torch.zeros(2,2),changed,cols)+changed,action._asset.data.joint_pos+.05)
    s.reset(torch.tensor([0]));base=torch.ones(2,2)*-1
    assert torch.equal(adapter.arm_reference(action,base,delta,cols)[0],base[0])
    s.held[0]=True;s.mix[0]=1;action._asset.data.joint_pos[0]=torch.tensor([.9,1.])
    assert torch.allclose(adapter.arm_reference(action,base,delta,cols)[0]+delta[0],action._asset.data.joint_pos[0])


def test_flag_requires_only_c(monkeypatch):
    module=load('_persistent_flags',ENVS.parent/'stage3_experiments.py')
    for env in ({'BOX_PERSISTENT_HOLD':'1'},{'BOX_PERSISTENT_HOLD':'yes'},
                {'BOX_PERSISTENT_HOLD':'1','BOX_BUFFER_STAGE':'3','BOX_RETAIN_X10':'1'}):
        with pytest.raises(ValueError):module.experiment(env)
    assert module.experiment(dict(BOX_PERSISTENT_HOLD='1',BOX_BUFFER_STAGE='3',BOX_FLEX_ARMS='1'))=='C'


def test_adapter_keeps_low_speed_hold_reward_and_relaxation_after_two_seconds(monkeypatch,tmp_path):
    from test_buffer_control import modules
    from test_stage3_ablations import fake_env
    # Use the literal adapter with the same fake physics boundary as existing tests.
    monkeypatch.setitem(sys.modules,'reflex.envs.persistent_hold_core',core)
    adapter=load('_persistent_full_adapter',ENVS/'persistent_hold.py')
    monkeypatch.setitem(sys.modules,'reflex.envs.persistent_hold',adapter)
    monkeypatch.setenv('BOX_BUFFER_STAGE','3');monkeypatch.setenv('BOX_FLEX_ARMS','1')
    monkeypatch.setenv('BOX_PERSISTENT_HOLD','1');monkeypatch.setenv('BOX_BUFFER_LOG_EVERY','0')
    monkeypatch.setenv('BOX_BUFFER_LOG_PATH',str(tmp_path/'hold.jsonl'))
    oldcore=load('_persistent_buffer_core',ENVS/'buffer_control_core.py')
    monkeypatch.setitem(sys.modules,'reflex.envs.buffer_control_core',oldcore)
    fake_throw=types.SimpleNamespace(compliant_intake_reward=lambda env:torch.zeros(2))
    monkeypatch.setitem(sys.modules,'reflex.envs.throw',fake_throw)
    env=fake_env(monkeypatch)
    env.scene['robot'].data.body_ang_vel_w.zero_()
    box=env.scene['balloon'].data;box.root_pos_w[:,0]=.5;box.root_lin_vel_w.zero_();box.root_ang_vel_w=torch.zeros(2,3)
    sensor=env.scene['buffer_box_contacts']
    sensor.robot_filter_count=2;sensor.robot_filter_names=['left_wrist_yaw_link','torso_link']
    sensor.pair_force_history=torch.zeros(2,4,1,3,3)
    sensor.pair_force_history[...,0,0]=4
    sensor.clear_history=lambda ids:sensor.pair_force_history.__setitem__(ids,0)
    mod=load('_persistent_full_buffer',ENVS/'buffer_control.py')
    monkeypatch.setitem(sys.modules,'reflex.envs.buffer_control',mod)
    for step in range(600):env.common_step_counter=step;out=mod.context(env)
    assert env._buffer_state.age.min()>11 and env._persistent_hold.held.all()
    assert not out['control'].any()  # old transition cap is still two seconds
    assert (out['retain']>0).all() and (out['relax']==1).all() and out['restore'].sum()==0
    sensor.pair_force_history[...,2,0]=4;env.common_step_counter+=1
    out=mod.context(env)
    assert out['retain'].sum()==0 and not out['persistent_held'].any()
    mod.buffer_throw_reset(env,torch.tensor([0]),hold_init=True)
    env.common_step_counter+=1
    assert mod.context(env)['retain'][0]==0 and env._persistent_hold.mix[0]==0
