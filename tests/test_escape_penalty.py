"""Local CPU tests. Simulated Isaac interfaces are NOT real PhysX validation."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import types
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
ENVS = ROOT/'source/reflex/reflex/envs'


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


core = load('_escape_test_core', ENVS/'escape_penalty_core.py')
Cfg, State = core.EscapeConfig, core.EscapeState


def sample(n=2, **overrides):
    data = dict(step=0, dt=.02, launched=torch.ones(n,dtype=torch.bool),
        contact=torch.ones(n,dtype=torch.bool),position_b=torch.tensor([[1.2,0.,0.]]).repeat(n,1),
        velocity_b=torch.tensor([[2.,0,0]]).repeat(n,1),strict_now=torch.zeros(n,dtype=torch.bool),cfg=Cfg())
    data.update(overrides)
    return data


def advance(state, start=0, stop=30, **kw):
    return [state.update(**sample(step=i,**kw))['event'].clone() for i in range(start,stop)]


@pytest.mark.parametrize('field,value',[
    ('launched',torch.zeros(2,dtype=torch.bool)),('contact',torch.zeros(2,dtype=torch.bool)),
    ('position_b',torch.tensor([[.8,0,0.],[.8,0,0.]])),
    ('velocity_b',torch.tensor([[-5.,0,0],[-5.,0,0]])),
    ('velocity_b',torch.tensor([[0.,10,0],[0.,10,0]]))])
def test_every_condition_is_required(field,value):
    s=State(2,'cpu');events=advance(s,**{field:value})
    assert torch.stack(events).sum()==0 and not s.escaped.any()


def test_hover_contact_does_not_arm_next_launch():
    s=State(2,'cpu');advance(s,stop=10,launched=torch.zeros(2,dtype=torch.bool))
    assert not s.contacted.any()
    assert torch.stack(advance(s,start=10,stop=30,contact=torch.zeros(2,dtype=torch.bool))).sum()==0


def test_persistence_hysteresis_and_exactly_one_hundred():
    s=State(2,'cpu');events=advance(s,stop=7)
    assert torch.stack(events).sum()==0
    e=s.update(**sample(step=7))
    assert (e['event']==-100).all() and s.escaped.all()
    events+= [e['event'].clone()]+advance(s,start=8,stop=50)
    assert torch.equal(torch.stack(events).sum(0),torch.tensor([-100.,-100.]))
    assert s.counts['escapes']==2


def test_brief_rebound_or_recovery_resets_dwell():
    s=State(2,'cpu');advance(s,stop=6)
    assert (s.dwell>.1).all() and not s.escaped.any()
    s.update(**sample(step=6,velocity_b=torch.zeros(2,3)))
    assert (s.dwell==0).all()
    assert torch.stack(advance(s,start=7,stop=14)).sum()==0
    assert (s.update(**sample(step=14))['event']==-100).all()


def test_latch_survives_recatch_but_rethrow_allows_another_penalty():
    s=State(2,'cpu');advance(s)
    assert s.escaped.all()
    # Recatch and quiet carrying do not refund any event.
    recovered=advance(s,start=30,stop=65,position_b=torch.ones(2,3)*.1,
        velocity_b=torch.zeros(2,3),strict_now=torch.ones(2,dtype=torch.bool))
    assert torch.stack(recovered).sum()==0 and s.escaped.all() and s.caught.all()
    s.throw_reset(torch.tensor([0,1]))
    assert not s.escaped.any() and not s.contacted.any()
    assert torch.stack(advance(s,start=65,stop=95)).sum()==-200
    s.episode_reset(torch.tensor([0]))
    events=advance(s,start=95,stop=125)
    assert torch.stack(events).sum(0).tolist()==[-100.,0.]
    assert s.counts['escapes']==5


def test_rethrow_without_episode_reset_requires_new_real_contact():
    s=State(2,'cpu');advance(s,stop=3)
    s.throw_reset(torch.tensor([0,1]))
    assert not s.contacted.any() and not s.escaped.any()
    assert torch.stack(advance(s,start=3,stop=30,contact=torch.zeros(2,dtype=torch.bool))).sum()==0


def test_hold_init_is_excluded_until_real_rethrow():
    s=State(2,'cpu');s.throw_reset(torch.tensor([0,1]),hold_init=True)
    assert torch.stack(advance(s)).sum()==0 and s.counts['throws']==0
    s.throw_reset(torch.tensor([0,1]))
    assert torch.stack(advance(s,start=30,stop=60)).sum()==-200


def test_same_step_reuse_partial_reset_and_strict_thresholds():
    s=State(2,'cpu');x=sample();a=s.update(**x);b=s.update(**x)
    assert a is b and torch.allclose(s.dwell,torch.tensor([.02,.02]))
    s.episode_reset(torch.tensor([0]));s.update(**x)
    assert s.dwell[0]==0 and s.dwell[1]==.02
    z=State(2,'cpu');advance(z,position_b=torch.tensor([[1.,0,0],[1.,0,0]]))
    assert not z.escaped.any()


def test_early_fall_and_noncontact_avoidance_are_visible_limitations():
    s=State(2,'cpu');s.update(**sample(contact=torch.tensor([False,True])))
    # The existing throw/reset callback records termination before clearing.
    s.throw_reset(torch.tensor([0,1]),terminated=torch.ones(2,dtype=torch.bool))
    assert s.counts['no_contact_terminated']==1
    assert s.counts['contact_terminated_before_escape']==1
    assert s.counts['pending_escape_terminated']==1
    assert s.counts['escapes']==0  # does NOT invent a penalty before 0.15s


@pytest.mark.parametrize('dt',[.01,.02,.04])
def test_time_based_confirmation(dt):
    s=State(2,'cpu');seq=advance(s,stop=30,dt=dt)
    first=next(i for i,r in enumerate(seq) if bool(r.any()))
    assert (first+1)*dt>=.15-1e-6 and first*dt<.15
    assert torch.stack(seq).sum()==-200


@pytest.mark.parametrize('cfg',[{'amount':-1},{'distance':0},{'radial_speed':0},{'persistence':0},{'amount':float('nan')}])
def test_configuration_rejects_invalid_values(cfg):
    with pytest.raises(ValueError):Cfg(**cfg)


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.setitem(sys.modules,'reflex.envs.escape_penalty_core',core)
    return load('_escape_test_adapter',ENVS/'escape_penalty.py')


@pytest.mark.parametrize('dt',[.01,.02,.04])
def test_reward_manager_dt_compensation_event_log_and_episode_reset(adapter,monkeypatch,tmp_path,dt):
    class Scene(dict):
        @property
        def sensors(self):return self
    history=torch.zeros(2,4,1,2,3);history[:,0,0,0,0]=5.
    sensor=types.SimpleNamespace(data=object(),pair_force_history=history,
        clear_history=lambda ids:history.__setitem__(ids,0))
    env=types.SimpleNamespace(scene=Scene(escape_box_robot=sensor),num_envs=2,device='cpu',
        common_step_counter=0,step_dt=dt,_box_launched=torch.ones(2,dtype=torch.bool),
        extras={},episode_length_buf=torch.ones(2,dtype=torch.long),reset_terminated=torch.zeros(2,dtype=torch.bool))
    monkeypatch.setattr(adapter,'torso_relative_kinematics',lambda env:(
        torch.tensor([[1.2,0.,0.],[1.2,0.,0.]]),torch.tensor([[2.,0,0],[2.,0,0]])))
    monkeypatch.setitem(sys.modules,'reflex.envs.base.chest_v5_terms',types.SimpleNamespace(
        compute_strict_components=lambda env:{'strict_all':torch.zeros(2,dtype=torch.bool)}))
    path=tmp_path/'escape.jsonl';applied=[]
    for i in range(40):
        env.common_step_counter=i
        # Reuse state after real impact; no continuing physical contact required.
        if i>0:history.zero_()
        applied.append(adapter.escape_penalty(env,log_every=1,log_path=str(path))*dt)
    assert torch.stack(applied).sum(0).tolist()==[-100.,-100.]
    assert env.extras['log']['Escape/escape_rate']==1
    assert env.extras['log']['Escape/contact_rate']==1
    rows=[json.loads(s) for s in path.read_text().splitlines()]
    assert sum(r.get('event')=='escape' for r in rows)==2
    adapter.escape_throw_reset(env,torch.tensor([0]))
    assert not env._escape_state.escaped[0] and env._escape_state.escaped[1]
    assert sum(json.loads(s).get('event')=='throw_end' for s in path.read_text().splitlines())==1


def test_new_config_keeps_original_rewards_events_and_disables_old(adapter,monkeypatch):
    class Term:
        def __init__(self,**kw):self.__dict__.update(kw)
    monkeypatch.setitem(sys.modules,'isaaclab.managers',types.SimpleNamespace(EventTermCfg=Term,RewardTermCfg=Term))
    monkeypatch.setitem(sys.modules,'isaaclab.sensors',types.SimpleNamespace(ContactSensorCfg=Term))
    monkeypatch.setitem(sys.modules,'reflex.envs.escape_penalty',adapter)
    monkeypatch.setitem(sys.modules,'reflex.envs.escape_penalty_sensor',types.SimpleNamespace(EscapeRobotContactSensor=object))
    mod=load('_escape_test_config',ENVS/'escape_penalty_config.py')
    cfg=types.SimpleNamespace(scene=types.SimpleNamespace(balloon=types.SimpleNamespace(
        prim_path='{ENV_REGEX_NS}/Balloon',spawn=types.SimpleNamespace()),robot=types.SimpleNamespace(
        prim_path='{ENV_REGEX_NS}/Robot'),impact_box_arm='old'),rewards=types.SimpleNamespace(
        hold_still='original',box_on_bed='original',impact_buffer='old'),events=types.SimpleNamespace(
        reset_box='reset',hover='hover',hold_init='hold_init'),decimation=4)
    mod.configure_escape_penalty(cfg)
    assert cfg.rewards.impact_buffer is None and cfg.scene.impact_box_arm is None
    assert cfg.rewards.hold_still==cfg.rewards.box_on_bed=='original'
    assert list(vars(cfg.events))==['reset_box','hover','hold_init']
    assert cfg.rewards.escape_penalty.weight==1 and cfg.rewards.escape_penalty.params['amount']==100
    assert cfg.scene.escape_box_robot.filter_prim_paths_expr==['{ENV_REGEX_NS}/Robot/.*']
    # Verify production teacher cannot register legacy terms even with old env flags.
    tree=ast.parse((ENVS/'teacher_env_cfg.py').read_text())
    calls=[n.func.id for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
    assert 'configure_escape_penalty' not in calls and 'configure_impact_buffer' not in calls


def test_dynamic_robot_filters_exclude_floor_mesh_and_force_cancellation(monkeypatch):
    class Base:
        def _initialize_impl(self):self.filters=self.cfg.filter_prim_paths_expr
    monkeypatch.setitem(sys.modules,'reflex.envs.impact_buffer_sensor',types.SimpleNamespace(ImpactPairContactSensor=Base))
    class Prim:
        def __init__(self,name,rigid=True):self.name,self.rigid=name,rigid
        def GetPath(self):return types.SimpleNamespace(pathString='/World/envs/env_0')
        def GetName(self):return self.name
        def HasAPI(self,api):return self.rigid
    def find(expr):
        if expr.endswith('/Robot/.*'):return [Prim('pelvis'),Prim('torso_link'),Prim('left_wrist_yaw_link'),Prim('mesh',False)]
        return [Prim('env_0')]
    monkeypatch.setitem(sys.modules,'isaaclab',types.SimpleNamespace(sim=types.SimpleNamespace(find_matching_prims=find)))
    monkeypatch.setitem(sys.modules,'pxr',types.SimpleNamespace(UsdPhysics=types.SimpleNamespace(RigidBodyAPI=object)))
    mod=load('_escape_test_sensor',ENVS/'escape_penalty_sensor.py')
    sensor=mod.EscapeRobotContactSensor();sensor.cfg=types.SimpleNamespace(
        prim_path='/World/envs/env_.*/Balloon',filter_prim_paths_expr=['/World/envs/env_.*/Robot/.*'])
    sensor._initialize_impl()
    assert len(sensor.filters)==3 and not any('ground' in x or 'mesh' in x for x in sensor.filters)
    assert any('torso_link' in x for x in sensor.filters)


def test_torso_translation_rotation_and_com_velocity_compensation(adapter,monkeypatch):
    def inverse(q,v):
        xyz=-q[:,1:];w=q[:,:1]
        return v+2*(w*torch.cross(xyz,v,dim=-1)+torch.cross(xyz,torch.cross(xyz,v,dim=-1),dim=-1))
    monkeypatch.setitem(sys.modules,'isaaclab.utils.math',types.SimpleNamespace(quat_apply_inverse=inverse))
    origin=torch.tensor([[[1.,2.,3.]]]);com=origin+torch.tensor([0.,0.,.1])
    omega=torch.tensor([[[0.,0.,2.]]]);lin=torch.tensor([[[.3,.4,.2]]])
    bp=origin[:,0]+torch.tensor([1.2,0.,0.]);bv=lin[:,0]+torch.cross(omega[:,0],bp-com[:,0],dim=-1)
    data=types.SimpleNamespace(body_pos_w=origin,body_com_pos_w=com,body_lin_vel_w=lin,
        body_com_lin_vel_w=lin,body_ang_vel_w=omega,body_quat_w=torch.tensor([[[1.,0.,0.,0.]]]))
    box=types.SimpleNamespace(root_pos_w=bp,root_lin_vel_w=bv)
    env=types.SimpleNamespace(scene={'robot':types.SimpleNamespace(body_names=['torso_link'],data=data),
                                   'balloon':types.SimpleNamespace(data=box)})
    p,v=adapter.torso_relative_kinematics(env)
    assert torch.allclose(p,torch.tensor([[1.2,0.,0.]]),atol=1e-6)
    assert torch.allclose(v,torch.zeros_like(v),atol=1e-6)
    def rot(t):return torch.stack((-t[...,1],t[...,0],t[...,2]),-1)
    offset=torch.tensor([2.,-1.,.5])
    data.body_pos_w=rot(origin)+offset;data.body_com_pos_w=rot(com)+offset
    data.body_com_lin_vel_w=data.body_lin_vel_w=rot(lin)
    data.body_quat_w[:]=torch.tensor([2**-.5,0.,0.,2**-.5])
    box.root_pos_w=rot(bp)+offset;box.root_lin_vel_w=rot(bv)
    p2,v2=adapter.torso_relative_kinematics(env)
    assert torch.allclose(p,p2,atol=1e-6) and torch.allclose(v,v2,atol=1e-6)


def test_existing_throw_hook_routes_to_escape_mode(adapter,monkeypatch):
    old_core=load('_escape_legacy_core_test',ENVS/'impact_buffer_core.py')
    monkeypatch.setitem(sys.modules,'reflex.envs.impact_buffer_core',old_core)
    monkeypatch.setitem(sys.modules,'reflex.envs.escape_penalty',adapter)
    old=load('_escape_legacy_adapter_test',ENVS/'impact_buffer.py')
    monkeypatch.setenv('BOX_ESCAPE_PENALTY','1')
    monkeypatch.setenv('BOX_IMPACT_BUFFER','0')
    class Scene(dict):
        @property
        def sensors(self):return self
    env=types.SimpleNamespace(num_envs=2,device='cpu',scene=Scene())
    old.impact_throw_reset(env,torch.tensor([0]),hold_init=True)
    assert env._escape_state.hold_init[0] and not hasattr(env,'_impact_buffer_state')
    env._escape_state.escaped[0]=True
    old.impact_throw_reset(env,torch.tensor([0]))
    assert not env._escape_state.escaped[0] and not env._escape_state.hold_init[0]


def test_termination_before_launch_is_not_hidden_by_flight_denominator():
    s=State(2,'cpu');s.update(**sample(launched=torch.zeros(2,dtype=torch.bool)))
    s.throw_reset(torch.tensor([0,1]),terminated=torch.ones(2,dtype=torch.bool))
    assert s.counts['throws']==0 and s.counts['escapes']==0
    assert s.counts['terminated_without_recorded_launch']==2
