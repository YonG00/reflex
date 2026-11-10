"""CPU reward-trigger tests. Physics integration is verified separately on GPU."""
import ast
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

core=load('_buffer_core_test',ENVS/'buffer_control_core.py')
Cfg,State=core.BufferConfig,core.BufferState


def sample(n=2,**kw):
    d=dict(step=0,dt=.02,launched=torch.ones(n,dtype=torch.bool),contact=torch.zeros(n,dtype=torch.bool),
        ground_contact=torch.zeros(n,dtype=torch.bool),box_bottom=torch.ones(n),safe=torch.ones(n,dtype=torch.bool),
        radius=torch.full((n,),.5),radial_speed=torch.zeros(n),relative_speed=torch.full((n,),1.5),
        incoming=torch.ones(n,dtype=torch.bool),strict=torch.zeros(n,dtype=torch.bool),cfg=Cfg())
    d.update(kw);return d


def test_intake_is_windowed_and_budget_cannot_be_refilled_by_dithering():
    s=State(2,'cpu');paid=0
    for step in range(100):
        r=s.update(**sample(step=step,incoming=torch.full((2,),step%2==0)))
        paid+=int(r['intake_gate'][0])
    assert paid==30 and torch.allclose(s.intake_used,torch.full((2,),.6),atol=1e-6)
    assert s.counts['throws']==2


@pytest.mark.parametrize('kw',[{'launched':torch.zeros(2,dtype=torch.bool)},
    {'incoming':torch.zeros(2,dtype=torch.bool)},{'safe':torch.zeros(2,dtype=torch.bool)},
    {'box_bottom':torch.zeros(2)}])
def test_no_idle_hover_lying_or_floor_intake_farming(kw):
    r=State(2,'cpu').update(**sample(**kw));assert not r['intake_gate'].any()


def test_retention_requires_real_recent_contact_but_not_instant_stillness():
    s=State(2,'cpu');assert s.update(**sample())['retain'].sum()==0
    r=s.update(**sample(step=1,contact=torch.ones(2,dtype=torch.bool),relative_speed=torch.full((2,),3.)))
    assert r['control'].all()
    r=s.update(**sample(step=2,relative_speed=torch.full((2,),3.)))
    assert (r['retain']>0).all()
    for step in range(3,19):r=s.update(**sample(step=step))
    assert not r['control'].any()  # detached >0.3 s


@pytest.mark.parametrize('kw',[{'safe':torch.zeros(2,dtype=torch.bool)},
    {'ground_contact':torch.ones(2,dtype=torch.bool)}, {'box_bottom':torch.zeros(2)},
    {'radius':torch.full((2,),1.)},{'radial_speed':torch.full((2,),3.)}])
def test_no_invalid_control_reward(kw):
    s=State(2,'cpu');s.update(**sample(contact=torch.ones(2,dtype=torch.bool)))
    r=s.update(**sample(step=1,**kw));assert r['retain'].sum()==0


def test_ground_or_gross_escape_latches_until_rethrow():
    s=State(2,'cpu');s.update(**sample(contact=torch.ones(2,dtype=torch.bool)))
    s.update(**sample(step=1,ground_contact=torch.tensor([True,False]),radius=torch.tensor([.5,1.4])))
    assert s.update(**sample(step=2,contact=torch.ones(2,dtype=torch.bool)))['retain'].sum()==0
    s.reset(torch.tensor([0,1]));s.update(**sample(step=3,contact=torch.ones(2,dtype=torch.bool)))
    assert (s.update(**sample(step=4))['retain']>0).all()


def test_control_window_is_bounded_even_with_constant_contact():
    s=State(2,'cpu');returns=[]
    for step in range(150):returns.append(s.update(**sample(step=step,contact=torch.ones(2,dtype=torch.bool)))['retain'].clone())
    assert torch.stack(returns).min()>=0 and torch.stack(returns).max()<=1
    assert s.control_time.max()<=2.00001 and torch.stack(returns)[110:].sum()==0


def test_settling_gradually_restores_posture_without_jitter_exploit():
    s=State(2,'cpu')
    for step in range(30):r=s.update(**sample(step=step,contact=torch.ones(2,dtype=torch.bool),relative_speed=torch.zeros(2)))
    assert s.settled.all() and (r['restore']>0).all() and (r['restore']<1).all()
    previous=r['restore'].clone()
    r=s.update(**sample(step=30,relative_speed=torch.full((2,),4.)))
    assert (r['restore']>=previous).all()  # cannot shake box to reset posture recovery
    for step in range(31,90):r=s.update(**sample(step=step,contact=torch.ones(2,dtype=torch.bool),relative_speed=torch.zeros(2)))
    assert (r['restore']==1).all() and r['relax'].sum()==0 and r['retain'].sum()==0


def test_hold_init_partial_reset_same_step_and_catch_statistics():
    s=State(2,'cpu');s.reset(torch.tensor([1]),hold_init=True)
    for i in range(35):r=s.update(**sample(step=i,contact=torch.ones(2,dtype=torch.bool),strict=torch.ones(2,dtype=torch.bool)))
    assert s.counts['contacts']==1 and s.counts['catches']==1 and r['retain'][1]==0
    t=s.control_time[0].clone();s.reset(torch.tensor([1]))
    s.update(**sample(step=34));assert s.control_time[0]==t
    s.update(**sample(step=35,contact=torch.ones(2,dtype=torch.bool)))
    assert s.counts['contacts']==2


@pytest.fixture
def modules(monkeypatch,tmp_path):
    monkeypatch.setitem(sys.modules,'reflex.envs.buffer_control_core',core)
    mod=load('_buffer_adapter_test',ENVS/'buffer_control.py')
    monkeypatch.setitem(sys.modules,'reflex.envs.buffer_control',mod)
    monkeypatch.setenv('BOX_BUFFER_STAGE','3');monkeypatch.setenv('BOX_BUFFER_LOG_EVERY','1')
    monkeypatch.setenv('BOX_BUFFER_LOG_PATH',str(tmp_path/'buffer.jsonl'))
    # Execute literal original reward functions, without importing Isaac's app.
    source=ast.parse((ENVS/'throw.py').read_text())
    names={'compliant_intake_reward','torso_asym_penalty','upright_hold_reward','_posture_gbx','_upright_gate_factor'}
    code=ast.Module(body=[n for n in source.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[])
    class Entity:
        def __init__(self,name):self.name=name
    old=types.ModuleType('reflex.envs.throw');old.__dict__.update(torch=torch,os=__import__('os'),SceneEntityCfg=Entity,
        INTAKE_GATE_DIST=.9,INTAKE_GATE_SPEED=1.,YIELD_V_CAP=.5,_get_posture_ids=lambda env:{'torso':0})
    exec(compile(code,str(ENVS/'throw.py'),'exec'),old.__dict__)
    monkeypatch.setitem(sys.modules,'reflex.envs.throw',old)
    monkeypatch.setitem(sys.modules,'isaaclab.managers',types.SimpleNamespace(SceneEntityCfg=Entity))
    return mod,old


def test_normal_backward_lean_has_no_extra_penalty_but_forward_side_are_kept(modules,monkeypatch):
    mod,old=modules
    cfg=types.SimpleNamespace(body_quat_w=torch.tensor([[[1.,0,0,0.]]]).repeat(2,1,1))
    robot=types.SimpleNamespace(data=cfg)
    env=types.SimpleNamespace(scene={'robot':robot})
    ctx=dict(restore=torch.zeros(2),safe=torch.ones(2,dtype=torch.bool),relax=torch.ones(2))
    monkeypatch.setattr(mod,'context',lambda env:ctx)
    q=torch.tensor([torch.cos(torch.tensor(-.5)),0.,torch.sin(torch.tensor(-.5)),0.])
    cfg.body_quat_w[:]=q
    assert old.torso_asym_penalty(env,back_max=.31).min()>0
    assert mod.buffer_torso_asym(env).sum()==0
    gate=mod.phase_upright_gate(env,torch.full((2,),.1));assert (gate==1).all()
    cfg.body_quat_w[:]=torch.tensor([torch.cos(torch.tensor(.3)),0.,torch.sin(torch.tensor(.3)),0.])
    assert mod.buffer_torso_asym(env).min()>0
    cfg.body_quat_w[:]=torch.tensor([torch.cos(torch.tensor(.3)),torch.sin(torch.tensor(.3)),0.,0.])
    assert mod.buffer_torso_asym(env).min()>0
    ctx['restore']=torch.ones(2)
    cfg.body_quat_w[:]=q
    assert mod.buffer_torso_asym(env).min()>0


def test_original_intake_formula_only_pays_directional_window(modules):
    _,old=modules
    box=types.SimpleNamespace(data=types.SimpleNamespace(root_pos_w=torch.tensor([[.7,0.,1.]]),root_lin_vel_w=torch.tensor([[-3.,0,0.]])))
    robot=types.SimpleNamespace(data=types.SimpleNamespace(root_pos_w=torch.tensor([[0.,0.,.8]]),root_lin_vel_w=torch.tensor([[-.2,0,-.2]])))
    env=types.SimpleNamespace(scene={'robot':robot,'balloon':box})
    assert torch.allclose(old.compliant_intake_reward(env),torch.tensor([.4]))


def test_config_stages_are_isolated_and_escape_absent(modules,monkeypatch):
    mod,_=modules
    class Term:
        def __init__(self,**kw):self.__dict__.update(kw)
    monkeypatch.setitem(sys.modules,'isaaclab.managers',types.SimpleNamespace(RewardTermCfg=Term))
    monkeypatch.setitem(sys.modules,'isaaclab.sensors',types.SimpleNamespace(ContactSensorCfg=Term))
    monkeypatch.setitem(sys.modules,'isaaclab.utils',types.SimpleNamespace(configclass=lambda cls:cls))
    monkeypatch.setitem(sys.modules,'reflex.envs',types.SimpleNamespace(buffer_control=mod))
    monkeypatch.setitem(sys.modules,'reflex.envs.escape_penalty_sensor',types.SimpleNamespace(EscapeRobotContactSensor=object))
    configure=load('_buffer_config_test',ENVS/'buffer_control_config.py').configure_buffer_control
    for stage in (1,2,3):
        monkeypatch.setenv('BOX_BUFFER_STAGE',str(stage))
        cfg=types.SimpleNamespace(scene=types.SimpleNamespace(balloon=types.SimpleNamespace(prim_path='Balloon',spawn=types.SimpleNamespace()),
            robot=types.SimpleNamespace(prim_path='Robot'),terrain=types.SimpleNamespace(prim_path='/World/ground')),
            rewards=types.SimpleNamespace(torso_asym=Term(func='old',weight=-.5,params={'back_max':.31}),
                upright_hold=Term(func='old',weight=1.5),compliant_intake=Term(func='old',weight=0),
                escape_penalty='old',impact_buffer='old',hold_still='original'),
            events=types.SimpleNamespace(hover='hover',launch='launch'),decimation=4)
        configure(cfg)
        assert cfg.rewards.escape_penalty is None and cfg.rewards.impact_buffer is None
        assert cfg.rewards.hold_still=='original' and list(vars(cfg.events))==['hover','launch']
        assert cfg.rewards.compliant_intake.weight==(.75 if stage>=2 else 0)
        assert hasattr(cfg.rewards,'retain_reward')==(stage==3)
        assert cfg.rewards.torso_asym.weight==-.5 and cfg.rewards.upright_hold.weight==1.5


def test_adapter_real_formulas_intake_retain_floor_and_posture_safety(modules,monkeypatch):
    mod,_=modules
    def inverse(q,v):
        xyz=-q[:,1:];w=q[:,:1]
        return v+2*(w*torch.cross(xyz,v,dim=-1)+torch.cross(xyz,torch.cross(xyz,v,dim=-1),dim=-1))
    monkeypatch.setitem(sys.modules,'isaaclab.utils.math',types.SimpleNamespace(quat_apply_inverse=inverse))
    esc_core=load('_buffer_escape_core',ENVS/'escape_penalty_core.py')
    monkeypatch.setitem(sys.modules,'reflex.envs.escape_penalty_core',esc_core)
    esc=load('_buffer_escape_math',ENVS/'escape_penalty.py')
    monkeypatch.setitem(sys.modules,'reflex.envs.escape_penalty',esc)
    monkeypatch.setitem(sys.modules,'reflex.envs.base.chest_v5_terms',types.SimpleNamespace(
        compute_strict_components=lambda env:{'strict_all':torch.zeros(2,dtype=torch.bool)}))
    class Scene(dict):
        @property
        def sensors(self):return self
    data=types.SimpleNamespace(root_pos_w=torch.tensor([[0.,0,.8],[0.,0,.8]]),
        root_lin_vel_w=torch.tensor([[-.2,0,-.2],[-.2,0,-.2]]),
        body_pos_w=torch.tensor([[[0.,0,1.1]],[[0.,0,1.1]]]),
        body_lin_vel_w=torch.zeros(2,1,3),body_ang_vel_w=torch.zeros(2,1,3),
        body_quat_w=torch.tensor([[[1.,0,0,0]],[[1.,0,0,0]]]),joint_pos=torch.ones(2,2))
    bd=types.SimpleNamespace(root_pos_w=torch.tensor([[.7,0.,1.1],[.7,0.,1.1]]),
        root_lin_vel_w=torch.tensor([[-3.,0,0],[3.,0,0]]),root_quat_w=data.body_quat_w[:,0].clone())
    hist=torch.zeros(2,4,1,2,3)
    sensor=types.SimpleNamespace(data=object(),robot_filter_count=1,pair_force_history=hist,
        clear_history=lambda ids:hist.__setitem__(ids,0))
    scene=Scene(robot=types.SimpleNamespace(data=data,body_names=['torso_link'],
        joint_names=['left_knee_joint','right_knee_joint']),balloon=types.SimpleNamespace(data=bd),
        buffer_box_contacts=sensor);scene.env_origins=torch.zeros(2,3)
    env=types.SimpleNamespace(scene=scene,num_envs=2,device='cpu',step_dt=.02,common_step_counter=0,
        _box_launched=torch.ones(2,dtype=torch.bool),_box_half_extents=torch.ones(2,3)*.1,
        extras={},episode_length_buf=torch.zeros(2,dtype=torch.long))
    r=mod.context(env)
    assert torch.allclose(r['intake'],torch.tensor([.4,0.]))
    assert mod.context(env) is r and env._buffer_state.intake_used[0]==.02
    env.common_step_counter=1;hist[0,1,0,0,0]=5
    assert mod.context(env)['control'][0]
    env.common_step_counter=2;hist.zero_()
    assert mod.retain_reward(env)[0]>0  # brief detach and high relative velocity are allowed
    env.common_step_counter=3;hist[0,0,0,1,0]=5
    assert mod.retain_reward(env)[0]==0  # independent box-ground filter
    env.common_step_counter=4;hist.zero_();data.root_pos_w[0,2]=.2
    assert mod.phase_upright_gate(env,torch.ones(2))[0]==0
    mod.buffer_throw_reset(env,torch.tensor([0]),hold_init=True)
    env.common_step_counter=5;data.root_pos_w[0,2]=.8;hist[0,0,0,0,0]=5
    assert mod.retain_reward(env)[0]==0


def test_floor_or_fallen_touch_does_not_inflate_effective_contact_rate():
    s=State(2,'cpu')
    r=s.update(**sample(contact=torch.ones(2,dtype=torch.bool),safe=torch.tensor([True,False]),
        ground_contact=torch.tensor([True,False])))
    assert s.counts['physical_contacts']==2 and s.counts['contacts']==0
    assert not s.contacted.any() and r['retain'].sum()==0
