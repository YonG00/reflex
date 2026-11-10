"""Deterministic CPU impact short tests; Isaac interfaces are explicitly mocked."""
import ast
import importlib.util
from pathlib import Path
import sys
import types

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / 'source/reflex/reflex/envs'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


core = load('_impact_core_test', ENV / 'impact_buffer_core.py')
Cfg, State = core.ImpactBufferConfig, core.ImpactBufferState


def inputs(n=3, **changes):
    x = dict(step=0, dt=.02, launched=torch.ones(n, dtype=torch.bool),
        contact=torch.zeros(n, dtype=torch.bool), position_b=torch.tensor([[.5, 0., 0.]]).repeat(n, 1),
        velocity_b=torch.zeros(n, 3), gap=torch.full((n,), .2), approach=torch.full((n,), 4.),
        arm_speed=torch.full((n,), 4.), strict_now=torch.zeros(n, dtype=torch.bool), cfg=Cfg())
    x.update(changes)
    return x


def test_pre_gates_proximity_and_speed_monotonic():
    s = State(3, 'cpu')
    r = s.update(**inputs(gap=torch.tensor([.1, .5, .9])))
    assert r['total'][0] < r['total'][1] < 0
    assert r['total'][2] == 0
    r = s.update(**inputs(step=1, approach=torch.tensor([2., 4., 8.])))
    assert r['total'][2] < r['total'][1] < r['total'][0] < 0
    assert (s.update(**inputs(step=2, launched=torch.zeros(3, dtype=torch.bool)))['total'] == 0).all()
    assert (s.update(**inputs(step=3, approach=torch.zeros(3)))['total'] == 0).all()


def test_first_contact_once_and_uses_previous_impact_speed():
    s = State(3, 'cpu'); s.update(**inputs(arm_speed=torch.full((3,), 7.)))
    r = s.update(**inputs(step=1, contact=torch.ones(3, dtype=torch.bool), arm_speed=torch.ones(3)))
    assert r['first'].all() and (r['first_speed'] == 7).all()
    assert (r['parts'][:, 0] == 0).all() and (r['contact_age'] == 0).all()
    initial = s.first_pos.clone()
    s.update(**inputs(step=2, contact=torch.ones(3, dtype=torch.bool), position_b=torch.ones(3, 3)))
    assert torch.equal(s.first_pos, initial) and s.counts['contacts'] == 3


def test_post_ramp_window_and_nonpositive():
    s = State(3, 'cpu')
    s.update(**inputs(contact=torch.ones(3, dtype=torch.bool)))
    values=[]
    for step in range(1, 28):
        r = s.update(**inputs(step=step, velocity_b=torch.full((3,3), 4.)))
        assert torch.isfinite(r['total']).all() and (r['total'] <= 0).all()
        values.append(r['total'][0].item())
    assert values[0] > values[4] < 0
    assert values[-1] == 0


def test_free_displacement_outward_penalty_and_high_speed():
    s = State(3, 'cpu')
    s.update(**inputs(contact=torch.ones(3, dtype=torch.bool)))
    r = s.update(**inputs(step=1, position_b=torch.tensor([[.6,0,0],[.8,0,0],[1.2,0,0]])))
    assert r['parts'][0,2] == 0 and r['parts'][2,2] < r['parts'][1,2] < 0
    r = s.update(**inputs(step=2, velocity_b=torch.tensor([[8.,0,0],[16.,0,0],[24.,0,0]])))
    assert r['parts'][2,1] < r['parts'][1,1] < r['parts'][0,1]  # no speed saturation loophole


def test_partial_reset_rethrow_and_hold_init_suppression():
    s = State(3, 'cpu'); s.update(**inputs(contact=torch.ones(3, dtype=torch.bool)))
    s.reset(torch.tensor([0]), terminated=torch.tensor([True]))
    assert not s.contacted[0] and s.contacted[1:].all()
    s.reset(torch.tensor([1]), hold_init=True)
    r = s.update(**inputs(step=1, contact=torch.ones(3, dtype=torch.bool)))
    assert r['first'][0] and not r['first'][1] and r['total'][1] == 0
    s.reset(torch.tensor([1]))
    r = s.update(**inputs(step=2, contact=torch.ones(3, dtype=torch.bool)))
    assert r['first'][1] and s.counts['early_terminated'] == 1
    assert s.counts['post_window_truncated'] == 2


def test_same_step_does_not_advance_state_twice():
    s = State(3, 'cpu'); x=inputs(contact=torch.ones(3, dtype=torch.bool))
    a=s.update(**x); b=s.update(**x)
    assert a is b and s.counts['throws'] == 3 and s.counts['contacts'] == 3


def test_strict_success_requires_persistence_and_resets_per_throw():
    s = State(3, 'cpu')
    for step in range(29):
        s.update(**inputs(step=step, strict_now=torch.ones(3, dtype=torch.bool)))
    assert s.counts['strict_success'] == 0
    s.update(**inputs(step=29, strict_now=torch.ones(3, dtype=torch.bool)))
    assert s.counts['strict_success'] == 3
    s.reset(torch.tensor([0,1,2]))
    s.update(**inputs(step=30, strict_now=torch.ones(3, dtype=torch.bool)))
    assert not s.strict.any()


def test_miss_and_early_end_are_logged_not_rewarded():
    s=State(3,'cpu'); s.update(**inputs(approach=torch.zeros(3)))
    s.reset(torch.tensor([0,1,2]), terminated=torch.tensor([True,False,True]))
    assert s.counts['missed'] == 3 and s.counts['near_missed'] == 3
    assert s.counts['early_terminated'] == 2


def test_config_validation():
    for change in ({'w_pre':-1}, {'window':0}, {'contact_force':0}, {'ramp':float('nan')}):
        with pytest.raises(ValueError): Cfg(**change)


@pytest.fixture
def adapter(monkeypatch):
    # Do not import reflex.envs.__init__ (requires a live Isaac application).
    monkeypatch.setitem(sys.modules, 'reflex.envs.impact_buffer_core', core)
    return load('_impact_adapter_test', ENV / 'impact_buffer.py')


def qinverse(q,v):
    # Unit quaternion inverse rotation, independent reference for the adapter test.
    xyz=-q[:,1:]; w=q[:,:1]
    return v+2*(w*torch.cross(xyz,v,dim=-1)+torch.cross(xyz,torch.cross(xyz,v,dim=-1),dim=-1))


def test_torso_rotation_and_rigid_motion_invariance(adapter,monkeypatch):
    monkeypatch.setitem(sys.modules,'isaaclab.utils.math',types.SimpleNamespace(quat_apply_inverse=qinverse))
    names=['left_elbow_link','right_elbow_link','left_wrist_yaw_link','right_wrist_yaw_link','torso_link']
    pos=torch.tensor([[[.2,.2,0],[.2,-.2,0],[.5,.2,0],[.5,-.2,0],[0.,0,0]]])
    q=torch.tensor([[[1.,0,0,0]]]).repeat(1,5,1)
    omega=torch.tensor([[[0.,0.,2.]]]).repeat(1,5,1)
    translation=torch.tensor([[[1.,2.,3.]]])
    lin=translation+torch.cross(omega,pos,dim=-1)
    data=types.SimpleNamespace(body_pos_w=pos,body_quat_w=q,body_lin_vel_w=lin,body_ang_vel_w=omega)
    bp=torch.tensor([[.6,0,0.]])
    bv=translation[:,0]+torch.cross(omega[:,0],bp,dim=-1)
    bd=types.SimpleNamespace(root_pos_w=bp,root_lin_vel_w=bv,root_quat_w=q[:,0])
    env=types.SimpleNamespace(scene={'robot':types.SimpleNamespace(body_names=names,data=data),
        'balloon':types.SimpleNamespace(data=bd)},num_envs=1,_box_half_extents=torch.ones(1,3)*.1)
    first=adapter._region_kinematics(env)
    assert torch.allclose(first[1],torch.zeros(1,3),atol=1e-6)
    offset=torch.tensor([3.,-2.,1.]); data.body_pos_w=pos+offset; bd.root_pos_w=bp+offset
    second=adapter._region_kinematics(env)
    for a,b in zip(first,second): assert torch.allclose(a,b,atol=2e-6)
    # Rotate all world geometry/velocities by 90 degrees about z.
    def rot(v): return torch.stack((-v[...,1],v[...,0],v[...,2]),-1)
    data.body_pos_w=rot(data.body_pos_w); data.body_lin_vel_w=rot(lin)
    data.body_quat_w[:]=torch.tensor([2**-.5,0,0,2**-.5]); bd.root_quat_w=data.body_quat_w[:,0]
    bd.root_pos_w=rot(bd.root_pos_w); bd.root_lin_vel_w=rot(bv)
    third=adapter._region_kinematics(env)
    for a,b in zip(first,third): assert torch.allclose(a,b,atol=2e-6)


def test_filtered_sensor_history_keeps_transient_hit_and_clears(monkeypatch):
    class Base:
        def _update_buffers_impl(self,ids): pass
        def reset(self,ids): pass
    monkeypatch.setitem(sys.modules,'isaaclab.sensors',types.SimpleNamespace(ContactSensor=Base))
    module=load('_impact_sensor_test',ENV/'impact_buffer_sensor.py')
    sensor=module.ImpactPairContactSensor()
    sensor.cfg=types.SimpleNamespace(history_length=4)
    sensor._data=types.SimpleNamespace(force_matrix_w=torch.zeros(2,1,8,3))
    sensor._data.force_matrix_w[0,0,2,0]=5
    sensor._update_buffers_impl(slice(None))
    sensor._data.force_matrix_w.zero_()
    for _ in range(3): sensor._update_buffers_impl(slice(None))
    assert sensor.pair_force_history[0].abs().max()==5
    assert sensor.pair_force_history[1].abs().max()==0
    sensor.clear_history(torch.tensor([0]))
    assert sensor.pair_force_history.abs().max()==0


def test_only_arm_filters_and_no_existing_event_reordering():
    source=(ENV/'impact_buffer_config.py').read_text()
    assert 'cfg.events' not in source and 'cfg.rewards.impact_buffer' in source
    tree=ast.parse((ENV/'throw.py').read_text())
    for name in ('reset_box_throw','hold_init_apply'):
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        assert sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='impact_throw_reset'
                   for n in ast.walk(fn))==1
    assert 'torso' not in ARM_NAMES


ARM_NAMES=tuple(f'{s}_{p}_link' for s in ('left','right') for p in
                ('elbow','wrist_roll','wrist_pitch','wrist_yaw'))


def test_config_one_negative_reward_box_arm_filters_and_unchanged_events(adapter,monkeypatch):
    class Term:
        def __init__(self,**kw): self.__dict__.update(kw)
    monkeypatch.setitem(sys.modules,'isaaclab.managers',types.SimpleNamespace(RewardTermCfg=Term))
    monkeypatch.setitem(sys.modules,'isaaclab.sensors',types.SimpleNamespace(ContactSensorCfg=Term))
    monkeypatch.setitem(sys.modules,'reflex.envs.impact_buffer',adapter)
    monkeypatch.setitem(sys.modules,'reflex.envs.impact_buffer_sensor',types.SimpleNamespace(ImpactPairContactSensor=object))
    module=load('_impact_config_test',ENV/'impact_buffer_config.py')
    events=types.SimpleNamespace(reset='reset',hover='hover',hold='hold')
    scene=types.SimpleNamespace(balloon=types.SimpleNamespace(prim_path='{ENV_REGEX_NS}/Balloon',
        spawn=types.SimpleNamespace()),robot=types.SimpleNamespace(prim_path='{ENV_REGEX_NS}/Robot'))
    cfg=types.SimpleNamespace(scene=scene,events=events,rewards=types.SimpleNamespace(
        hold_still='original_hold',box_on_bed='original_bed'),decimation=4)
    module.configure_impact_buffer(cfg)
    filters=scene.impact_box_arm.filter_prim_paths_expr
    assert len(filters)==8 and all(x.endswith(ARM_NAMES) for x in filters)
    assert not any('torso' in x or 'ground' in x for x in filters)
    assert cfg.events is events and list(vars(events))==['reset','hover','hold']
    assert cfg.rewards.hold_still=='original_hold' and cfg.rewards.box_on_bed=='original_bed'
    assert cfg.rewards.impact_buffer.weight==1 and cfg.rewards.impact_buffer.params['w_pre']==.2


def test_adapter_short_rollout_logs_and_lifecycle(adapter,monkeypatch,tmp_path):
    # 4 controlled trajectories: buffered catch, rebound, chest/ground-only
    # contact (absent from pair forces), and HOLD_INIT. This is NOT PhysX.
    n=4
    sensor=types.SimpleNamespace(data=object(),pair_force_history=torch.zeros(n,4,1,8,3))
    sensor.clear_history=lambda ids: sensor.pair_force_history.__setitem__(ids,0)
    class Scene(dict):
        @property
        def sensors(self): return self
    env=types.SimpleNamespace(num_envs=n,device='cpu',common_step_counter=0,step_dt=.02,
        scene=Scene(impact_box_arm=sensor),_box_launched=torch.ones(n,dtype=torch.bool),
        _impact_hold_init=torch.tensor([False,False,False,True]),extras={},
        episode_length_buf=torch.zeros(n,dtype=torch.long),reset_terminated=torch.zeros(n,dtype=torch.bool))
    strict=torch.zeros(n,dtype=torch.bool)
    monkeypatch.setitem(sys.modules,'reflex.envs.base.chest_v5_terms',types.SimpleNamespace(
        compute_strict_components=lambda env:{'strict_all':strict}))
    monkeypatch.setenv('BOX_IMPACT_BUFFER','1')
    def motion(env):
        step=env.common_step_counter
        p=torch.tensor([[.5,0,0.]]).repeat(n,1)
        v=torch.zeros(n,3)
        if step>=5:
            p[1,0]+=.04*(step-5);v[1,0]=2.
        return p,v,torch.full((n,),.1),torch.full((n,),6.),torch.full((n,),6.)
    monkeypatch.setattr(adapter,'_region_kinematics',motion)
    log=tmp_path/'impact.jsonl'
    terms=[]
    for step in range(40):
        env.common_step_counter=step;env.episode_length_buf[:]=step
        sensor.pair_force_history.zero_()
        if step==5: sensor.pair_force_history[:2,2,0,0,0]=5  # transient substep
        strict[0]=step>=6
        value=adapter.impact_buffer_penalty(env,log_every=1,log_path=str(log))
        terms.append(value.clone())
    assert torch.isfinite(torch.stack(terms)).all() and (torch.stack(terms)<=0).all()
    assert all(x[3]==0 for x in terms)
    assert env._impact_buffer_state.counts['contacts']==2
    assert env._impact_buffer_state.counts['throws']==3
    assert env._impact_buffer_state.counts['strict_success']==1
    assert terms[15][1]<terms[15][0]
    adapter.impact_throw_reset(env,torch.tensor([0,1,2]))
    assert not env._impact_buffer_state.contacted[:3].any()
    assert sensor.pair_force_history.abs().max()==0
    import json
    rows=[json.loads(s) for s in log.read_text().splitlines()]
    snapshots=[r for r in rows if 'penalty_total' in r]
    assert len(snapshots)==40 and sum(r.get('event')=='throw_end' for r in rows)==3
    assert snapshots[-1]['first_arm_contact_rate']==pytest.approx(2/3)
    assert snapshots[-1]['strict_catch_rate']==pytest.approx(1/3)
    assert snapshots[-1]['max_escape_distance']>0
    for r in snapshots:
        assert r['penalty_total']==pytest.approx(r['penalty_pre']+r['penalty_damp']+r['penalty_escape'])


def test_partial_reset_same_step_never_double_advances_other_envs():
    s=State(3,'cpu'); s.update(**inputs(contact=torch.ones(3,dtype=torch.bool)))
    x=inputs(step=1,velocity_b=torch.ones(3,3));s.update(**x)
    age=s.age[1:].clone()
    s.reset(torch.tensor([0]))
    result=s.update(**x)
    assert torch.equal(s.age[1:],age) and result['total'][0]==0
    assert not s.contacted[0]


def test_post_window_inclusive_float_boundary():
    s=State(3,'cpu');s.update(**inputs(contact=torch.ones(3,dtype=torch.bool)))
    for step in range(1,26):
        r=s.update(**inputs(step=step,velocity_b=torch.ones(3,3)))
    assert r['post_gate'].all()
    r=s.update(**inputs(step=26,velocity_b=torch.ones(3,3)))
    assert not r['post_gate'].any()


def test_pre20x_only_scales_pre_contact_term():
    baseline=State(3,'cpu'); variant=State(3,'cpu')
    a=baseline.update(**inputs()); b=variant.update(**inputs(cfg=Cfg(w_pre=4.0)))
    assert torch.allclose(b['parts'][:,0],a['parts'][:,0]*20)
    for state,cfg in ((baseline,Cfg()),(variant,Cfg(w_pre=4.0))):
        state.update(**inputs(step=1,contact=torch.ones(3,dtype=torch.bool),cfg=cfg))
    common=dict(step=2,velocity_b=torch.ones(3,3)*3,position_b=torch.ones(3,3))
    a=baseline.update(**inputs(**common)); b=variant.update(**inputs(**common,cfg=Cfg(w_pre=4.0)))
    assert torch.equal(a['parts'][:,1:],b['parts'][:,1:])
