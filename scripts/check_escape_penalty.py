#!/usr/bin/env python3
"""Prescribed CPU trajectories only: no physics engine, PPO or paid GPU."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import torch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',default='outputs/escape_penalty_cpu_short')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('escape_short_core',root/'source/reflex/reflex/envs/escape_penalty_core.py')
    core=importlib.util.module_from_spec(spec);sys.modules[spec.name]=core;spec.loader.exec_module(core)
    cfg=core.EscapeConfig();state=core.EscapeState(8,'cpu');state.throw_reset(torch.tensor([7]),hold_init=True)
    names=['no_contact','hover_contact','brief_rebound','severe_escape','escape_then_recatch',
           'tangential_fast','early_fall','hold_init']
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    total=torch.zeros(8)
    with (out/'samples.jsonl').open('w') as stream:
        for step in range(100):
            launched=torch.ones(8,dtype=torch.bool);launched[1]=False
            contact=torch.zeros(8,dtype=torch.bool)
            if step==0:contact[1:]=True
            p=torch.tensor([[.6,0.,0.]]).repeat(8,1);v=torch.zeros(8,3)
            if step>=10:
                p[[0,3,4,6],0]=1.4;v[[0,3,4,6],0]=2.
                p[5,0]=1.4;v[5,1]=5.
                if step<14:p[2,0]=1.4;v[2,0]=2.
            if step>=40:p[4,0]=.6;v[4]=0
            if step==13:
                state.throw_reset(torch.tensor([6]),terminated=torch.tensor([True]))
            if step>=13:launched[6]=False
            strict=torch.zeros(8,dtype=torch.bool)
            if step>=40:strict[[2,4]]=True
            r=state.update(step=step,dt=.02,launched=launched,contact=contact,
                position_b=p,velocity_b=v,strict_now=strict,cfg=cfg)
            # Simulate the adapter's 1/dt then RewardManager's *dt.
            applied=(r['event']/.02)*.02;total+=applied
            stream.write(json.dumps(dict(step=step,applied_penalty=applied.tolist(),
                escaped=state.escaped.tolist(),contacted=state.contacted.tolist(),
                radius=r['radius'].tolist(),radial_speed=r['radial_speed'].tolist(),
                dwell=r['dwell'].tolist()))+'\n')
    expected=torch.tensor([0.,0.,0.,-100.,-100.,0.,0.,0.])
    assert torch.equal(total,expected) and state.escaped[4]
    counts={k:float(v) for k,v in state.counts.items()}
    report=dict(scope='synthetic deterministic CPU test; not physical or learned performance',
        steps=100,dt=.02,config=vars(cfg),case_total_penalties=dict(zip(names,total.tolist())),
        escaped_before_episode_reset=state.escaped.tolist(),counts=counts,
        escape_rate=counts['escapes']/counts['throws'],
        contact_rate=counts['contacts']/counts['throws'],
        catch_success_rate=counts['catches']/counts['throws'],
        known_risk='No-contact avoidance and termination before persistence can avoid -100; original mechanisms are unchanged.')
    state.episode_reset(torch.arange(8))
    assert not state.escaped.any() and not state.contacted.any()
    report['episode_reset_cleared_all_latches']=True
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
