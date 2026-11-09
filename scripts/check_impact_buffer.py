#!/usr/bin/env python3
"""Fast deterministic CPU contract exercise, NOT a physics/training experiment."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import torch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',default='outputs/impact_buffer_cpu_short')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('impact_short_core',root/'source/reflex/reflex/envs/impact_buffer_core.py')
    core=importlib.util.module_from_spec(spec);sys.modules[spec.name]=core;spec.loader.exec_module(core)
    cfg=core.ImpactBufferConfig();state=core.ImpactBufferState(6,'cpu')
    cases=['buffered_catch','fast_rebound','avoid_miss','early_termination','hold_init','ground_or_chest_only']
    state.reset(torch.tensor([4]),hold_init=True)
    totals=torch.zeros(6,3); pre_sum=post_sum=0.; pre_n=post_n=0
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    with (out/'samples.jsonl').open('w') as stream:
        for step in range(100):
            launched=torch.ones(6,dtype=torch.bool)
            if step==20: state.reset(torch.tensor([3]),terminated=torch.tensor([True]))
            if step>=20: launched[3]=False
            contact=torch.zeros(6,dtype=torch.bool)
            if step==30: contact[:2]=True
            position=torch.tensor([[.5,0.,0.]]).repeat(6,1)
            velocity=torch.zeros(6,3)
            if step>30:
                position[1,0]+=(step-30)*.05;velocity[1,0]=6.
            approach=torch.full((6,),6.);approach[2]=0
            strict=torch.zeros(6,dtype=torch.bool);strict[0]=step>=31
            r=state.update(step=step,dt=.02,launched=launched,contact=contact,
                position_b=position,velocity_b=velocity,gap=torch.full((6,),.1),
                approach=approach,arm_speed=torch.full((6,),6.),strict_now=strict,cfg=cfg)
            totals+=r['parts']
            pre_sum+=float((r['approach']*r['pre_gate']).sum());pre_n+=int(r['pre_gate'].sum())
            post_sum+=float((r['post_speed']*r['post_gate']).sum());post_n+=int(r['post_gate'].sum())
            stream.write(json.dumps(dict(step=step,penalty_total=r['total'].tolist(),
                penalty_parts=r['parts'].tolist(),pre_approach_speed=r['approach'].tolist(),
                first_contact_speed=state.first_speed.tolist(),post_speed=r['post_speed'].tolist(),
                max_escape=state.max_escape.tolist(),contacted=state.contacted.tolist()))+'\n')
    state.reset(torch.tensor([0,1,2,4,5]))
    counts={k:float(v) for k,v in state.counts.items()}
    summary=dict(scope='deterministic synthetic CPU trajectories; no PhysX or PPO learning',
        steps=100,dt=.02,counts=counts,first_arm_contact_rate=counts['contacts']/counts['throws'],
        strict_catch_rate=counts['strict_success']/counts['throws'],
        pre_approach_speed_mean=pre_sum/max(pre_n,1),post_speed_mean=post_sum/max(post_n,1),
        maximum_escape_distance=float(state.max_escape_seen),
        penalty_component_means=(totals/100).mean(0).tolist(),
        penalty_total_mean=float((totals/100).sum(-1).mean()),
        case_mean_penalties={name:float(totals[i].sum()/100) for i,name in enumerate(cases)},
        limitation='An avoided/missed throw has lower impact cost. Existing catch rewards remain necessary; GPU policy comparison is pending.')
    assert counts['contacts']==2 and counts['throws']==5 and counts['strict_success']==1
    assert counts['early_terminated']==1 and float(totals[4].sum())==0
    assert float(totals[1].sum())<float(totals[0].sum())<0
    assert torch.isfinite(totals).all() and (totals<=0).all()
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
