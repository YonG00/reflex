#!/usr/bin/env python3
"""Synthetic CPU buffering trajectories, NOT robot learning or physical proof."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',default='outputs/buffer_cpu_short')
    args=p.parse_args();root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('buffer_short_core',root/'source/reflex/reflex/envs/buffer_control_core.py')
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    names=['incoming_yield_squat','idle_squat','moving_control','box_on_floor','lying_robot',
           'brief_detach','escaped_box','settling_handoff']
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);reports={}
    for stage in (1,2,3):
        s=mod.BufferState(8,'cpu');intake_total=torch.zeros(8);retain_total=torch.zeros(8)
        with (out/f'stage{stage}.jsonl').open('w') as f:
            for step in range(150):
                contact=torch.tensor([False,False,True,True,True,step==0,True,True])
                safe=torch.tensor([True,True,True,True,False,True,True,True])
                floor=torch.tensor([False,False,False,True,False,False,False,False])
                radius=torch.full((8,),.6);radius[6]=1.5
                speed=torch.full((8,),1.5);speed[7]=.1
                incoming=torch.tensor([True,False,False,False,False,False,False,False])
                strict=torch.tensor([False,False,False,False,False,False,False,True])
                r=s.update(step=step,dt=.02,launched=torch.ones(8,dtype=torch.bool),contact=contact,
                    ground_contact=floor,box_bottom=torch.ones(8),safe=safe,radius=radius,
                    radial_speed=torch.zeros(8),relative_speed=speed,incoming=incoming,strict=strict,cfg=mod.BufferConfig())
                # Prescribed original intake formula = .6*.4 + .4*.4 = .4.
                intake=.4*r['intake_gate']*(.75 if stage>=2 else 0.)
                retain=r['retain']*(.5 if stage>=3 else 0.)
                intake_total+=intake*.02;retain_total+=retain*.02
                f.write(json.dumps(dict(step=step,intake=intake.tolist(),retain=retain.tolist(),
                    control=r['control'].tolist(),posture_relax=r['relax'].tolist(),restore=r['restore'].tolist()))+'\n')
        assert intake_total[1]==0 and retain_total[3]==retain_total[4]==retain_total[6]==0
        if stage>=2:assert intake_total[0]>0
        if stage==3:assert retain_total[2]>0 and retain_total[5]>0 and retain_total[7]>0
        reports[str(stage)]={name:dict(intake_integral=float(intake_total[i]),retain_integral=float(retain_total[i]))
                             for i,name in enumerate(names)}
    summary=dict(scope=__doc__,steps=150,dt=.02,intake_weight=.75,retain_weight=.5,stages=reports,
                 limitations='No live physics, contact filtering or learned strategy is validated by these prescribed trajectories.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
