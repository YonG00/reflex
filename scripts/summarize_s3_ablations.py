"""First-throw-only comparison; preserve raw rethrow counts and fail on missing seeds."""
import json
import sys
from pathlib import Path
root=Path(sys.argv[1]);result={}
for group in ('S3','A','B','C'):
    summaries=[]
    for seed in (8,9):
        data=json.loads((root/f'{group}_s{seed}.json').read_text())
        rows=data['throw_log'];first=[r for r in rows if r['idx']==1]
        if len(first)!=320 or len({r['env'] for r in first})!=320:raise RuntimeError(f'{group}/{seed}: incomplete first throws')
        for r in first:
            for key in ('physical_hold_success','caught_loose','physical_contact','control_seconds'):
                if key not in r:raise RuntimeError(f'missing shared metric {key}')
        n=len(first)
        def rate(key):return sum(bool(r[key]) for r in first)/n
        def mean(key):return sum(r.get(key,0.) for r in first)/n
        def rom(key):
            values=[b-a for r in first for a,b in zip(r.get(key+'_min',[]),r.get(key+'_max',[]))]
            return sum(values)/len(values) if values else None
        summaries.append(dict(seed=seed,first_throws=n,rethrows=len(rows)-n,
            strict=rate('caught'),loose=rate('caught_loose'),physical_hold_success=rate('physical_hold_success'),
            contact_rate=rate('physical_contact'),fall=rate('fell'),ground=rate('physical_ground'),escape=rate('physical_escape'),
            mean_control_seconds=mean('control_seconds'),
            mean_arm_contact_seconds=mean('arm_contact_seconds'),
            mean_bilateral_arm_contact_seconds=mean('bilateral_arm_contact_seconds'),
            mean_head_shoulder_contact_seconds=mean('head_shoulder_contact_seconds'),
            mean_max_back_lean_degrees=mean('max_back_lean_degrees'),
            mean_shoulder_range=rom('shoulder'),mean_elbow_range=rom('elbow'),
            mean_longest_control_seconds=sum(r['longest_control_seconds'] for r in first)/n))
    result[group]=summaries
print(json.dumps(result,indent=2))
