"""Fail closed on checkpoint schema; never pad incompatible inputs."""
import hashlib
import json
import sys
import torch
from pathlib import Path
p=Path(sys.argv[1]);c=torch.load(p,map_location='cpu',weights_only=False)
s=c['model_state_dict']
actor=[tuple(v.shape) for k,v in s.items() if k.endswith('actor.0.weight')]
critic=[tuple(v.shape) for k,v in s.items() if k.endswith('critic.0.weight')]
if not actor or not critic or actor[0][1]!=183 or critic[0][1]!=305:
    raise ValueError(f'expected actor183/critic305, got {actor}/{critic}')
if c.get('iter')!=599:raise ValueError(f'expected iteration599, got {c.get("iter")}')
if 'optimizer_state_dict' not in c:raise ValueError('native PPO resume requires optimizer state')
h=hashlib.sha256()
with p.open('rb') as f:
    for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
print(json.dumps(dict(checkpoint=str(p.resolve()),sha256=h.hexdigest(),iteration=c['iter'],actor=actor,
    critic=critic,checkpoint_keys=sorted(c),resume='native runner.load; AMP uses original repository restore behavior')))
