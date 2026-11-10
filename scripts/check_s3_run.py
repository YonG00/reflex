"""Post-run acceptance checks; do not treat a live or partial log as completion."""
import json
import math
import re
import sys
from pathlib import Path
text=Path(sys.argv[1]).read_text();updates=int(sys.argv[3])
iterations=[int(v) for v in re.findall(r'Learning iteration\s+(\d+)',re.sub(r'\x1b\[[0-9;]*m','',text),re.I)]
if iterations!=list(range(599,599+updates)):
    raise RuntimeError(f'expected {updates} additional updates 599..{598+updates}; got {iterations}')
rows=[json.loads(s) for s in Path(sys.argv[2]).read_text().splitlines()]
metrics=[r for r in rows if 'invalid_positive_intake' in r]
if not metrics:raise RuntimeError('missing reward diagnostics')
for r in metrics:
    if r['invalid_positive_intake'] or r['invalid_positive_retain']:raise RuntimeError(f'invalid reward {r}')
    if any(not math.isfinite(v) for v in r.values() if isinstance(v,(int,float))):raise RuntimeError('nonfinite diagnostics')
group=metrics[-1]['experiment']
if group=='B' and not any(r['intake_active'] for r in metrics):raise RuntimeError('B intake never activated')
if group=='C' and not any(r.get('flex_shoulder_scale_mean',.02)>.02 for r in metrics):raise RuntimeError('C scale never ramped')
print(json.dumps(dict(group=group,updates=len(iterations),first=iterations[0],last=iterations[-1],
    diagnostics=len(metrics),invalid_positive=0,behavior_review='pending videos and matched evaluation')))
