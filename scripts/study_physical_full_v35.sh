#!/usr/bin/env bash
# Real PhysX/Replay H8 and full Actor H32 numerical control, no formal training.
set -euo pipefail
if [[ ${1:-} == --help || $# -ne 2 ]]; then
  echo "Usage: $0 CHECKPOINT NEW_OUTPUT_DIR"
  echo "ISAACLAB_PY; REFLEX_DATA_DIR; V35_FULL_TIMEOUT=240; V35_FULL_STAGE=wm; V35_TASK_EXECUTION=eager"
  [[ ${1:-} == --help ]] && exit 0
  exit 2
fi
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
checkpoint=$1
output=$2
[[ -f "$checkpoint" && ! -e "$output" ]] || exit 2
[[ ${V35_FULL_TIMEOUT:-240} =~ ^[1-9][0-9]*$ ]] || exit 2
checkpoint=$(cd "$(dirname "$checkpoint")" && pwd)/$(basename "$checkpoint")
mkdir -p "$output"
output=$(cd "$output" && pwd)
[[ ${V35_FULL_STAGE:-wm} =~ ^(wm|actor|all)$ ]] || exit 2
[[ ${V35_TASK_EXECUTION:-eager} =~ ^(eager|compiled)$ ]] || exit 2
export V35_FULL_STAGE=${V35_FULL_STAGE:-wm}
source "$ROOT/scripts/local/env.sh"
export PYTHONPATH="$ROOT/source/reflex:$ROOT/source/whole_body_tracking${PYTHONPATH:+:$PYTHONPATH}"
export TRITON_DEFAULT_FP_FUSION=0
# Separate cache from default-FMA production runs. Caller may reuse an already
# verified FMA=0 cache; no baseline-FMA cache may be supplied here.
export TORCHINDUCTOR_CACHE_DIR=${V35_FULL_INDUCTOR_CACHE:-$output/inductor_cache}
export TRITON_CACHE_DIR=${V35_FULL_TRITON_CACHE:-$output/triton_cache}
"$ISAACLAB_PY" - "$ROOT" <<'PY'
import importlib.util, sys
from pathlib import Path
actual = Path(importlib.util.find_spec('reflex.world_model.train').origin).resolve()
assert actual == Path(sys.argv[1], 'source/reflex/reflex/world_model/train.py').resolve(), actual
print('Verified source:', actual)
PY
"$ISAACLAB_PY" - "$ROOT/configs/world_model/joint_execution_v32_verified_actor_trial.json" "$output/config.json" "${V35_TASK_EXECUTION:-eager}" <<'PYCONFIG'
import json, sys
cfg=json.load(open(sys.argv[1]))
if sys.argv[3]=='eager':
    for key in ('compile_fk','compile_step','compile_v24_step'):
        cfg[key]=False
json.dump(cfg,open(sys.argv[2],'w'),indent=2)
PYCONFIG
git -C "$ROOT" rev-parse HEAD > "$output/revision.txt"
"$ISAACLAB_PY" "$ROOT/scripts/local/run_diagnostic_group.py" --timeout "${V35_FULL_TIMEOUT:-240}" --grace 5 -- \
  bash "$ROOT/scripts/train_stage1_world_model.sh" \
  --mode physical_full_v35_check --num_envs 4096 \
  --config "$output/config.json" \
  --checkpoint "$checkpoint" --reinitialize_replay --migrate_event_execution \
  --diagnostic_repeats 3 --diagnostic_warmups 1 \
  --output "$output/run" --no_debug_vis > "$output/run.log" 2>&1
"$ISAACLAB_PY" - "$output/run/physical_full_v35.json" <<'PY'
import json, sys
r=json.load(open(sys.argv[1]))
assert r['completed'] and r['model_restored']
print('Stage:',r['requested_stage'])
print('Candidate stage gate:',r.get('candidate_numeric_passed',r.get('candidate_wm_numeric_passed',r.get('candidate_actor_numeric_passed'))))
print('Eager task execution is a numerical isolation test, not a production speed comparison.')
print('No formal training or checkpoint writes.')
PY
