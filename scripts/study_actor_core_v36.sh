#!/usr/bin/env bash
# Complete production Actor benchmark. Numeric failures remain research results.
set -euo pipefail
if [[ ${1:-} == --help || $# -ne 2 ]]; then
  echo "Usage: $0 CHECKPOINT NEW_OUTPUT_DIR"
  echo "V36_TIMEOUT=300; V36_REPEATS=20. Production FMA, 2048/H32, no training."
  [[ ${1:-} == --help ]] && exit 0
  exit 2
fi
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
checkpoint=$1
output=$2
[[ -f "$checkpoint" && ! -e "$output" ]] || exit 2
[[ ${V36_TIMEOUT:-300} =~ ^[1-9][0-9]*$ && ${V36_REPEATS:-20} =~ ^[1-9][0-9]*$ ]] || exit 2
checkpoint=$(cd "$(dirname "$checkpoint")" && pwd)/$(basename "$checkpoint")
mkdir -p "$output"
output=$(cd "$output" && pwd)
source "$ROOT/scripts/local/env.sh"
export PYTHONPATH="$ROOT/source/reflex:$ROOT/source/whole_body_tracking${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONFAULTHANDLER=1
# Optional diagnostic only: compare compiler worker count without changing math.
if [[ -n ${V36_COMPILE_THREADS:-} ]]; then
  [[ $V36_COMPILE_THREADS =~ ^[1-9][0-9]*$ ]] || exit 2
  export TORCHINDUCTOR_COMPILE_THREADS=$V36_COMPILE_THREADS
fi
export TRITON_DEFAULT_FP_FUSION=1
# An explicitly separate FMA=1 cache; never use V35's FMA=0 cache.
export TORCHINDUCTOR_CACHE_DIR=${V36_INDUCTOR_CACHE:-/tmp/reflex_v36_fma1_inductor}
export TRITON_CACHE_DIR=${V36_TRITON_CACHE:-/tmp/reflex_v36_fma1_triton}
"$ISAACLAB_PY" - "$ROOT" <<'PY'
import importlib.util,sys
from pathlib import Path
actual=Path(importlib.util.find_spec('reflex.world_model.train').origin).resolve()
assert actual==Path(sys.argv[1],'source/reflex/reflex/world_model/train.py').resolve()
print('Verified source:',actual)
PY
git -C "$ROOT" rev-parse HEAD > "$output/revision.txt"
"$ISAACLAB_PY" "$ROOT/scripts/local/run_diagnostic_group.py" --timeout "${V36_TIMEOUT:-300}" --grace 5 -- \
  bash "$ROOT/scripts/train_stage1_world_model.sh" \
  --mode actor_fusion_study_v36 --num_envs 4096 \
  --config "$ROOT/configs/world_model/joint_execution_v32_verified_actor_trial.json" \
  --checkpoint "$checkpoint" --reinitialize_replay --migrate_event_execution \
  --diagnostic_repeats "${V36_REPEATS:-20}" --diagnostic_warmups 2 \
  --output "$output/run" --no_debug_vis > "$output/run.log" 2>&1
"$ISAACLAB_PY" - "$output/run/actor_fusion_v36.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
assert r['completed'] and r['performance_usable']
for name,v in r['variants'].items():
    print(name,'strict_numeric_passed=',v['numeric_passed'],'seconds=',v['wall_seconds']['median'])
print('Read numerical risks and cold compile cost. No formal configuration promotion.')
PY
