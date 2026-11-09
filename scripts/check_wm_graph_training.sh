#!/usr/bin/env bash
# Fresh stable WM only: never reinterpret legacy checkpoint encoding.
set -euo pipefail
if [[ $# -ne 1 ]]; then
  echo "Usage: $0 NEW_OUTPUT_DIR (fresh stable-encoding WM/Actor joint check)"
  exit 2
fi
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
[[ ! -e $1 ]] || exit 2
mkdir -p "$1"
output=$(cd "$1" && pwd)
source "$ROOT/scripts/local/env.sh"
export PYTHONPATH="$ROOT/source/reflex:$ROOT/source/whole_body_tracking${PYTHONPATH:+:$PYTHONPATH}"
export TORCHINDUCTOR_CACHE_DIR=${WM_GRAPH_INDUCTOR_CACHE:-/tmp/reflex_stable_training_inductor}
export TRITON_CACHE_DIR=${WM_GRAPH_TRITON_CACHE:-/tmp/reflex_stable_training_triton}
"$ISAACLAB_PY" "$ROOT/scripts/local/run_diagnostic_group.py" --timeout 900 --grace 10 -- \
  bash "$ROOT/scripts/train_stage1_world_model.sh" --mode wm_graph_joint_check \
  --num_envs 4096 --config "${WM_GRAPH_CONFIG:-$ROOT/configs/world_model/joint_h16_batch4096_wm_graph.json}" \
  --diagnostic_repeats 5 --diagnostic_warmups 2 --no_debug_vis --output "$output" \
  > "$output/run.log" 2>&1
"$ISAACLAB_PY" - "$output/wm_graph_joint.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
assert r['completed'] and r['numeric_passed'] and r['performance_usable']
assert r.get('cuda_graph_launch_events',0)>0 and not any(r['profile_compile_delta'].values())
assert r['model_restored'] and r['actor_restored']
for name,v in r['variants'].items(): print(name,v['seconds'])
print('Joint numeric/Graph gates passed; inspect paired speed and memory before training.')
PY
