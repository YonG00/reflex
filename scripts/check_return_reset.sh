#!/usr/bin/env bash
# Only scratch GPU diagnostics; does not launch the30-minute training run.
set -euo pipefail
[[ $# -eq 1 ]] || { echo "Usage: $0 NEW_OUTPUT_DIR"; exit 2; }
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
[[ ! -e $1 ]] || { echo "Output already exists"; exit 2; }
mkdir -p "$1"
output=$(cd "$1" && pwd)
source "$ROOT/scripts/local/env.sh"
export PYTHONPATH="$ROOT/source/reflex:$ROOT/source/whole_body_tracking${PYTHONPATH:+:$PYTHONPATH}"
"$ISAACLAB_PY" "$ROOT/scripts/local/run_diagnostic_group.py" --timeout 900 --grace 10 -- \
 bash "$ROOT/scripts/train_stage1_world_model.sh" --mode return_reset_check \
 --num_envs 4096 --config "$ROOT/configs/world_model/joint_return_reset_h16_batch4096.json" \
 --no_debug_vis --output "$output" > "$output/run.log" 2>&1
"$ISAACLAB_PY" - "$output/return_reset_check.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
assert r['completed'] and r['passed'] and r['actor_parameters_changed']
assert r['original_resample_preserved'] and r['native_events_passed']
print(json.dumps(r,indent=2))
print('Reset/event/WM/Actor smoke passed; inspect numeric and timing evidence before formal training.')
PY
