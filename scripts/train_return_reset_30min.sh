#!/usr/bin/env bash
# Fresh H16/dual4096 trial. Total1800s from training-loop start, including WM warmup.
set -euo pipefail
[[ $# -eq 2 ]] || { echo "Usage: $0 PASSED_GPU_CHECK_DIR NEW_TRAIN_OUTPUT_DIR"; exit 2; }
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$ROOT/scripts/local/env.sh"
"$ISAACLAB_PY" - "$1/return_reset_check.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
assert r['completed'] and r['passed'] and r['native_events_passed'] and r['actor_parameters_changed']
PY
[[ ! -e $2 ]] || { echo "Output already exists"; exit 2; }
mkdir -p "$2"
output=$(cd "$2" && pwd)
export PYTHONPATH="$ROOT/source/reflex:$ROOT/source/whole_body_tracking${PYTHONPATH:+:$PYTHONPATH}"
"$ISAACLAB_PY" "$ROOT/scripts/local/run_diagnostic_group.py" --timeout 2700 --grace 10 -- \
 bash "$ROOT/scripts/train_stage1_world_model.sh" --mode train --num_envs 4096 \
 --config "${REFLEX_RETURN_RESET_CONFIG:-$ROOT/configs/world_model/joint_return_reset_h16_batch4096.json}" \
 --max_iterations 100000 --max_seconds 1800 --save_every 200 --enable_actor \
 --no_debug_vis --output "$output" > "$output/run.log" 2>&1
"$ISAACLAB_PY" - "$output" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);r=json.loads((p/'training_phases.json').read_text())
assert r['state']=='completed' and (p/'checkpoint_final.pt').is_file()
print(json.dumps(r,indent=2))
PY
