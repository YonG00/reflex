#!/usr/bin/env bash
# Complete-chain diagnosis only. Requires a new output directory and checkpoint.
set -euo pipefail
if [[ ${1:-} == --help || $# -ne 2 ]]; then
  echo "Usage: $0 CHECKPOINT NEW_OUTPUT_DIR"
  echo "Set ISAACLAB_PY and REFLEX_DATA_DIR to the existing GPU environment/data."
  echo "V34_REPEATS=20; V34_STAGE_TIMEOUT=2400 seconds. No formal training."
  [[ ${1:-} == --help ]] && exit 0
  exit 2
fi
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
checkpoint=$1
output=$2
[[ -f "$checkpoint" && ! -e "$output" ]] || { echo "Need existing checkpoint and new output directory" >&2; exit 2; }
command -v timeout >/dev/null || { echo "GNU timeout required" >&2; exit 2; }
repeats=${V34_REPEATS:-20}
limit=${V34_STAGE_TIMEOUT:-2400}
[[ $repeats =~ ^[1-9][0-9]*$ && $limit =~ ^[1-9][0-9]*$ ]] || exit 2
(( repeats <= 100 )) || exit 2
checkpoint=$(cd "$(dirname "$checkpoint")" && pwd)/$(basename "$checkpoint")
mkdir -p "$output"
output=$(cd "$output" && pwd)
source "$ROOT/scripts/local/env.sh"
export PYTHONPATH="$ROOT/source/reflex:$ROOT/source/whole_body_tracking${PYTHONPATH:+:$PYTHONPATH}"
git -C "$ROOT" rev-parse HEAD > "$output/revision.txt"
git -C "$ROOT" diff --stat > "$output/working_tree.txt"
telemetry_pid=
cleanup() { if [[ -n "$telemetry_pid" ]]; then kill "$telemetry_pid" 2>/dev/null || true; wait "$telemetry_pid" 2>/dev/null || true; fi; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=timestamp,name,memory.used,utilization.gpu,power.draw,clocks.sm --format=csv -l 1 > "$output/gpu.csv" 2>&1 &
  telemetry_pid=$!
fi
# Native PhysX equivalence includes forced event cases and H32 launch gradient.
timeout --signal=TERM --kill-after=30 "$limit" bash "$ROOT/scripts/train_stage1_world_model.sh" \
  --mode event_condition_probe --num_envs 64 \
  --config "$ROOT/configs/world_model/joint_execution_v34_actor_trial.json" \
  --checkpoint "$checkpoint" --reinitialize_replay --migrate_event_execution \
  --output "$output/native" --no_debug_vis > "$output/native.log" 2>&1
"$ISAACLAB_PY" - "$output/native/event_condition_probe.json" <<'PYPROBE'
import json, sys
r = json.load(open(sys.argv[1]))
assert r['passed'] and r['positive_perturb_writes'] > 0
assert r['launch_velocity_robot_position_gradient_norm'] > 0
assert r['actor_imagination_horizon'] == 32
assert r['actor_waiting_to_launch_transitions'] > 0
PYPROBE
timeout --signal=TERM --kill-after=30 "$limit" bash "$ROOT/scripts/train_stage1_world_model.sh" \
  --mode actor_execution_v34_check --num_envs 4096 \
  --config "$ROOT/configs/world_model/joint_execution_v34_actor_trial.json" \
  --checkpoint "$checkpoint" --reinitialize_replay --migrate_event_execution \
  --diagnostic_repeats "$repeats" --diagnostic_warmups 2 \
  --output "$output/run" --no_debug_vis > "$output/run.log" 2>&1
"$ISAACLAB_PY" - "$output/run/actor_execution_v34.json" <<'PYEND'
import json, sys
r = json.load(open(sys.argv[1]))
for key in ('completed', 'numeric_passed', 'performance_usable'):
    if not r.get(key):
        raise SystemExit(f'Diagnostic gate failed: {key}; inspect logs')
print('Paired checks passed. No automatic promotion or formal training.')
PYEND
