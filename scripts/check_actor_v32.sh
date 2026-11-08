#!/usr/bin/env bash
# Run only after the user opens the GPU. No formal training or checkpoint writes.
set -euo pipefail
if [[ ${1:-} == --help || $# -ne 2 ]]; then
  echo "Usage: $0 CHECKPOINT NEW_OUTPUT_DIR"
  echo "Set ISAACLAB_PY and REFLEX_DATA_DIR to the GPU server's existing environment/data."
  echo "V32_REPEATS=20; V32_STAGE_TIMEOUT=1800 seconds."
  [[ ${1:-} == --help ]] && exit 0
  exit 2
fi
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
checkpoint=$1
output=$2
[[ -f "$checkpoint" ]] || { echo "Missing checkpoint" >&2; exit 2; }
[[ ! -e "$output" ]] || { echo "Refusing existing output: $output" >&2; exit 2; }
command -v timeout >/dev/null || { echo "GNU timeout required" >&2; exit 2; }
repeats=${V32_REPEATS:-20}
limit=${V32_STAGE_TIMEOUT:-1800}
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
echo "Paired V32 diagnostic only: $output"
timeout --signal=TERM --kill-after=30 "$limit" bash "$ROOT/scripts/train_stage1_world_model.sh" \
  --mode actor_execution_v32_check --num_envs 4096 \
  --config "$ROOT/configs/world_model/joint_execution_v30_verified_actor_trial.json" \
  --checkpoint "$checkpoint" --reinitialize_replay \
  --diagnostic_repeats "$repeats" --diagnostic_warmups 2 \
  --output "$output/run" --no_debug_vis > "$output/run.log" 2>&1
"$ISAACLAB_PY" - "$output/run/actor_execution_v32.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
for name, v in r['variants'].items():
    print(name, 'median seconds:', v.get('wall_seconds'), 'numeric:', v.get('numeric_passed'))
for key in ('completed', 'numeric_passed', 'performance_usable', 'model_unchanged', 'actor_restored'):
    if not r.get(key):
        raise SystemExit(f'Diagnostic gate failed: {key}; inspect per-variant reports; no training started')
print('Diagnostic gates passed; speed gains still need review; no automatic promotion or training')
PY
