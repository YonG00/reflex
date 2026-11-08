#!/usr/bin/env bash
# Complete-chain diagnosis only. Requires a new output directory and checkpoint.
set -euo pipefail
if [[ ${1:-} == --help || $# -ne 2 ]]; then
  echo "Usage: $0 CHECKPOINT NEW_OUTPUT_DIR"
  echo "Set ISAACLAB_PY and REFLEX_DATA_DIR to the existing GPU environment/data."
  echo "V33_REPEATS=20; V33_STAGE_TIMEOUT=2400 seconds. No formal training."
  [[ ${1:-} == --help ]] && exit 0
  exit 2
fi
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
checkpoint=$1
output=$2
[[ -f "$checkpoint" && ! -e "$output" ]] || { echo "Need existing checkpoint and new output directory" >&2; exit 2; }
command -v timeout >/dev/null || { echo "GNU timeout required" >&2; exit 2; }
repeats=${V33_REPEATS:-20}
limit=${V33_STAGE_TIMEOUT:-2400}
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
timeout --signal=TERM --kill-after=30 "$limit" bash "$ROOT/scripts/train_stage1_world_model.sh" \
  --mode execution_study_v33 --num_envs 4096 \
  --config "$ROOT/configs/world_model/joint_execution_v32_verified_actor_trial.json" \
  --checkpoint "$checkpoint" --reinitialize_replay \
  --diagnostic_repeats "$repeats" --diagnostic_warmups 2 \
  --output "$output/run" --no_debug_vis > "$output/run.log" 2>&1
"$ISAACLAB_PY" -m reflex.world_model.diagnostics_v31.trace_analysis \
  "$output/run/full/actor_full_trace.json" "$output/run/full/actor_full_trace_summary.json" > "$output/trace_summary.log"
"$ISAACLAB_PY" - "$output/run/execution_study_v33.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
for key in ('completed', 'numeric_passed', 'attribution_usable', 'timings_usable'):
    if not r.get(key):
        raise SystemExit(f'Diagnostic gate failed: {key}; inspect logs and partial reports')
print('Diagnostic checks passed. Interpret source/runtime measurements before selecting an execution rewrite.')
PY
