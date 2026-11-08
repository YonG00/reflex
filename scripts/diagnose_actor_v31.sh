#!/usr/bin/env bash
# Diagnostic updates are restored/discarded; this does not start training.
set -euo pipefail
if [[ ${1:-} == --help || $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 CHECKPOINT NEW_OUTPUT_DIR [EXISTING_TRACKERLAB_REPO]"
  echo "Run on the GPU server with ISAACLAB_PY set to the Isaac Lab Python."
  echo "Optional: V31_STAGE_TIMEOUT=2400 (seconds per process); V31_REPEATS=20."
  [[ ${1:-} == --help ]] && exit 0
  exit 2
fi
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
checkpoint=$1
output=$2
tracker=${3:-}
[[ -f "$checkpoint" ]] || { echo "Missing checkpoint: $checkpoint" >&2; exit 2; }
[[ ! -e "$output" ]] || { echo "Refusing existing output: $output" >&2; exit 2; }
command -v timeout >/dev/null || { echo "GNU timeout is required on the GPU server" >&2; exit 2; }
limit=${V31_STAGE_TIMEOUT:-2400}
repeats=${V31_REPEATS:-20}
[[ $limit =~ ^[1-9][0-9]*$ && $repeats =~ ^[1-9][0-9]*$ ]] || exit 2
(( repeats <= 100 )) || exit 2
if [[ -n "$tracker" ]]; then
  [[ -d "$tracker/source/trackerLab" ]] || { echo "Missing TrackerLab checkout" >&2; exit 2; }
  tracker=$(cd "$tracker" && pwd)
fi
checkpoint=$(cd "$(dirname "$checkpoint")" && pwd)/$(basename "$checkpoint")
mkdir -p "$output"
output=$(cd "$output" && pwd)
source "$ROOT/scripts/local/env.sh"
export PYTHONPATH="$ROOT/source/reflex:$ROOT/source/whole_body_tracking${PYTHONPATH:+:$PYTHONPATH}"
telemetry_pid=
cleanup() { if [[ -n "$telemetry_pid" ]]; then kill "$telemetry_pid" 2>/dev/null || true; wait "$telemetry_pid" 2>/dev/null || true; fi; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,power.draw,clocks.sm --format=csv -l 1 > "$output/gpu.csv" 2>&1 &
  telemetry_pid=$!
fi
git -C "$ROOT" rev-parse HEAD > "$output/revision.txt"
git -C "$ROOT" diff --stat > "$output/working_tree.txt"
echo "Full Actor/Critic profile and same-Replay capsule: $output/full"
timeout --signal=TERM --kill-after=30 "$limit" bash "$ROOT/scripts/train_stage1_world_model.sh" \
  --mode actor_v31_diagnostics --num_envs 4096 \
  --config "$ROOT/configs/world_model/joint_execution_v30_verified_actor_trial.json" \
  --checkpoint "$checkpoint" --reinitialize_replay \
  --diagnostic_repeats "$repeats" --diagnostic_warmups 2 \
  --output "$output/full" --no_debug_vis > "$output/full.log" 2>&1
"$ISAACLAB_PY" - "$output/full/actor_full_v31.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
for key in ('completed', 'numeric_passed', 'attribution_usable'):
    if not r.get(key):
        raise SystemExit(f'Full diagnostic gate failed: {key}; inspect report and log')
print('Full diagnostic gates passed; median complete seconds:', r['wall_seconds'])
PY
failed=0
run_minimal() {
  local label=$1
  shift
  echo "Minimal H32 control: $label"
  if timeout --signal=TERM --kill-after=30 "$limit" "$ISAACLAB_PY" \
    -m reflex.world_model.diagnostics_v31.minimal "$@" \
    --batch 2048 --horizon 32 --repeats "$repeats" --warmups 2 \
    --output "$output/$label" > "$output/$label.log" 2>&1; then
    if ! "$ISAACLAB_PY" - "$output/$label/minimal_actor_v31.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
if not r.get('completed') or not r.get('numeric_passed'):
    raise SystemExit('Minimal numeric/completion gate failed')
for name, d in r['drivers'].items():
    if not d['wm_frozen'] or not d['physical_bptt_nonzero']:
        raise SystemExit(f'Minimal gradient gate failed: {name}')
print('Minimal diagnostic gates passed')
PY
    then failed=1; fi
  else
    echo "$label process failed or timed out; inspect its log" >&2
    failed=1
  fi
}
run_minimal reflex_minimal --capsule "$output/full/minimal_capsule.pt"
if [[ -n "$tracker" ]]; then
  run_minimal tracker_minimal --tracker_repo "$tracker"
fi
echo "Reports: $output (no training checkpoint saved)"
exit "$failed"
