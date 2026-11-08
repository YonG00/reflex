#!/usr/bin/env bash
# Standalone capsule-only diagnosis. No Isaac, replay collection or training.
set -euo pipefail
if [[ ${1:-} == --help || $# -ne 2 ]]; then
  echo "Usage: $0 EXISTING_CAPSULE NEW_OUTPUT_DIR"
  echo "ISAACLAB_PY; V35_TIMEOUT=600; V35_NO_FMA=0 (opt-in isolated Triton experiment)"
  [[ ${1:-} == --help ]] && exit 0
  exit 2
fi
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
capsule=$1
output=$2
[[ -f "$capsule" && ! -e "$output" ]] || { echo "Need existing capsule and new output" >&2; exit 2; }
[[ ${V35_TIMEOUT:-600} =~ ^[1-9][0-9]*$ ]] || exit 2
[[ ${V35_NO_FMA:-0} =~ ^[01]$ ]] || exit 2
command -v timeout >/dev/null || { echo "GNU timeout required" >&2; exit 2; }
mkdir -p "$output"
output=$(cd "$output" && pwd)
export PYTHONPATH="$ROOT/source/reflex${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
if [[ ${V35_NO_FMA:-0} == 1 ]]; then
  # Must be set before Python/Torch import. Do not reuse the baseline FX cache.
  export TRITON_DEFAULT_FP_FUSION=0
  export TORCHINDUCTOR_CACHE_DIR="$output/inductor_cache"
  export TRITON_CACHE_DIR="$output/triton_cache"
else
  [[ ${TRITON_DEFAULT_FP_FUSION:-1} == 1 ]] || { echo "Unexpected inherited FP fusion setting" >&2; exit 2; }
fi
git -C "$ROOT" rev-parse HEAD > "$output/revision.txt"
git -C "$ROOT" diff --stat > "$output/working_tree.txt"
# Python rejects an existing output, so reports live in a fresh child directory.
timeout --signal=TERM --kill-after=20 "${V35_TIMEOUT:-600}" \
  "${ISAACLAB_PY:-/opt/conda/envs/isaaclab/bin/python}" -m reflex.world_model.physical_numerics_v35 \
  --capsule "$capsule" --output "$output/run" --device cuda:0 \
  --steps 1 6 16 32 --drivers eager segmented wm_mlp wm_layerwise \
  --repeats 10 --warmups 2 > "$output/run.log" 2>&1
# Completion is not a correctness pass; display individual gates honestly.
"${ISAACLAB_PY:-/opt/conda/envs/isaaclab/bin/python}" - "$output/run/h32/minimal_actor_v31.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
assert r['completed']
for name, v in r['drivers'].items():
    print(name, 'numeric_passed=', v['numeric']['passed'], 'seconds=', v['wall_seconds']['median'])
print('No candidate is promoted automatically. A failed reference/candidate check remains a failure.')
PY
