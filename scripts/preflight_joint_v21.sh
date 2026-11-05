#!/usr/bin/env bash
# Run only after a GPU has been enabled. Does not start a long training run.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
source scripts/local/env.sh
OUT=${1:?Supply a NEW output directory}
if [[ -e "$OUT" ]]; then echo "Output already exists: $OUT" >&2; exit 1; fi
mkdir -p "$OUT"
bash scripts/train_stage1_world_model.sh --mode preflight --num_envs 64 --preflight_steps 128 \
 --config configs/world_model/joint_repaired_v21.json --output "$OUT/parity"
bash scripts/train_stage1_world_model.sh --mode train --num_envs 1024 --max_iterations 60 \
 --max_seconds 300 --save_every 20 --enable_actor --profile_joint \
 --config configs/world_model/joint_repaired_v21.json --output "$OUT/smoke"
