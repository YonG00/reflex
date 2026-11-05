#!/usr/bin/env bash
# Resume ONLY a repaired v3 smoke checkpoint after its GPU checks pass.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
source scripts/local/env.sh
CKPT=${1:?Supply repaired v3 checkpoint}
OUT=${2:?Supply a NEW output directory}
if [[ -e "$OUT" ]]; then echo "Output already exists: $OUT" >&2; exit 1; fi
bash scripts/train_stage1_world_model.sh --mode train --num_envs 1024 --max_iterations 320000 \
 --max_seconds 1800 --save_every 100 --enable_actor --checkpoint "$CKPT" \
 --config configs/world_model/joint_repaired_v21_compiled.json --output "$OUT"
