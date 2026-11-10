#!/usr/bin/env bash
# Bounded three-stage physics/PPO regression, 128 envs x 12 updates per stage.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CKPT=${1:?usage: regress_stage1_buffer.sh CHECKPOINT [OUTPUT_DIRECTORY]}
OUT=${2:-$ROOT/outputs/buffer_stage_regression_$(date +%Y%m%d_%H%M%S)}
[ -s "$CKPT" ] || { echo "checkpoint missing: $CKPT"; exit 1; }
CKPT=$(cd "$(dirname "$CKPT")" && pwd)/$(basename "$CKPT")
[ ! -e "$OUT/stage1.log" ] || { echo "choose a fresh output directory: $OUT"; exit 1; }
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
trap 'rc=$?; if [ "$rc" -ne 0 ]; then echo "failed exit=$rc $(date -Is)" > "$OUT/status.txt"; fi' EXIT
for stage in 1 2 3; do
  echo "stage=$stage $(date -Is)" > "$OUT/status.txt"
  BOX_BUFFER_LOG_EVERY=1 BOX_BUFFER_LOG_PATH="$OUT/stage${stage}.jsonl" \
    bash "$ROOT/scripts/train_stage1_buffer.sh" "$stage" --num_envs 128 --max_iterations 12 \
    --seed 42 --run_name "buffer_stage${stage}_short" --warm_start "$CKPT" \
    > "$OUT/stage${stage}.log" 2>&1
done
echo "complete $(date -Is)" > "$OUT/status.txt"
