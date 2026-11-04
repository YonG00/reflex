#!/bin/bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
MODE=${1:-teacher}; shift || true
STAMP=$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)
OUT=$REFLEX_ROOT/outputs/play/${STAMP}_${MODE}
mkdir -p "$OUT"
for SEED in 8 9 10; do
  bash scripts/eval.sh "$MODE" --num_envs 1 --seed "$SEED" --out "$OUT/seed_${SEED}.json" --video_path "$OUT/seed_${SEED}.mp4" "$@" 2>&1 | tee "$OUT/seed_${SEED}.log"
done
