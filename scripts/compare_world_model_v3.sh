#!/usr/bin/env bash
# Frozen-data 2x2 experiment. No simulator, Actor updates or AMP training.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
CKPT=${1:?Usage: $0 CHECKPOINT URDF OUTPUT [UPDATES]}
URDF=${2:?Supply the exact URDF used for collection}
OUT=${3:?Supply a new experiment directory}
UPDATES=${4:-256}
PY=${ISAACLAB_PY:-python}
mkdir -p "$OUT"
for size in small large; do
  if [[ "$size" == small ]]; then widths=256,256; else widths=2048,2048,2048; fi
  for objective in multitask state; do
    tag=${size}_${objective}
    "$PY" -m reflex.world_model.offline --checkpoint "$CKPT" --urdf "$URDF" \
      --fresh --objective "$objective" --hidden_dims "$widths" --activation silu \
      --cases 8:1 --updates "$UPDATES" --batch 128 --lr 0.0002 --grad_clip 1 \
      --eval_every 64 --output "$OUT/$tag" > "$OUT/$tag.log" 2>&1
  done
done
printf 'complete\n' > "$OUT/status.txt"
