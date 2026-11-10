#!/usr/bin/env bash
# Explicit separate invocations: short first, full only after inspecting short results.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODE=${1:?usage: run_s3_ablations.sh short|full MODEL_599 OUTPUT_DIR}
CKPT=${2:?model_599.pt required};OUT=${3:?fresh output directory required}
case "$MODE" in short) NENV=128;UPDATES=12;; full) NENV=4096;UPDATES=200;; *) exit 2;; esac
for key in BOX_RETAIN_X10 BOX_DYNAMIC_BUFFER BOX_FLEX_ARMS; do
  [ "${!key:-0}" = 0 ] || { echo "unset $key before the independent queue";exit 2; }
done
# Validate baseline reward defaults before spending any GPU time.
GROUP=S3 source "$ROOT/scripts/s3_experiment_env.sh"
[ ! -e "$OUT" ] || { echo 'output directory must be new';exit 2; }
mkdir -p "$OUT";OUT=$(cd "$OUT" && pwd)
CKPT=$(cd "$(dirname "$CKPT")" && pwd)/$(basename "$CKPT")
git -C "$ROOT" rev-parse HEAD > "$OUT/source_commit.txt"
git -C "$ROOT" diff --binary > "$OUT/source_diff.patch"
${ISAACLAB_PY:-python} "$ROOT/scripts/s3_checkpoint_check.py" "$CKPT" > "$OUT/checkpoint.json"
trap 'rc=$?; if [ "$rc" -ne 0 ]; then echo "failed exit=$rc" > "$OUT/status.txt"; fi' EXIT
for GROUP in S3 A B C; do
 RUN="s3_${GROUP}_${MODE}_$(date +%Y%m%d_%H%M%S)_$$"
 echo "$GROUP training" > "$OUT/status.txt"
 BOX_BUFFER_LOG_EVERY=1 BOX_BUFFER_LOG_PATH="$OUT/${GROUP}.jsonl" \
   bash "$ROOT/scripts/train_buffer_s3_ablation.sh" "$GROUP" "$CKPT" --num_envs "$NENV" \
   --max_iterations "$UPDATES" --seed 42 --run_name "$RUN" > "$OUT/${GROUP}.train.log" 2>&1
 ${ISAACLAB_PY:-python} "$ROOT/scripts/check_s3_run.py" "$OUT/${GROUP}.train.log" "$OUT/${GROUP}.jsonl" "$UPDATES" > "$OUT/${GROUP}.check.json"
 # Select only the exact unique run just launched; never another group's checkpoint.
 FINAL=$(${ISAACLAB_PY:-python} - "$ROOT" "$RUN" <<'PY'
import sys
from pathlib import Path
runs=list((Path(sys.argv[1])/'logs/rsl_rl/reflex').glob('*_'+sys.argv[2]))
if len(runs)!=1:raise RuntimeError(f'expected exactly one run: {runs}')
ckpts=list(runs[0].glob('model_*.pt'))
print(max(ckpts,key=lambda p:int(p.stem.split('_')[-1])).resolve())
PY
 )
 printf '%s\n' "$FINAL" > "$OUT/${GROUP}.checkpoint.txt"
 if [ "$MODE" = full ]; then
  for SEED in 8 9; do
   echo "$GROUP evaluation seed=$SEED" > "$OUT/status.txt"
   bash "$ROOT/scripts/eval_buffer_s3_ablation.sh" "$GROUP" --ckpt "$FINAL" --seed "$SEED" --num_envs 320 \
     --out "$OUT/${GROUP}_s${SEED}.json" --video_dir "$OUT/video_${GROUP}_s${SEED}" > "$OUT/${GROUP}_s${SEED}.eval.log" 2>&1
  done
 fi
done
if [ "$MODE" = full ]; then
 ${ISAACLAB_PY:-python} "$ROOT/scripts/summarize_s3_ablations.py" "$OUT" > "$OUT/comparison.json"
fi
echo complete > "$OUT/status.txt"
