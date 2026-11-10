#!/usr/bin/env bash
# Native resume: same checkpoint/optimizer, no observation padding or policy-only warm start.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
GROUP=${1:?usage: train_buffer_s3_ablation.sh S3|A|B|C MODEL_599 [training args]}
CKPT=${2:?model_599.pt required}; shift 2
source "$ROOT/scripts/s3_experiment_env.sh"
[ -s "$CKPT" ] || { echo "missing checkpoint: $CKPT" >&2; exit 1; }
CKPT=$(cd "$(dirname "$CKPT")" && pwd)/$(basename "$CKPT")
# A dedicated read-only source directory is exposed to the existing native resume selector.
SOURCE_ID="s3_ablation_source_${GROUP}_$(date +%Y%m%d_%H%M%S)_$$"
SOURCE_DIR="$ROOT/logs/rsl_rl/reflex/$SOURCE_ID"
mkdir -p "$SOURCE_DIR"
ln -s "$CKPT" "$SOURCE_DIR/model_599.pt"
PYTHONPATH="$ROOT/source/reflex${PYTHONPATH:+:$PYTHONPATH}" ${ISAACLAB_PY:-python} "$ROOT/scripts/s3_checkpoint_check.py" "$CKPT"
bash "$ROOT/scripts/train_stage1_buffer.sh" 3 --num_envs 4096 --max_iterations 200 \
  --run_name "s3_${GROUP}" --resume True --load_run "^${SOURCE_ID}$" --checkpoint '^model_599.pt$' "$@"
