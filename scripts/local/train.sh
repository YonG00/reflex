#!/bin/bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
STAGE=${1:-}; shift || true
case "$STAGE" in
  bc) bash scripts/train_stage2.sh bc "$@" ;;
  ppo) bash scripts/train_stage2.sh ppo --init "$REFLEX_DATA_DIR/stage2/bc_init.pt" "$@" ;;
  stage1_scratch) bash scripts/train_stage1.sh "$@" ;;
  stage1_continue) bash scripts/train_stage1.sh --warm_start "$REFLEX_DATA_DIR/checkpoints/teacher_stage1.pt" "$@" ;;
  stage3) bash scripts/train_stage3.sh "$@" ;;
  *) echo 'Usage: train.sh stage1_scratch|stage1_continue|bc|ppo|stage3 [upstream options]'; exit 2 ;;
esac
