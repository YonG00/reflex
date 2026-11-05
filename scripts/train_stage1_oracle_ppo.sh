#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
need "$MOTION" "$AMP_DATASET"
load_env "$ROOT/configs/train/stage1_teacher.env" "$ROOT/configs/world_model/oracle.env"
cd "$ROOT"
$PY -m reflex.teacher.train --task Reflex-Teacher-Oracle-G1-v0 --headless --motion_file "$MOTION" \
  --experiment_name reflex_oracle_ppo --logger tensorboard "$@"
