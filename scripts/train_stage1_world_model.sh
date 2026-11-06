#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
need "$MOTION" "$AMP_DATASET"
load_env "$ROOT/configs/train/stage1_teacher.env" "$ROOT/configs/world_model/oracle.env"
if [[ -n "${REFLEX_WM_ENV_FILE:-}" ]]; then load_env "$REFLEX_WM_ENV_FILE"; fi
cd "$ROOT"
$PY -m reflex.world_model.train --headless "$@"
