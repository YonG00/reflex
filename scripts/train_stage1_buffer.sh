#!/usr/bin/env bash
# Stage 1=lean freedom; 2=+directional intake; 3=+transition retention.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
STAGE=${1:?usage: train_stage1_buffer.sh 1|2|3 [original training args]}
shift
case "$STAGE" in 1|2|3) ;; *) echo 'stage must be 1,2,3'; exit 2;; esac
export BOX_BUFFER_STAGE=$STAGE BOX_ESCAPE_PENALTY=0 BOX_IMPACT_BUFFER=0
exec bash "$ROOT/scripts/train_stage1_box_info.sh" "$@"
