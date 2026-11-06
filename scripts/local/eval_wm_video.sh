#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
exec "$ISAACLAB_PY" "$REFLEX_ROOT/scripts/local/eval_wm_video.py" "$@"
