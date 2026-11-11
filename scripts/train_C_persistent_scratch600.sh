#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export BOX_PERSISTENT_HOLD=1
exec bash "$ROOT/scripts/train_C_scratch600.sh" --run_name C_persistent_hold_scratch600 "$@"
