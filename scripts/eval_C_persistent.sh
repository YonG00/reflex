#!/usr/bin/env bash
# Evaluate the persistent C policy with the same feature flags as training.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export BOX_PERSISTENT_HOLD=1
exec bash "$ROOT/scripts/eval_buffer_s3_ablation.sh" C "$@"
