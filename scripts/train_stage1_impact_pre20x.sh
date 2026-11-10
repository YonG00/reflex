#!/usr/bin/env bash
# Official PPO+AMP + 19-D box information; pre-contact penalty 0.2 -> 4.0.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
set -a
source "$ROOT/configs/train/impact_pre20x.env"
set +a
exec bash "$ROOT/scripts/train_stage1_box_info.sh" "$@"
