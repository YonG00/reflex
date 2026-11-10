#!/usr/bin/env bash
# Official PPO + extended box information, with escape failure ONLY.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export BOX_ESCAPE_PENALTY=1 BOX_IMPACT_BUFFER=0
exec bash "$ROOT/scripts/train_stage1_box_info.sh" "$@"
