#!/usr/bin/env bash
# C reward/action ablation, original single actor, 600 fresh PPO updates.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
for arg in "$@"; do
 case "$arg" in --resume|--resume=*|--warm_start|--warm_start=*|--warm_start_ckpt|--warm_start_ckpt=*|--load_run|--load_run=*|--checkpoint|--checkpoint=*)
  echo 'C scratch600 must not load or resume a checkpoint' >&2;exit 2;;
 esac
done
for key in BOX_RETAIN_X10 BOX_DYNAMIC_BUFFER; do
 [ "${!key:-0}" = 0 ] || { echo "C must run independently: unset $key" >&2;exit 2; }
done
export BOX_BUFFER_STAGE=3 BOX_FLEX_ARMS=1 BOX_RETAIN_X10=0 BOX_DYNAMIC_BUFFER=0
export BOX_ESCAPE_PENALTY=0 BOX_IMPACT_BUFFER=0
export BOX_BUFFER_INTAKE_WEIGHT=.75 BOX_BUFFER_RETAIN_WEIGHT=.5
exec bash "$ROOT/scripts/train_stage1_buffer.sh" 3 --num_envs 4096 --max_iterations 600 \
 --seed 42 --run_name C_scratch600 "$@"
