#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
GROUP=${1:?usage: eval_buffer_s3_ablation.sh S3|A|B|C --ckpt CHECKPOINT [eval args]};shift
source "$ROOT/scripts/s3_experiment_env.sh"
exec bash "$ROOT/scripts/eval.sh" teacher_box_info --physical_hold_metrics "$@"
