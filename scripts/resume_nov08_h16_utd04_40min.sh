#!/usr/bin/env bash
# Resume Nov08 Actor805 checkpoint. No fresh WM pretraining; fresh PhysX/Replay.
set -euo pipefail
[[ $# -eq 2 ]] || { echo "Usage: $0 NOV08_CHECKPOINT NEW_OUTPUT_DIR"; exit 2; }
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
checkpoint=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
[[ -f "$checkpoint" && ! -e "$2" ]] || { echo "Missing checkpoint or output already exists"; exit 2; }
source "$ROOT/scripts/local/env.sh"
config="$ROOT/configs/world_model/joint_resume_nov08_h16_b3584_utd04.json"
# Reject a wrong experiment before loading Isaac Sim or spending GPU time.
"$ISAACLAB_PY" - "$checkpoint" "$config" <<'PY'
import json,sys,torch
s=torch.load(sys.argv[1],map_location='cpu',weights_only=False)
c=json.load(open(sys.argv[2])); old=s['config']
assert s['actor_updates']==805, 'Expected Nov08 final Actor805, not the Nov09 Actor249 model'
assert old.get('wm_encoding','legacy')=='legacy'
assert old['horizon']==16 and old['batch']==4096 and old['updates_per_env_step']==0.4
assert old.get('return_reset_max_step_reward') is None
assert c['collect_steps']==1 and c['updates_per_env_step']==0.4
assert c['joint_updates_per_env_step'] is None
assert all(s.get(k) is not None for k in ('model','actor','critic','model_opt','actor_opt','critic_opt','obs_norm','critic_norm','amp')), 'Incomplete resume checkpoint'
PY
mkdir -p "$2"
output=$(cd "$2" && pwd)
cp "$config" "$output/requested_config.json"
"$ISAACLAB_PY" "$ROOT/scripts/local/migrate_joint_batch.py" "$checkpoint" "$output/resume_batch3584.pt" --batch 3584 > "$output/batch_migration.log"
export PYTHONPATH="$ROOT/source/reflex:$ROOT/source/whole_body_tracking${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"$ISAACLAB_PY" "$ROOT/scripts/local/run_diagnostic_group.py" --timeout 4800 --grace 10 -- \
 bash "$ROOT/scripts/train_stage1_world_model.sh" --mode train --num_envs 4096 \
 --config "$config" --checkpoint "$output/resume_batch3584.pt" --reinitialize_replay --initialize_return_reset \
 --max_iterations 100000 --max_seconds 2400 --save_every 200 --enable_actor \
 --no_debug_vis --output "$output" > "$output/run.log" 2>&1
"$ISAACLAB_PY" - "$output" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);r=json.loads((p/'training_phases.json').read_text())
assert r['state']=='completed' and (p/'checkpoint_final.pt').is_file()
assert r['initial_actor_updates']==805
assert r['wm_warmup_completed_before_loop']
assert r['actor_updates']>805
print(json.dumps(r,indent=2))
PY
