#!/bin/bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
CSV=$REFLEX_DATA_DIR/bones_seed/g1/csv/210707/walk_forward_grab_normal_003__A006_M.csv
B10="$REFLEX_DATA_DIR/amass_g1/g1/ACCAD/Male2Walking_c3d/B10 -  Walk turn left 45_poses_120_jpos.npy"
for f in "$CSV" "$B10"; do test -f "$f" || { echo "Missing $f"; exit 1; }; done
"$ISAACLAB_PY" tools/motion/build_reference_motion.py --headless --bones_csv "$CSV"
"$ISAACLAB_PY" tools/motion/build_amass_motion.py --headless --z_offset 0.0 --amass_npy "$B10" --out "$REFLEX_DATA_DIR/motions/male2_walk_turn_left_45.npz"
"$ISAACLAB_PY" tools/motion/build_amass_motion.py --headless --z_offset 0.76 --amass_npy "$B10" --out "$REFLEX_DATA_DIR/motions/male2_walk_v2.npz"
"$ISAACLAB_PY" tools/motion/build_amp_dataset.py --reference_motion "$REFLEX_DATA_DIR/motions/walk_then_catch_v17_30s.npz" --male2_walk_v2 "$REFLEX_DATA_DIR/motions/male2_walk_v2.npz" --male2_turn_left_45 "$REFLEX_DATA_DIR/motions/male2_walk_turn_left_45.npz" --out "$REFLEX_DATA_DIR/amp/amp_dataset.pt"
