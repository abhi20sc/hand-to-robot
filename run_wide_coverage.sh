#!/usr/bin/env bash
# Experiment: the first policy only fails where my demos never put the can.
# Replaying is free, so replay every human demo across the WHOLE test area,
# retrain, and test again on the same grid.
# Run inside the `libero` conda env from the repo folder:  bash run_wide_coverage.sh
set -e
export MUJOCO_GL=${MUJOCO_GL:-osmesa} PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-osmesa}
TRACKS=${TRACKS:-$PWD/data/tracks}
VIDEOS=${VIDEOS:-$HOME/hand_demos}
OUT=${OUT:-$HOME/results_wide}
X="-0.30 0.16"   # same area as the evaluation grid
Y="-0.36 0.10"

(cd sim && python retarget_replay.py --tracks "$TRACKS" --videos "$VIDEOS" --out "$OUT/replay" \
    --pattern "normal_*" --aug "${AUG:-12}" --x-range $X --y-range $Y)
python policy/train_bc.py --demos "$OUT/replay/demos" --out "$OUT/policy"
python policy/evaluate.py --policy "$OUT/policy/policy.pt" --demos "$OUT/replay/demos" \
    --out "$OUT/eval" --grid 7 --x-range $X --y-range $Y
echo "All done. Results in $OUT"
