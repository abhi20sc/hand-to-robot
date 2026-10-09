#!/usr/bin/env bash
# How much does each ingredient matter? Re-uses the replays from run_robot_steps.sh,
# so it only trains and evaluates (no new replays).
#   A) no augmentation : each human demo replayed once, at its original can position
#   B) 4 human clips   (with augmentation)
#   C) 8 human clips   (with augmentation)
# The full model (all clips + augmentation) is ~/results/eval.
# Run inside the `libero` conda env from the repo folder:  bash run_ablations.sh
set -e
export MUJOCO_GL=${MUJOCO_GL:-osmesa} PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-osmesa}
DEMOS=${DEMOS:-$HOME/results/replay/demos}
OUT=${OUT:-$HOME/results_ablations}

run() {   # name, extra train_bc arguments
    python policy/train_bc.py --demos "$DEMOS" --out "$OUT/$1/policy" "${@:2}"
    python policy/evaluate.py --policy "$OUT/$1/policy/policy.pt" --demos "$DEMOS" \
        --out "$OUT/$1/eval" --grid 7 --gifs 0 | tee "$OUT/$1.log"
}
mkdir -p "$OUT"
run no_augmentation --max-replays 1
run clips_4 --clips 4
run clips_8 --clips 8

echo; echo "=== summary ==="
for n in no_augmentation clips_4 clips_8; do echo "$n: $(grep '^Success' "$OUT/$n.log")"; done
