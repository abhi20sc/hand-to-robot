#!/usr/bin/env bash
# Last experiment: action chunking. Same data as before, but the network predicts the
# next 10 actions at once. Tested on the demo-region data (~/results) and on the
# whole-table data (~/results_wide), where plain BC hesitated in the middle.
# Run inside the `libero` conda env from the repo folder:  bash run_chunking.sh
set -e
export MUJOCO_GL=${MUJOCO_GL:-osmesa} PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-osmesa}
OUT=${OUT:-$HOME/results_chunking}
mkdir -p "$OUT"
WIDE="--x-range -0.30 0.16 --y-range -0.36 0.10"

python policy/train_bc.py --demos ~/results/replay/demos --out "$OUT/narrow/policy" --chunk 10
python policy/train_bc.py --demos ~/results_wide/replay/demos --out "$OUT/wide/policy" --chunk 10

# narrow data: carry out 5 planned steps before re-planning, or re-plan every step
python policy/evaluate.py --policy "$OUT/narrow/policy/policy.pt" --demos ~/results/replay/demos \
    --out "$OUT/narrow/eval_exec5" --grid 7 --gifs 2 | tee "$OUT/narrow_exec5.log"
python policy/evaluate.py --policy "$OUT/narrow/policy/policy.pt" --demos ~/results/replay/demos \
    --out "$OUT/narrow/eval_exec1" --grid 7 --gifs 0 --execute 1 | tee "$OUT/narrow_exec1.log"
# whole-table data
python policy/evaluate.py --policy "$OUT/wide/policy/policy.pt" --demos ~/results_wide/replay/demos \
    --out "$OUT/wide/eval_exec5" --grid 7 --gifs 2 $WIDE | tee "$OUT/wide_exec5.log"

echo; echo "=== summary (plain BC for comparison: narrow 12/34, wide 5/34) ==="
for n in narrow_exec5 narrow_exec1 wide_exec5; do echo "$n: $(grep '^Success' "$OUT/$n.log")"; done
