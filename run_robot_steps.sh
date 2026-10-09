#!/usr/bin/env bash
# Steps 2-5: replay the human demos on the robot, train the policy, evaluate it,
# then improve it with its own successes and evaluate again.
# Run inside the `libero` conda env from the repo folder:  bash run_robot_steps.sh
set -e
export MUJOCO_GL=${MUJOCO_GL:-osmesa} PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-osmesa}
TRACKS=${TRACKS:-$PWD/data/tracks}
VIDEOS=${VIDEOS:-$HOME/hand_demos}
OUT=${OUT:-$HOME/results}

echo "== Step 2: replay each human demo on the robot (plus extra can positions) =="
(cd sim && python retarget_replay.py --tracks "$TRACKS" --videos "$VIDEOS" \
    --out "$OUT/replay" --pattern "normal_*" --aug "${AUG:-8}")

echo "== Step 2b: replay each edge-case clip once =="
(cd sim && python retarget_replay.py --tracks "$TRACKS" --videos "$VIDEOS" \
    --out "$OUT/replay_edge" --pattern "edge*" --aug 0) || true

echo "== Step 3: train the policy =="
python policy/train_bc.py --demos "$OUT/replay/demos" --out "$OUT/policy"

echo "== Step 4: evaluate it =="
python policy/evaluate.py --policy "$OUT/policy/policy.pt" --demos "$OUT/replay/demos" \
    --out "$OUT/eval" --grid "${GRID:-7}"

echo "== Step 5: improve it with its own successes, then evaluate again =="
python policy/self_improve.py --policy "$OUT/policy/policy.pt" --demos "$OUT/replay/demos" \
    --out "$OUT/self_improve" --rounds "${ROUNDS:-3}"
python policy/evaluate.py --policy "$OUT/self_improve/policy_round${ROUNDS:-3}/policy.pt" \
    --demos "$OUT/replay/demos" --out "$OUT/eval_after_self_improve" --grid "${GRID:-7}"
echo "All done. Results in $OUT"
