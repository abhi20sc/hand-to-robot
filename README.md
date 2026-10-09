# From my hand to a Panda arm: can-into-bin from 20 phone videos

I filmed my own hand putting a can into a bin with a phone, turned those videos into
robot demonstrations in LIBERO, trained a policy on them, and tested where it works,
where it breaks, and why.

![my hand (left) and the robot replaying it (right)](media/side_by_side.gif)

**Task:** LIBERO-Object, *"pick up the alphabet soup and place it in the basket"*. I chose
it because I could film the same thing at home: a can and a bin.

Of the challenge's suggested directions, this covers retargeting my data to a robot arm,
training a policy on it, and a simple RL-style self-improvement step on top (which did
not help, see below).

## Results at a glance

| | |
|---|---|
| Clips I recorded | 20 (16 normal, 4 deliberate edge cases) |
| Clips where the grasp and release were found | 20 / 20 |
| Robot replays of my demos that succeed | 138 / 144 in my demo region, 195 / 208 across the whole table |
| **Final policy, 34 can positions across the table** | **23 / 34 (68%)** |
| First policy (plain behaviour cloning), same test | 12 / 34 (35%) |

The final policy is the same small network as the first one, trained on replays that
cover the whole table, and predicting 10 actions at a time instead of 1. How I got
there, including the experiments that failed, is below.

![final policy](media/policy_final.gif)

## How it works

```
phone video ─► 1. hand tracking ─► 2. retarget + replay ─► 3. behaviour cloning ─► 4. evaluate
               (MediaPipe)         (LIBERO, Panda arm)       (small MLP, 10-step    (grid of 34 can
                                                              action chunks)         positions)
```

### 1. Hand tracking (`perception/track_hands.py`)
MediaPipe gives 21 hand keypoints per frame. From them I compute one signal, the **grip
aperture**: thumb-tip to index-tip distance divided by hand size, so it does not change
when my hand moves towards the camera.

Holding the can is the longest low-aperture stretch during which the hand also travels.
Its start is the grasp and its end the release. The low/high threshold comes from
Otsu's method, so there is no hand-tuned constant.

**The can is never detected.** It is wherever my hand closed; the bin is wherever it
opened.

![aperture over one clip](media/aperture_example.png)

### 2. Retargeting and replay (`sim/retarget_replay.py`)
Two points are known in both worlds: where I grasped (= the can) and where I released
(= the bin). Two point pairs define exactly one 2-D similarity transform (rotation +
scale + shift). Applying it to my hand path puts my route onto the robot's table.
**No camera calibration: the can and the bin are the reference points.**

- A top-down phone camera cannot see height, so the gripper height comes from the task
  phase (travel high, line up, go down, grasp, lift, carry, lower, release).
- I hold the can from the side with my whole hand; the Panda grips it from above with
  two fingers. I transfer *where and when* (path and grasp/release timing), not hand shape.
- The transform is recomputed for wherever the can is, so **each of my demos is replayed
  at many can positions**. This turns 16 clips into ~200 robot demos.
- The extra replays add small random pushes to the robot while recording the
  controller's clean correction, so the data also shows how to recover from mistakes.

### 3. Behaviour cloning (`policy/train_bc.py`)
A 3-layer MLP takes 14 numbers (gripper position, finger opening, can and basket
relative to the gripper) and outputs velocity commands and open/close. In the final
version it outputs the **next 10 actions**; the robot carries out 5, then asks again
(action chunking, the core idea of ACT).

Validation holds out **whole clips**, not random time steps, so it measures
generalisation to demos the network has not seen.

### 4. Evaluation (`policy/evaluate.py`)
The policy starts from a 7×7 grid of can positions (34 that are not blocked by other
objects), wider than where I put the can when filming. Every failure is labelled:
`no_grasp`, `dropped`, `missed_basket`, `timeout`.

| First policy (12 / 34) | Final policy (23 / 34) |
|---|---|
| ![](media/heatmap_first.png) | ![](media/heatmap_final.png) |

Black dots are the can positions in each policy's training data, and the dashed box is
the region they were sampled from (my demo region on the left, the whole table on the
right). Grey squares are spots another object already occupies. The dense dot is the
can's default position, where every demo is first replayed without changes. The final
policy's remaining failures are at the far edge of the table and right next to the basket.

## The experiments, in order

Each row changes one thing. Success counts are out of 11 positions inside the region my
demos covered, 23 outside, and 34 in total; every policy is evaluated once on the same
grid.

| # | Policy | Inside (11) | Outside (23) | Total (34) |
|---|---|---|---|---|
| 1 | Behaviour cloning, replays in my demo region | 9 | 3 | 12 |
| 2 | Same, each demo replayed **once** (no augmentation) | 0 | 0 | 0 |
| 3 | Same, only 4 of my clips | 4 | 1 | 5 |
| 4 | Same, only 8 of my clips | 8 | 2 | 10 |
| 5 | #1 + self-improvement (3 rounds) | 6 | 1 | 7 |
| 6 | Behaviour cloning, replays over the **whole table** | 1 | 4 | 5 |
| 7 | #1 + action chunking | 10 | 4 | 14 |
| 8 | #7 but re-plan every step | 4 | 1 | 5 |
| **9** | **Whole-table replays + action chunking (final)** | **9** | **14** | **23** |

What each one taught me:

- **Augmentation is what makes it work (#1 vs #2).** With each demo replayed only at its
  original spot, the network never learns "go to wherever the can is" and fails
  everywhere. That policy also had the *lowest* validation loss of all, which is a
  reminder that offline loss is not the same as the robot succeeding.
- **More of my clips helps, with diminishing returns (#3, #4, #1):** 4 → 8 clips doubles
  success, 8 → 14 adds one more.
- **Self-improvement made it worse (#5).** I ran the policy with a little noise, kept the
  successes (reward = can in basket) and retrained. Success per round fell 27 → 25 → 21
  out of 60. The saved actions were the noisy ones the robot executed, so each round
  learned sloppier motion, and random noise found successes where the policy already
  worked rather than at the edges.
- **More coverage alone made it worse (#6).** The policy reached new corners but lost
  the middle: 17 of 34 runs grasped the can and then never delivered it (`timeout`).
  My diagnosis: with many different routes in the data, nearly identical states come
  with conflicting actions, and a network that predicts one action at a time averages
  them into a hesitant motion.
- **Action chunking fixed it (#9).** Predicting 10 steps and committing to 5 of them
  means the robot follows one coherent route instead of averaging between routes at
  every step. On the whole-table data this took success from 5 to 23. On the narrow data
  (#7) it changed little, because there were fewer conflicting routes to begin with.
- **Committing is the important part (#8).** The same chunked network re-planning every
  step drops back to 5 / 34, so the gain comes from executing the plan, not from
  predicting the future as an extra training signal.

Caveat: each policy is evaluated once on 34 positions, so differences of one or two
successes are noise. The jumps that the conclusions rest on (0 vs 9, 5 vs 23) are not.

## Edge cases

| Clip | What I did | Tracking | Robot replay |
|---|---|---|---|
| Edge_Fast | about 2× speed | grasp found, hold only 1.2 s | success |
| Edge Occlusion | arm covering the can | grasp found | success |
| Edge_Lying | can on its side | grasp found | success |
| Edge_Right | other hand | hand seen in 85% of frames | success |
| Normal_1 | normal | tracking lost for 1.6 s | success after the detector fix |

The pipeline is robust to speed and to my arm covering the can, because it only needs
the hand. It is blind to the object's state: with the can lying down, the robot still
does an upright top-down grasp, because only *where* and *when* are transferred, not
*how*. Tracking dropouts during the grasp moment are the main risk for the detector.

## Other things that went wrong

- **My first grasp detector was wrong for my own data.** I assumed my hand would open
  wide just before grasping. My plots showed the opposite for a side grip: the aperture
  is *lowest* while holding. It picked the wrong moment in 2 clips and missed 2. The
  "longest low stretch with the most hand travel" rule fixed all of them.
- **Labels that disagree.** The first cloned policy hovered over the can forever. The
  controller only closed the gripper after settling within 3 mm, so almost identical
  states were labelled both "open" and "close". Closing within 1.2 cm made the decision
  a clean function of the state.
- **Episode limits.** I hold the can for 8–14 s, so a faithful copy takes 600+ steps. My
  first evaluation stopped at 400 steps and counted half the runs as failures, and
  LIBERO's own 1000-step limit cut off the next attempt.

## What I would do next
- **A policy that represents several good actions explicitly** (ACT's CVAE or a diffusion
  policy). Chunking hides the averaging problem by committing; it does not remove it.
- **Images instead of object positions** as input, so the policy does not rely on the
  simulator telling it where the can is.
- **Height from the video**, with a second camera or the phone's depth sensor, instead of
  assuming it from the task phase.
- **Residual RL** on top of the cloned policy for the last millimetres of the grasp.

## Run it

`data/tracks/` holds the hand keypoints, grasp and release frames that step 1 extracted
from my 20 clips, so steps 2-4 run without the videos (about 300 MB each, not in the repo).

```bash
# 1. hand tracking, only needed for new videos (Python 3.10 env)
conda create -n hands python=3.10 -y && conda activate hands
pip install mediapipe==0.10.14 opencv-python matplotlib "numpy<2"
python perception/track_hands.py --videos <folder of clips> --out <output folder>

# 2-4. robot side (LIBERO env, see https://github.com/Lifelong-Robot-Learning/LIBERO)
# run from the repo folder; the scripts read data/tracks by default
conda activate libero
bash run_robot_steps.sh       # demo-region replays, policy #1, evaluation, self-improvement (#5)
bash run_ablations.sh         # #2-#4
bash run_wide_coverage.sh     # whole-table replays, policy #6
bash run_chunking.sh          # #7, #8 and the final policy #9
```
Everything runs on a laptop CPU (WSL Ubuntu, no GPU), roughly 4 hours for all of it.

## Repo layout
```
data/tracks/*.npz           my 20 clips after hand tracking (keypoints, grasp, release)
perception/track_hands.py   video → hand keypoints, grasp and release
sim/libero_utils.py         small helpers for the LIBERO task
sim/retarget_replay.py      hand path → robot demos (replayed at many can positions)
policy/train_bc.py          behaviour cloning, optional action chunking
policy/evaluate.py          success heatmap over a grid of can positions
policy/self_improve.py      keep the policy's own successes, retrain
run_*.sh                    the experiments above
```
