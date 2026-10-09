"""
Step 2: turn each human demo into a robot demo in LIBERO.

From the video we know, in image pixels:
  - where the hand closed  (= where the can was)
  - where the hand opened  (= where the bin was)
  - the path of the hand before and between those moments

In the sim we know where the can and the basket are. Two matching point pairs
define one 2-D similarity transform (rotate + scale + shift). Applying it to the
hand path maps the human's route onto the table in the sim. No camera
calibration is needed: the can and the bin are the reference points.

Because the transform is recomputed for wherever the can is, one human demo can
be replayed for many can positions. Those replays are the training data for step 3.

A top-down camera cannot see height, so the gripper height comes from the phase
of the task instead (move high, go down, grasp, lift, carry, lower, release).

Usage (libero env, from the sim folder):
    python retarget_replay.py --tracks ../data/tracks --out ~/results/replay --aug 8
    (add --videos <folder of clips> to also get side-by-side GIFs)
"""
import argparse
import csv
import fnmatch
import glob
import os

import imageio
import numpy as np

import libero_utils as U

CTRL_HZ = 20            # LIBERO control frequency
SLOWDOWN = 1.5          # robot moves 1.5x slower than the human did
GAIN = 1.5              # proportional gain of the waypoint follower
APPROACH_SECONDS = 1.0  # how much of the human's reach before the grasp to reuse
# Region the Panda reaches comfortably (metres). Points further out put the arm in
# joint configurations it cannot recover from, so retargeted paths are clipped to it.
SAFE_MIN = np.array([-0.32, -0.38])
SAFE_MAX = np.array([0.17, 0.30])
OPEN, CLOSE = -1.0, 1.0


# ------------------------------------------------------------------ human side
def load_track(path):
    d = np.load(path)
    return dict(name=os.path.splitext(os.path.basename(path))[0], fps=float(d["fps"]),
                grasp=int(d["grasp"]), release=int(d["release"]),
                palm=d["palm"], grip=d["grip_point"])


def human_paths(t):
    """Reach and carry paths in pixels. The palm centre is shifted so the path goes
    through the fingertip midpoint at the grasp and release moments."""
    g, r = t["grasp"], t["release"]
    off_g = t["grip"][g] - t["palm"][g]
    off_r = t["grip"][r] - t["palm"][r]

    start = max(0, g - int(APPROACH_SECONDS * t["fps"]))
    reach = t["palm"][start:g + 1] + off_g
    w = np.linspace(0, 1, r - g + 1)[:, None]
    carry = t["palm"][g:r + 1] + (1 - w) * off_g + w * off_r

    reach = reach[~np.isnan(reach[:, 0])]   # drop frames where tracking was lost
    carry = carry[~np.isnan(carry[:, 0])]
    return reach, carry, t["grip"][g], t["grip"][r]


def similarity(a0, a1, b0, b1):
    """2-D similarity T with T(a0) = b0 and T(a1) = b1.
    Written with complex numbers: T(z) = m*z + c, where m = scale * rotation."""
    m = (complex(*b1) - complex(*b0)) / (complex(*a1) - complex(*a0))
    c = complex(*b0) - m * complex(*a0)

    def T(p):
        z = m * (p[:, 0] + 1j * p[:, 1]) + c
        return np.stack([z.real, z.imag], 1)
    return T, abs(m)


def to_sim(t, can_xy, basket_xy):
    """Map the human paths onto the sim table. Returns reach, carry (metres) and scale."""
    reach, carry, can_px, bin_px = human_paths(t)
    flip = np.array([1.0, -1.0])  # image y points down, world y points up (seen from above)
    T, scale = similarity(can_px * flip, bin_px * flip, can_xy, basket_xy)
    return T(reach * flip), T(carry * flip), scale


def resample(path, fps):
    """Human path at video fps -> one waypoint per robot control step."""
    n = max(2, round(len(path) / fps * SLOWDOWN * CTRL_HZ))
    s, d = np.linspace(0, 1, len(path)), np.linspace(0, 1, n)
    return np.stack([np.interp(d, s, path[:, i]) for i in range(2)], 1)


# ------------------------------------------------------------------ robot side
def build_plan(t, obs):
    """List of waypoints (xyz, gripper, phase, tolerance).
    tolerance = None: one control step towards xyz (used to follow a path).
    tolerance = number: keep going until within that many metres (max 60 steps)."""
    can, basket = obs[f"{U.CAN}_pos"], obs[f"{U.BASKET}_pos"]
    reach, carry, scale = to_sim(t, can[:2], basket[:2])
    reach = np.clip(resample(reach, t["fps"]), SAFE_MIN, SAFE_MAX)
    carry = np.clip(resample(carry, t["fps"]), SAFE_MIN, SAFE_MAX)

    high = U.CARRY_Z
    above_can = can[2] + U.HOVER_DZ
    at_can = can[2] + U.GRASP_DZ
    above_basket = basket[2] + U.PLACE_DZ

    plan = [(np.r_[reach[0], high], OPEN, "reach", 0.01)]                # human's reach
    plan += [(np.r_[xy, high], OPEN, "reach", None) for xy in reach[1:]]
    plan += [(np.r_[can[:2], high], OPEN, "reach", 0.01),
             (np.r_[can[:2], above_can], OPEN, "align", 0.003),       # line up precisely
             (np.r_[can[:2], at_can], OPEN, "descend", 0.003)]
    plan += [(np.r_[can[:2], at_can], CLOSE, "grasp", None)] * 15
    plan += [(np.r_[can[:2], high], CLOSE, "lift", 0.01)]
    plan += [(np.r_[xy, high], CLOSE, "carry", None) for xy in carry]   # human's carry
    plan += [(np.r_[basket[:2], high], CLOSE, "carry", 0.01),
             (np.r_[basket[:2], above_basket], CLOSE, "lower", 0.01)]
    plan += [(np.r_[basket[:2], above_basket], OPEN, "release", None)] * 15
    plan += [(np.r_[basket[:2], high], OPEN, "retreat", 0.01)]
    return plan, scale


def execute(env, obs, plan, record=False, noise=0.0, rng=None):
    """Follow the plan with a proportional controller and log (state, action) pairs.

    noise > 0: the robot executes a randomly disturbed action, but the log keeps the
    controller's clean action. The data then shows how to recover from small mistakes,
    which is exactly what a cloned policy needs at test time (the "DART" trick)."""
    states, actions, frames = [], [], []
    for target, grip, phase, tol in plan:
        for _ in range(60 if tol else 1):
            err = target - obs["robot0_eef_pos"]
            a = np.zeros(7, dtype=np.float32)
            a[:3] = np.clip(err / 0.05 * GAIN, -1, 1)
            a[6] = grip
            # Close as soon as the gripper is within 1.2 cm of the grasp pose (and open
            # within 1.2 cm of the drop pose). Without this, almost identical states get
            # both labels ("still moving, open" / "now close") and the cloned policy
            # hovers over the can without ever closing.
            if phase == "descend" and np.linalg.norm(err) < 0.012:
                a[6] = CLOSE
            if phase == "lower" and np.linalg.norm(err) < 0.012:
                a[6] = OPEN
            states.append(U.lowdim(obs))
            actions.append(a)
            executed = a.copy()
            if noise > 0:
                executed[:3] = np.clip(executed[:3] + rng.normal(0, noise, 3), -1, 1)
            obs, _, _, _ = env.step(executed)
            if record:
                frames.append(obs["agentview_image"][::-1])
            if tol and np.linalg.norm(err) < tol:
                break
    return U.success(env), np.array(states), np.array(actions), frames


def side_by_side_gif(video_path, t, sim_frames, out_path, height=256):
    """Human clip (left) next to the robot replay (right)."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    human = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        w = int(f.shape[1] * height / f.shape[0])
        human.append(cv2.cvtColor(cv2.resize(f, (w, height)), cv2.COLOR_BGR2RGB))
    cap.release()
    if not human or not sim_frames:
        return
    start = max(0, t["grasp"] - int(APPROACH_SECONDS * t["fps"]))
    end = min(len(human) - 1, t["release"] + int(0.5 * t["fps"]))
    idx = np.linspace(start, end, len(sim_frames)).astype(int)
    frames = [np.concatenate([human[i], s], 1) for i, s in zip(idx, sim_frames)]
    imageio.mimsave(out_path, frames[::3], duration=0.15)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", required=True, help="folder of .npz files from track_hands.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--videos", help="original clips, for side-by-side GIFs")
    ap.add_argument("--aug", type=int, default=8, help="extra random can positions per demo")
    ap.add_argument("--pattern", default="normal_*", help="which clips to use (case-insensitive)")
    ap.add_argument("--noise", type=float, default=0.15,
                    help="action noise in the extra replays (0 = off), see execute()")
    ap.add_argument("--x-range", type=float, nargs=2, default=U.CAN_X_RANGE,
                    help="where to place the can in the extra replays (metres)")
    ap.add_argument("--y-range", type=float, nargs=2, default=U.CAN_Y_RANGE)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = os.path.expanduser(args.out)
    os.makedirs(os.path.join(out, "demos"), exist_ok=True)
    os.makedirs(os.path.join(out, "gifs"), exist_ok=True)
    tracks = sorted(p for p in glob.glob(os.path.join(os.path.expanduser(args.tracks), "*.npz"))
                    if fnmatch.fnmatch(os.path.basename(p).lower(), args.pattern.lower() + ".npz"))
    if not tracks:
        raise SystemExit("no tracks found")

    rng = np.random.default_rng(args.seed)
    render_env = U.make_env(render=True)    # first replay of each demo, for the GIF
    fast_env = U.make_env(render=False)     # augmented replays, physics only
    rows = []
    for path in tracks:
        t = load_track(path)
        if t["grasp"] < 0:
            print(f"[{t['name']}] skipped: no grasp found in the video")
            continue

        n_ok = n_run = 0
        for k in range(args.aug + 1):
            env = render_env if k == 0 else fast_env
            env.seed(int(rng.integers(1e6)))
            env.reset()
            obs = U.settle(env)
            if k > 0:                               # move the can somewhere new
                xy = U.sample_can_xy(rng, obs, args.x_range, args.y_range)
                if xy is None:
                    continue
                obs = U.set_can_xy(env, xy)
            can_xy = obs[f"{U.CAN}_pos"][:2].copy()
            plan, scale = build_plan(t, obs)
            noise = args.noise if k > 0 else 0.0   # keep the first replay clean for the GIF
            ok, states, actions, frames = execute(env, obs, plan, record=(k == 0), noise=noise, rng=rng)
            n_ok += ok
            n_run += 1

            np.savez(os.path.join(out, "demos", f"{t['name']}_{k:02d}.npz"),
                     success=ok, can_xy=can_xy, states=states, actions=actions)
            rows.append(dict(clip=t["name"], rollout=k, can_x=round(can_xy[0], 3),
                             can_y=round(can_xy[1], 3), success=int(ok),
                             steps=len(actions), mm_per_px=round(scale * 1000, 2)))
            if k == 0:
                imageio.mimsave(os.path.join(out, "gifs", f"{t['name']}_robot.gif"),
                                frames[::3], duration=0.15)
                videos = glob.glob(os.path.join(os.path.expanduser(args.videos or ""), t["name"] + ".*"))
                if videos:
                    side_by_side_gif(videos[0], t, frames,
                                     os.path.join(out, "gifs", f"{t['name']}_side_by_side.gif"))
        print(f"[{t['name']}] replayed {n_run} times, success {n_ok}/{n_run}", flush=True)

    with open(os.path.join(out, "replay_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nOverall replay success: {sum(r['success'] for r in rows)}/{len(rows)}")


if __name__ == "__main__":
    main()
