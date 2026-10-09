"""
Step 4: test the policy on a grid of can start positions.

The grid is wider than the region used for training data, so the plot shows both
where the policy works and where it stops working. Each failure gets a cause:
    no_grasp      - the can was never lifted
    dropped       - lifted, then fell outside the basket
    missed_basket - put down next to the basket, not in it
    timeout       - still holding the can when time ran out

Usage (libero env, from the repo root):
    python policy/evaluate.py --policy ~/results/policy/policy.pt \
        --demos ~/results/replay/demos --out ~/results/eval --grid 7
"""
import argparse
import csv
import glob
import os
import sys
from collections import Counter

import imageio
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sim"))
import libero_utils as U          # noqa: E402
from train_bc import Policy       # noqa: E402

GRID_X = (-0.30, 0.16)   # training region is x in (-0.22, 0.10)
GRID_Y = (-0.36, 0.10)   # training region is y in (-0.30, 0.05)
MAX_STEPS = 1000   # LIBERO episode limit; replays of slow human demos can take 600+ steps
COLOURS = {"success": "#2e8b57", "no_grasp": "#d1495b", "dropped": "#edae49",
           "missed_basket": "#8e6c8a", "timeout": "#7a7a7a", "occupied": "#e6e6e6"}


def rollout(env, obs, policy, record=False):
    """Run the policy once. Returns (result, frames)."""
    frames, highest = [], 0.0
    policy.reset()
    for _ in range(MAX_STEPS):
        obs, _, _, _ = env.step(policy.act(U.lowdim(obs)))
        highest = max(highest, obs[f"{U.CAN}_pos"][2])
        if record:
            frames.append(obs["agentview_image"][::-1])
        if U.success(env):
            return "success", frames

    can, basket = obs[f"{U.CAN}_pos"], obs[f"{U.BASKET}_pos"]
    if highest < 0.09:
        return "no_grasp", frames
    if can[2] > 0.09:
        return "timeout", frames
    if np.linalg.norm(can[:2] - basket[:2]) < 0.12:
        return "missed_basket", frames
    return "dropped", frames


def in_region(r, x_range, y_range):
    return x_range[0] <= r["can_x"] <= x_range[1] and y_range[0] <= r["can_y"] <= y_range[1]


def plot(rows, train_xy, path, x_range, y_range):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    for result, colour in COLOURS.items():
        pts = np.array([(r["can_y"], r["can_x"]) for r in rows if r["result"] == result])
        if len(pts):
            label = "spot taken by another object" if result == "occupied" else result
            ax.scatter(pts[:, 0], pts[:, 1], s=420, marker="s", color=colour, label=label,
                       edgecolors="white", linewidths=1.5)
    if len(train_xy):
        ax.scatter(train_xy[:, 1], train_xy[:, 0], s=10, color="black", alpha=0.6,
                   label="can positions in training data")
    ax.add_patch(plt.Rectangle((y_range[0], x_range[0]),
                               y_range[1] - y_range[0],
                               x_range[1] - x_range[0],
                               fill=False, ls="--", color="black", lw=1))
    tested = [r for r in rows if r["result"] != "occupied"]
    n_ok = sum(r["result"] == "success" for r in tested)
    ax.set_title(f"Policy success by can start position: {n_ok}/{len(tested)}\n"
                 "(dashed box = region of the training data)")
    ax.set_xlabel("can y (m), towards the basket →")
    ax.set_ylabel("can x (m), away from the robot →")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=9)
    ax.set_aspect("equal")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--demos", help="replay folder, to plot where the training cans were")
    ap.add_argument("--out", required=True)
    ap.add_argument("--grid", type=int, default=7)
    ap.add_argument("--gifs", type=int, default=3, help="how many example rollouts to save as GIFs")
    ap.add_argument("--execute", type=int,
                    help="chunked policies: steps to carry out before re-planning (default: half the chunk)")
    ap.add_argument("--x-range", type=float, nargs=2, default=U.CAN_X_RANGE,
                    help="region the training cans came from (for the plot and the inside/outside split)")
    ap.add_argument("--y-range", type=float, nargs=2, default=U.CAN_Y_RANGE)
    args = ap.parse_args()
    out = os.path.expanduser(args.out)
    os.makedirs(out, exist_ok=True)

    policy = Policy.load(args.policy)
    if args.execute:
        policy.execute = args.execute

    fast_env, render_env = U.make_env(render=False), None
    rows, gifs = [], 0
    for i, x in enumerate(np.linspace(*GRID_X, args.grid)):
        for j, y in enumerate(np.linspace(*GRID_Y, args.grid)):
            record = gifs < args.gifs and (i + j) % 3 == 0
            if record and render_env is None:
                render_env = U.make_env(render=True)
            env = render_env if record else fast_env
            env.seed(0)
            env.reset()
            obs = U.settle(env)
            row = dict(can_x=round(x, 3), can_y=round(y, 3))
            if not U.spot_is_free(obs, np.array([x, y])):
                rows.append(dict(row, result="occupied"))
                continue
            obs = U.set_can_xy(env, np.array([x, y]))
            result, frames = rollout(env, obs, policy, record)
            rows.append(dict(row, result=result))
            print(f"can at ({x:+.2f}, {y:+.2f}) -> {result}", flush=True)
            if record:
                imageio.mimsave(os.path.join(out, f"policy_{result}_{i}{j}.gif"), frames[::3], duration=0.12)
                gifs += 1

    with open(os.path.join(out, "eval_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["can_x", "can_y", "result"])
        w.writeheader()
        w.writerows(rows)

    train_xy = []
    for f in glob.glob(os.path.join(os.path.expanduser(args.demos or ""), "*.npz")):
        d = np.load(f)
        if bool(d["success"]):
            train_xy.append(d["can_xy"])
    plot(rows, np.array(train_xy), os.path.join(out, "success_heatmap.png"), args.x_range, args.y_range)

    tested = [r for r in rows if r["result"] != "occupied"]
    inside = [r for r in tested if in_region(r, args.x_range, args.y_range)]
    outside = [r for r in tested if not in_region(r, args.x_range, args.y_range)]

    def rate(rs):
        return f"{sum(r['result'] == 'success' for r in rs)}/{len(rs)}"
    print(f"\nSuccess: overall {rate(tested)} | inside training region {rate(inside)}"
          f" | outside {rate(outside)}")
    print("Outcomes:", dict(Counter(r["result"] for r in tested)))


if __name__ == "__main__":
    main()
