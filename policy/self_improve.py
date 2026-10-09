"""
Step 5 (optional): improve the policy with its own successes.

This is the simplest form of reinforcement learning, with reward = 1 if the can
ends up in the basket and 0 otherwise:
    repeat for a few rounds:
        1. run the current policy from random can positions, with a little action
           noise so it tries slightly different things (exploration)
        2. keep only the rollouts that succeeded (reward 1)
        3. retrain the policy on the original demos + those successes
The hope is that rollouts which worked by luck teach the policy something new.
(Also known as filtered behaviour cloning or self-imitation learning.)
In my runs it made the policy worse; the README explains why.

Usage (libero env, from the repo root):
    python policy/self_improve.py --policy ~/results/policy/policy.pt \
        --demos ~/results/replay/demos --out ~/results/self_improve --rounds 3
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "sim"))
import libero_utils as U          # noqa: E402
from train_bc import Policy       # noqa: E402

MAX_STEPS = 1000   # LIBERO episode limit; replays of slow human demos can take 600+ steps
# try the can a bit beyond the original training region, so the policy can grow it
X_RANGE = (U.CAN_X_RANGE[0] - 0.05, U.CAN_X_RANGE[1] + 0.03)
Y_RANGE = (U.CAN_Y_RANGE[0] - 0.04, U.CAN_Y_RANGE[1] + 0.03)


def rollout(env, obs, policy, noise, rng):
    """One episode. Returns (success, states, executed actions)."""
    states, actions = [], []
    policy.reset()
    for _ in range(MAX_STEPS):
        a = policy.act(U.lowdim(obs))
        a[:3] = np.clip(a[:3] + rng.normal(0, noise, 3), -1, 1)
        states.append(U.lowdim(obs))
        actions.append(a)
        obs, _, _, _ = env.step(a)
        if U.success(env):
            return True, np.array(states), np.array(actions)
    return False, None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, help="starting policy from train_bc.py")
    ap.add_argument("--demos", required=True, help="original replay demos")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--episodes", type=int, default=60, help="rollouts per round")
    ap.add_argument("--noise", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    out = os.path.expanduser(args.out)

    # all training data lives in one folder: original demos + every success found
    data = os.path.join(out, "demos")
    os.makedirs(data, exist_ok=True)
    for f in os.listdir(os.path.expanduser(args.demos)):
        shutil.copy(os.path.join(os.path.expanduser(args.demos), f), data)

    env = U.make_env(render=False)
    policy_path = os.path.expanduser(args.policy)
    history = []
    for r in range(1, args.rounds + 1):
        policy = Policy.load(policy_path)
        wins = 0
        for e in range(args.episodes):
            env.seed(int(rng.integers(1e6)))
            env.reset()
            obs = U.settle(env)
            xy = np.array([rng.uniform(*X_RANGE), rng.uniform(*Y_RANGE)])
            if not U.spot_is_free(obs, xy):
                continue
            obs = U.set_can_xy(env, xy)
            ok, states, actions = rollout(env, obs, policy, args.noise, rng)
            if ok:
                wins += 1
                np.savez(os.path.join(data, f"self_round{r}_{e:03d}.npz"),
                         success=True, can_xy=xy, states=states, actions=actions)
        history.append(dict(round=r, successes=wins, episodes=args.episodes))
        print(f"round {r}: {wins}/{args.episodes} noisy rollouts succeeded -> added to the data",
              flush=True)

        policy_dir = os.path.join(out, f"policy_round{r}")
        subprocess.run([sys.executable, os.path.join(HERE, "train_bc.py"),
                        "--demos", data, "--out", policy_dir], check=True)
        policy_path = os.path.join(policy_dir, "policy.pt")

    with open(os.path.join(out, "history.json"), "w") as f:
        json.dump(history, f, indent=1)
    print(f"\nfinal policy: {policy_path}")
    print("evaluate it with: python policy/evaluate.py --policy", policy_path,
          "--out", os.path.join(out, "eval"))


if __name__ == "__main__":
    main()
