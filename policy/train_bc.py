"""
Step 3: behaviour cloning on the robot demos from step 2.

A small MLP learns to copy the controller: given the current state, output the
gripper's velocity command (x, y, z) and whether the gripper should be closed.
Only replays that succeeded are used.

Validation holds out whole human clips (all replays made from them), not random
time steps. Random steps from the same trajectory are nearly identical to their
neighbours, so they would make the validation loss look better than it is.

Usage (libero env, from the repo root):
    python policy/train_bc.py --demos ~/results/replay/demos --out ~/results/policy
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
import torch.nn as nn


def features(states):
    """(N, 11) raw state -> (N, 14) input. Positions relative to the gripper make the
    task the same wherever the can is, which helps the network generalise."""
    s = np.asarray(states, dtype=np.float32)
    gripper, fingers, can, basket = s[:, 0:3], s[:, 3:5], s[:, 5:8], s[:, 8:11]
    return np.concatenate([gripper, fingers, can - gripper, basket - gripper, can], 1)


class Policy(nn.Module):
    """chunk = how many future actions the network predicts at once.
    chunk = 1 is plain behaviour cloning. With chunk > 1 (action chunking, as in ACT),
    the robot plans several steps ahead and carries out the first half of the plan
    before asking the network again. Committing to a short plan stops it from
    averaging between demos that went different ways at every single step."""

    def __init__(self, hidden=256, chunk=1):
        super().__init__()
        self.chunk = chunk
        self.execute = max(1, chunk // 2)   # how many planned steps to carry out before re-planning
        self.net = nn.Sequential(
            nn.Linear(14, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 4 * chunk))
        # input normalisation, set from the training data and saved with the weights
        self.register_buffer("mean", torch.zeros(14))
        self.register_buffer("std", torch.ones(14))
        self.plan = []

    @classmethod
    def load(cls, path):
        weights = torch.load(os.path.expanduser(path), map_location="cpu", weights_only=True)
        # the last layer has 4 outputs per planned step, so its size gives the chunk length
        policy = cls(chunk=weights["net.6.weight"].shape[0] // 4)
        policy.load_state_dict(weights)
        return policy.eval()

    def forward(self, x):
        out = self.net((x - self.mean) / self.std).view(len(x), self.chunk, 4)
        return torch.tanh(out[..., :3]), out[..., 3]  # velocities in [-1, 1], gripper logits

    def reset(self):
        self.plan = []

    @torch.no_grad()
    def act(self, state):
        if not self.plan:
            velocity, grip = self(torch.as_tensor(features(state[None])))
            for k in range(self.execute):
                action = np.zeros(7, dtype=np.float32)
                action[:3] = velocity[0, k].numpy()
                action[6] = 1.0 if grip[0, k].item() > 0 else -1.0
                self.plan.append(action)
        return self.plan.pop(0)


def future_actions(actions, chunk):
    """(T, 4) actions -> (T, chunk, 4): at each step, the next `chunk` actions
    (the last action is repeated past the end of the demo)."""
    idx = np.minimum(np.arange(len(actions))[:, None] + np.arange(chunk), len(actions) - 1)
    return actions[idx]


def load_demos(folder, max_replays=None, chunk=1):
    """Returns {clip name: (inputs, targets)} using successful replays only.
    max_replays=1 keeps only the replay at the original can position (no augmentation)."""
    data = {}
    files = sorted(glob.glob(os.path.join(os.path.expanduser(folder), "*.npz")))
    for f in files:
        clip, index = os.path.splitext(os.path.basename(f))[0].rsplit("_", 1)
        if max_replays is not None and int(index) >= max_replays:
            continue
        d = np.load(f)
        if not bool(d["success"]):
            continue
        x = features(d["states"])
        y = future_actions(d["actions"][:, [0, 1, 2, 6]], chunk)
        if clip in data:
            x, y = np.concatenate([data[clip][0], x]), np.concatenate([data[clip][1], y])
        data[clip] = (x, y)
    return data, len(files)


def loss_fn(policy, x, y):
    velocity, grip = policy(x)
    move = ((velocity - y[..., :3]) ** 2).mean()
    gripper = nn.functional.binary_cross_entropy_with_logits(grip, (y[..., 3] > 0).float())
    return move, gripper


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--noise", type=float, default=0.005,
                    help="std (metres) of noise added to positions while training")
    ap.add_argument("--val-clips", type=int, default=2, help="human clips held out for validation")
    ap.add_argument("--clips", type=int, help="train on only this many human clips (data ablation)")
    ap.add_argument("--max-replays", type=int, help="use only the first N replays of each clip "
                    "(1 = no augmentation)")
    ap.add_argument("--chunk", type=int, default=1,
                    help="predict this many future actions at once (1 = plain BC)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    data, n_files = load_demos(args.demos, args.max_replays, args.chunk)
    clips = list(rng.permutation(sorted(data)))
    n_val = min(args.val_clips, len(clips) - 1)
    val_clips = clips[:n_val]
    train_clips = clips[n_val:n_val + args.clips] if args.clips else clips[n_val:]
    xt = torch.tensor(np.concatenate([data[c][0] for c in train_clips]))
    yt = torch.tensor(np.concatenate([data[c][1] for c in train_clips]))
    xv = torch.tensor(np.concatenate([data[c][0] for c in val_clips]))
    yv = torch.tensor(np.concatenate([data[c][1] for c in val_clips]))
    print(f"{len(clips)} clips, {n_files} replay files (only successful ones are used)")
    print(f"train: {len(train_clips)} clips, {len(xt)} steps | validation: {val_clips}, {len(xv)} steps")

    policy = Policy(chunk=args.chunk)
    policy.mean.copy_(xt.mean(0))
    policy.std.copy_(xt.std(0) + 1e-3)
    opt = torch.optim.AdamW(policy.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    position_cols = [0, 1, 2] + list(range(5, 14))  # everything except the finger opening

    log = []
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, len(xt), (args.batch,))
        x = xt[idx].clone()
        x[:, position_cols] += args.noise * torch.randn(len(idx), len(position_cols))
        move, gripper = loss_fn(policy, x, yt[idx])
        loss = move + 0.5 * gripper
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()

        if step == 1 or step % 500 == 0:
            with torch.no_grad():
                v_move, v_grip = loss_fn(policy, xv, yv)
                acc = ((policy(xv)[1] > 0) == (yv[..., 3] > 0)).float().mean().item()
            log.append(dict(step=step, train_move=move.item(), val_move=v_move.item(),
                            train_grip=gripper.item(), val_grip=v_grip.item(), val_grip_acc=acc))
            print(f"step {step:5d} | move loss train {move.item():.4f} val {v_move.item():.4f}"
                  f" | gripper acc val {acc:.3f}", flush=True)

    out = os.path.expanduser(args.out)
    os.makedirs(out, exist_ok=True)
    torch.save(policy.state_dict(), os.path.join(out, "policy.pt"))
    with open(os.path.join(out, "train_log.json"), "w") as f:
        json.dump(dict(train_clips=train_clips, val_clips=val_clips, args=vars(args), log=log), f, indent=1)
    plot_curves(log, os.path.join(out, "training_curves.png"))
    print(f"saved {os.path.join(out, 'policy.pt')}")


def plot_curves(log, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    steps = [r["step"] for r in log]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(steps, [r["train_move"] for r in log], label="train (seen clips)")
    ax.plot(steps, [r["val_move"] for r in log], label="validation (held-out clips)")
    ax.set_xlabel("training step")
    ax.set_ylabel("velocity error (MSE)")
    ax.set_yscale("log")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
