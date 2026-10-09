"""
Step 1: track the hand in each clip and find when it grasps and releases the can.

For every video:
  1. MediaPipe Hands gives 21 hand keypoints per frame (in pixels).
  2. Short tracking dropouts are filled in and the keypoints are smoothed.
  3. Grip aperture = thumb-tip to index-tip distance / hand size. Dividing by hand
     size means the value doesn't change when the hand moves towards the camera.
  4. Holding the can = the longest stretch of low aperture during which the hand
     also moves a lot (it is carrying something). Its start is the grasp, its end
     the release.

The can is never detected directly: it is wherever the hand closed. The bin is
wherever the hand opened again.

Outputs in --out: tracks/<clip>.npz, plots/<clip>.png, overlays/<clip>.mp4, summary.csv

Usage (hands env):
    python track_hands.py --videos ~/hand_demos --out ~/hand_demos_out
    python track_hands.py --videos ~/hand_demos --out ~/hand_demos_out --reuse   # skip MediaPipe
"""
import argparse
import csv
import glob
import os

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# MediaPipe keypoint indices
WRIST, THUMB_TIP, INDEX_BASE, INDEX_TIP, MIDDLE_BASE, PINKY_BASE = 0, 4, 5, 8, 9, 17
PALM = [WRIST, INDEX_BASE, MIDDLE_BASE, PINKY_BASE]
BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
         (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18),
         (18, 19), (19, 20), (0, 17)]


def read_frames(path, max_width=1280):
    """Yield video frames, downscaled to at most max_width pixels wide."""
    cap = cv2.VideoCapture(path)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[1] > max_width:
            s = max_width / frame.shape[1]
            frame = cv2.resize(frame, None, fx=s, fy=s)
        yield frame
    cap.release()


def track_video(path):
    """Keypoints (T, 21, 2) in pixels, NaN where no hand was found."""
    import mediapipe as mp
    hands = mp.solutions.hands.Hands(max_num_hands=1, model_complexity=1,
                                     min_detection_confidence=0.5, min_tracking_confidence=0.5)
    keypoints = []
    for frame in read_frames(path):
        h, w = frame.shape[:2]
        result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if result.multi_hand_landmarks:
            keypoints.append([[p.x * w, p.y * h] for p in result.multi_hand_landmarks[0].landmark])
        else:
            keypoints.append(np.full((21, 2), np.nan))
    hands.close()
    fps = cv2.VideoCapture(path).get(cv2.CAP_PROP_FPS) or 30.0
    return np.array(keypoints, dtype=float), fps


# ------------------------------------------------------------------ signal helpers
def runs(mask):
    """(start, end) index pairs, end exclusive, of each stretch where mask is True."""
    edges = np.diff(np.r_[0, mask.astype(int), 0])
    return list(zip(np.where(edges == 1)[0], np.where(edges == -1)[0]))


def fill_short_gaps(x, max_gap):
    """Linearly interpolate NaN gaps of up to max_gap frames. x has shape (T, D)."""
    x = x.copy()
    valid = ~np.isnan(x[:, 0])
    if valid.sum() < 2:
        return x
    t = np.arange(len(x))
    for start, end in runs(~valid):
        if start > 0 and end < len(x) and end - start <= max_gap:   # never extrapolate
            for d in range(x.shape[1]):
                x[start:end, d] = np.interp(t[start:end], t[valid], x[valid, d])
    return x


def smooth(x, k):
    """Moving average over k frames that ignores NaNs. x has shape (T, D)."""
    out = np.full_like(x, np.nan)
    kernel = np.ones(k)
    for d in range(x.shape[1]):
        ok = ~np.isnan(x[:, d])
        total = np.convolve(np.where(ok, x[:, d], 0.0), kernel, mode="same")
        count = np.convolve(ok.astype(float), kernel, mode="same")
        out[ok, d] = total[ok] / count[ok]
    return out


def otsu_threshold(values, bins=64):
    """The cut that best splits values into two groups (Otsu's method)."""
    hist, edges = np.histogram(values, bins=bins)
    centres = (edges[1:] + edges[:-1]) / 2
    n_low = np.cumsum(hist)
    n_high = n_low[-1] - n_low
    mean_low = np.cumsum(hist * centres) / np.maximum(n_low, 1)
    mean_high = (np.sum(hist * centres) - np.cumsum(hist * centres)) / np.maximum(n_high, 1)
    spread = n_low * n_high * (mean_low - mean_high) ** 2
    return float(centres[np.argmax(spread[:-1])])


# ------------------------------------------------------------------ grasp detection
def analyse(keypoints, fps):
    """Keypoints (T, 21, 2) -> aperture signal plus the grasp and release frames (-1 if none)."""
    T = len(keypoints)
    k = fill_short_gaps(keypoints.reshape(T, -1), max_gap=int(0.3 * fps))
    k = smooth(k, int(0.1 * fps) | 1).reshape(T, 21, 2)        # | 1 makes the window odd

    palm = k[:, PALM].mean(1)                                   # stand-in for the gripper
    grip_point = (k[:, THUMB_TIP] + k[:, INDEX_TIP]) / 2       # where the object sits
    hand_size = np.linalg.norm(k[:, MIDDLE_BASE] - k[:, WRIST], axis=1)
    aperture = np.linalg.norm(k[:, THUMB_TIP] - k[:, INDEX_TIP], axis=1) / hand_size
    res = dict(palm=palm, grip_point=grip_point, hand_size=hand_size, aperture=aperture,
               grasp=-1, release=-1, carry_px=0.0, threshold=np.nan)

    seen = ~np.isnan(aperture)
    if seen.sum() < fps:   # less than a second of hand
        return res

    # In my recordings (side grip) the aperture is lowest while holding the can.
    # Otsu's threshold splits "low" from "high" without a hand-tuned constant, and a
    # small band around it (hysteresis) stops jitter from flipping the state.
    threshold = otsu_threshold(aperture[seen])
    band = 0.05 * np.ptp(np.percentile(aperture[seen], [5, 95]))
    holding = np.zeros(T, bool)
    state = False
    for t in range(T):
        if seen[t]:
            if aperture[t] < threshold - band:
                state = True
            elif aperture[t] > threshold + band:
                state = False
        holding[t] = state

    # The real hold is the low stretch in which the hand travels furthest.
    best_distance = np.nanmedian(hand_size)   # must move at least one hand length
    for start, end in runs(holding):
        if end - start < 0.5 * fps:
            continue
        frames = start + np.where(seen[start:end])[0]
        distance = np.linalg.norm(palm[frames[-1]] - palm[frames[0]])
        if distance > best_distance:
            best_distance = distance
            res.update(grasp=int(frames[0]), release=int(frames[-1]), carry_px=float(distance))
    res["threshold"] = threshold
    return res


# ------------------------------------------------------------------ outputs
def write_plot(path, name, res, fps):
    t = np.arange(len(res["aperture"])) / fps
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(t, res["aperture"], color="#2a6fdb", lw=1.8, label="grip aperture")
    ax.axhline(res["threshold"], color="#999", ls="--", lw=1, label="holding threshold")
    if res["grasp"] >= 0:
        ax.axvspan(res["grasp"] / fps, res["release"] / fps, color="#e4572e", alpha=0.15,
                   label="holding the can")
    ax.set(title=name, xlabel="time (s)", ylabel="thumb–index / hand size")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def write_overlay(video, path, keypoints, res, fps):
    writer = None
    g, r = res["grasp"], res["release"]
    for t, frame in enumerate(read_frames(video)):
        if t >= len(keypoints):
            break
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        holding = 0 <= g <= t <= r
        colour = (0, 0, 255) if holding else (0, 200, 0)    # red while holding
        p = keypoints[t]
        if not np.isnan(p[0, 0]):
            for a, b in BONES:
                cv2.line(frame, tuple(p[a].astype(int)), tuple(p[b].astype(int)), colour, 2)
        if g >= 0:
            for idx, label in ((g, "grasp"), (r, "release")):
                xy = tuple(res["grip_point"][idx].astype(int))
                cv2.circle(frame, xy, 12, (255, 0, 255), 2)
                cv2.putText(frame, label, (xy[0] + 14, xy[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 0, 255), 2)
        text = f"t={t / fps:5.2f}s  {'HOLDING' if holding else 'open'}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)
        writer.write(frame)
    if writer:
        writer.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reuse", action="store_true", help="reuse saved keypoints, redo only the analysis")
    ap.add_argument("--no-overlay", action="store_true", help="skip the overlay videos")
    args = ap.parse_args()

    folder, out = os.path.expanduser(args.videos), os.path.expanduser(args.out)
    videos = sorted(f for f in glob.glob(os.path.join(folder, "*"))
                    if f.lower().endswith((".mp4", ".mov")))
    if not videos:
        raise SystemExit(f"no videos in {folder}")
    for sub in ("tracks", "plots", "overlays"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)

    rows = []
    for video in videos:
        name = os.path.splitext(os.path.basename(video))[0]
        saved = os.path.join(out, "tracks", f"{name}.npz")
        if args.reuse and os.path.exists(saved):
            d = np.load(saved)
            keypoints, fps = d["keypoints"], float(d["fps"])
        else:
            print(f"[{name}] tracking ...", flush=True)
            keypoints, fps = track_video(video)

        res = analyse(keypoints, fps)
        np.savez(saved, keypoints=keypoints, fps=fps, **res)
        write_plot(os.path.join(out, "plots", f"{name}.png"), name, res, fps)
        if not args.no_overlay:
            write_overlay(video, os.path.join(out, "overlays", f"{name}.mp4"), keypoints, res, fps)

        found = res["grasp"] >= 0
        seen = 100 * np.mean(~np.isnan(keypoints[:, 0, 0]))
        rows.append(dict(clip=name, seconds=round(len(keypoints) / fps, 1), hand_seen_pct=round(seen, 1),
                         grasp_s=round(res["grasp"] / fps, 2) if found else "",
                         release_s=round(res["release"] / fps, 2) if found else "",
                         carry_px=round(res["carry_px"]), status="ok" if found else "NO GRASP FOUND"))
        print(f"[{name}] hand seen {seen:.0f}% | "
              + (f"grasp {res['grasp'] / fps:.2f}s, release {res['release'] / fps:.2f}s"
                 if found else "no grasp found"), flush=True)

    with open(os.path.join(out, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nsummary: {os.path.join(out, 'summary.csv')}")


if __name__ == "__main__":
    main()
