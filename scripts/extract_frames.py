"""Stage 2: video -> sharp still frames for COLMAP.

    python scripts/extract_frames.py sneaker.mov data/sneaker/images --fps 3

Samples the video at (fps x window) frames per second, scores each sample with
our own Laplacian-variance blur measure (bloxiscan/blur.py), keeps the sharpest
sample in every window, then drops frames far below the median sharpness.
Writes JPGs plus scores.csv (for the blur-filter ablation in the report).
Use --no-filter to keep every sample at --fps (the "filter off" ablation).
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bloxiscan.blur import select_sharp, sharpness  # noqa: E402


def resize_max(img, max_side):
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s >= 1:
        return img
    return cv2.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("out_dir")
    ap.add_argument("--fps", type=float, default=3.0, help="frames per second to keep")
    ap.add_argument("--window", type=int, default=2, help="samples per kept frame")
    ap.add_argument("--min-ratio", type=float, default=0.35,
                    help="drop frames below this fraction of the median sharpness")
    ap.add_argument("--max-side", type=int, default=1600, help="resize saved frames (pixels)")
    ap.add_argument("--score-side", type=int, default=960, help="resolution used for scoring")
    ap.add_argument("--no-filter", action="store_true", help="ablation: keep every sample")
    a = ap.parse_args()

    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        sys.exit(f"Could not open {a.video}")
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    window = 1 if a.no_filter else max(1, a.window)
    step = max(1, int(round(native_fps / (a.fps * window))))

    os.makedirs(a.out_dir, exist_ok=True)
    samples, scores = [], []
    idx = 0
    while True:
        if not cap.grab():
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if ok:
                scores.append(sharpness(resize_max(frame, a.score_side)))
                samples.append((idx, resize_max(frame, a.max_side)))
        idx += 1
    cap.release()
    if not samples:
        sys.exit("No frames read from the video.")

    keep = list(range(len(samples))) if a.no_filter else select_sharp(scores, window, a.min_ratio)
    keep_set = set(keep)
    for k in keep:
        frame_idx, img = samples[k]
        cv2.imwrite(os.path.join(a.out_dir, f"frame_{frame_idx:06d}.jpg"), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])

    with open(os.path.join(Path(a.out_dir).parent, "scores.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_s", "sharpness", "kept"])
        for k, ((frame_idx, _), s) in enumerate(zip(samples, scores)):
            w.writerow([frame_idx, round(frame_idx / native_fps, 3), round(s, 2), int(k in keep_set)])

    print(f"{idx} video frames, {len(samples)} sampled, {len(keep)} kept -> {a.out_dir}")


if __name__ == "__main__":
    main()
