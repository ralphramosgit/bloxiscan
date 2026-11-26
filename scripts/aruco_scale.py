"""Metric scale from the printed ArUco marker.

    colmap model_converter --input_path work/sneaker/dense/0/sparse \
        --output_path work/sneaker/dense/0/sparse_txt --output_type TXT
    python scripts/aruco_scale.py --model work/sneaker/dense/0/sparse_txt \
        --images work/sneaker/dense/0/images --length 0.100 --out out/sneaker/scale.json

1. Detect the marker's 4 corners in every frame (OpenCV's ArUco detector).
2. Triangulate each corner from all frames with our DLT (bloxiscan/geometry.py),
   dropping frames with large reprojection error.
3. l = mean reconstructed side length;  s = L / l  converts COLMAP units to metres.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bloxiscan.colmap_io import read_model_text  # noqa: E402
from bloxiscan.geometry import triangulate_robust  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="COLMAP text model (undistorted)")
    ap.add_argument("--images", required=True, help="folder with the model's images")
    ap.add_argument("--length", type=float, required=True, help="printed side L in metres")
    ap.add_argument("--dict", default="DICT_4X4_50")
    ap.add_argument("--id", type=int, default=0)
    ap.add_argument("--max-err", type=float, default=3.0, help="px")
    ap.add_argument("--out", default="scale.json")
    a = ap.parse_args()

    cams, imgs = read_model_text(a.model)
    if any(c.has_distortion for c in cams.values()):
        print("warning: model has lens distortion; use the undistorted dense/0/sparse model")
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, a.dict)),
                                  cv2.aruco.DetectorParameters())
    Ps, obs = [], []  # obs[k] = (4, 2) corners in view k
    for im in sorted(imgs.values(), key=lambda i: i.name):
        path = os.path.join(a.images, im.name)
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        corners, ids, _ = det.detectMarkers(gray)
        if ids is None or a.id not in ids.ravel():
            continue
        k = list(ids.ravel()).index(a.id)
        Ps.append(im.P(cams[im.camera_id]))
        obs.append(corners[k].reshape(4, 2))
    print(f"Marker {a.id} found in {len(obs)} of {len(imgs)} frames")
    if len(obs) < 3:
        sys.exit("Need the marker in at least 3 frames.")

    X, errs = [], []
    for c in range(4):
        Xc, kept, e = triangulate_robust(Ps, [o[c] for o in obs], a.max_err)
        X.append(Xc)
        errs.append(float(np.mean(e)))
        print(f"  corner {c}: {len(kept)} views, mean reprojection error {np.mean(e):.2f} px")
    X = np.array(X)
    sides = [float(np.linalg.norm(X[i] - X[(i + 1) % 4])) for i in range(4)]
    diags = [float(np.linalg.norm(X[0] - X[2])), float(np.linalg.norm(X[1] - X[3]))]
    ell = float(np.mean(sides))
    s = a.length / ell
    squareness = float(np.mean(diags) / ell)  # sqrt(2) = 1.414 for a perfect square
    spread = float(np.std(sides) / ell)
    print(f"Reconstructed side l = {ell:.5f} model units; scale s = L / l = {s:.5f}")
    print(f"Checks: diagonal/side = {squareness:.3f} (expect 1.414), side spread = {spread:.1%}")
    with open(a.out, "w") as f:
        json.dump({"scale": s, "printed_side_m": a.length, "reconstructed_side": ell,
                   "sides": sides, "diagonal_over_side": squareness, "side_spread": spread,
                   "views": len(obs), "mean_reproj_px": errs,
                   "corners_model_units": X.tolist()}, f, indent=2)
    print(f"Wrote {a.out}")


if __name__ == "__main__":
    main()
