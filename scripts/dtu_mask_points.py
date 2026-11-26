"""Keep only the DTU object: drop points that fall outside the object masks.

    python scripts/dtu_mask_points.py --scan-dir dtu/scan24 \
        --input work/scan24/dense/fused.ply --out work/scan24/fused_masked.ply

COLMAP's dense cloud also contains the table and background. Meshing those
would waste the triangle budget, so we project every point into each view with
P_i = world_mat_i @ scale_mat_i (from cameras.npz, same frame as the provided
COLMAP poses) and keep a point only if it lands inside the (slightly grown)
object mask in every view where it is visible. Same rule 2DGS uses for DTU.
Course concepts: projection matrices, morphological dilation.
"""
import argparse
import glob
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan-dir", required=True, help="folder with cameras.npz and mask/")
    ap.add_argument("--input", required=True, help="point cloud .ply")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dilate", type=int, default=24, help="mask growth in pixels")
    a = ap.parse_args()

    cams = np.load(os.path.join(a.scan_dir, "cameras.npz"))
    masks = sorted(glob.glob(os.path.join(a.scan_dir, "mask", "*.png")))
    if not masks:
        sys.exit(f"No masks in {a.scan_dir}/mask")
    pcd = o3d.io.read_point_cloud(a.input)
    X = np.asarray(pcd.points)
    Xh = np.c_[X, np.ones(len(X))]
    keep = np.ones(len(X), dtype=bool)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * a.dilate + 1, 2 * a.dilate + 1))

    for i, path in enumerate(masks):
        P = (cams[f"world_mat_{i}"] @ cams[f"scale_mat_{i}"])[:3, :4]
        m = cv2.imread(path, cv2.IMREAD_GRAYSCALE) > 127
        m = cv2.dilate(m.astype(np.uint8), kernel) > 0
        h, w = m.shape
        x = Xh @ P.T
        front = x[:, 2] > 0
        u = np.full(len(X), -1, dtype=np.int64)
        v = np.full(len(X), -1, dtype=np.int64)
        u[front] = np.round(x[front, 0] / x[front, 2]).astype(np.int64)
        v[front] = np.round(x[front, 1] / x[front, 2]).astype(np.int64)
        inside = front & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        ok = np.ones(len(X), dtype=bool)
        ok[inside] = m[v[inside], u[inside]]
        keep &= ok

    out = pcd.select_by_index(np.flatnonzero(keep).tolist())
    o3d.io.write_point_cloud(a.out, out)
    print(f"Kept {keep.sum():,} of {len(X):,} points ({keep.mean():.0%}) using {len(masks)} masks; "
          f"wrote {a.out}")


if __name__ == "__main__":
    main()
