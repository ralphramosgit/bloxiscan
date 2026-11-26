"""DTU MVS 2014 evaluation protocol, re-implemented from its description.

The official protocol (MATLAB, and the Python port DTUeval-python by
J. Zhang, MIT license, https://github.com/jzhangbs/DTUeval-python) works like:

1. Densely sample the predicted mesh, then thin it so no two points are closer
   than `density` mm (default 0.2).
2. Keep predicted points inside the scan's bounding box BB (with a margin of
   `patch` mm below and 2*`patch` above), then keep those whose voxel in the
   observation mask ObsMask is set (= the region the scanner actually saw).
3. Keep ground-truth (STL) points above the table plane P.
4. accuracy = mean distance pred(in mask) -> STL, completeness = mean distance
   STL(above plane) -> pred(in box); distances >= max_dist (20 mm) are ignored.
   overall = (accuracy + completeness) / 2.

Validate this file against DTUeval-python on one scan before trusting it.
Inputs must be in DTU's world frame, in millimetres.
"""
import os

import numpy as np
from scipy.io import loadmat
from scipy.spatial import cKDTree

from .metrics import sample_surface, triangle_areas


def load_obsmask(dtu_dir, scan):
    m = loadmat(os.path.join(dtu_dir, "ObsMask", f"ObsMask{scan}_10.mat"))
    return m["ObsMask"].astype(bool), m["BB"].astype(np.float64), float(np.asarray(m["Res"]).ravel()[0])


def load_plane(dtu_dir, scan):
    return np.asarray(loadmat(os.path.join(dtu_dir, "ObsMask", f"Plane{scan}.mat"))["P"],
                      dtype=np.float64).reshape(4)


def load_stl(dtu_dir, scan):
    import open3d as o3d
    path = os.path.join(dtu_dir, "Points", "stl", f"stl{scan:03d}_total.ply")
    return np.asarray(o3d.io.read_point_cloud(path).points)


def thin_points(points, radius, rng=0):
    """Greedy thinning: visit points in random order, keep a point and discard
    every other point within `radius` of it."""
    rng = np.random.default_rng(rng)
    pts = points[rng.permutation(len(points))]
    tree = cKDTree(pts)
    removed = np.zeros(len(pts), dtype=bool)
    keep = []
    for i in range(len(pts)):
        if removed[i]:
            continue
        keep.append(i)
        removed[tree.query_ball_point(pts[i], radius)] = True
    return pts[keep]


def dense_mesh_samples(V, F, density=0.2, oversample=2.0, rng=0):
    """About `oversample` points per density^2 of surface, plus the vertices."""
    n = int(oversample * triangle_areas(V, F).sum() / density ** 2)
    return np.vstack([sample_surface(V, F, max(n, 1), rng), V])


def mask_points(points, obsmask, bb, res, patch=60.0):
    """Return (points inside the padded bounding box, those also inside ObsMask)."""
    inbox = np.all((points >= bb[0] - patch) & (points < bb[1] + 2 * patch), axis=1)
    data_in = points[inbox]
    grid = np.round((data_in - bb[0]) / res).astype(np.int64)
    ingrid = np.all((grid >= 0) & (grid < np.array(obsmask.shape)), axis=1)
    g = grid[ingrid]
    in_obs = obsmask[g[:, 0], g[:, 1], g[:, 2]]
    return data_in, data_in[ingrid][in_obs]


def above_plane(points, plane):
    return points[points @ plane[:3] + plane[3] > 0]


def evaluate_dtu(V, F, scan, dtu_dir, density=0.2, max_dist=20.0, patch=60.0, rng=0):
    obsmask, bb, res = load_obsmask(dtu_dir, scan)
    stl = load_stl(dtu_dir, scan)
    plane = load_plane(dtu_dir, scan)
    pred = thin_points(dense_mesh_samples(V, F, density, rng=rng), density, rng=rng)
    data_in, data_obs = mask_points(pred, obsmask, bb, res, patch)
    stl_up = above_plane(stl, plane)
    d2s = cKDTree(stl).query(data_obs)[0]
    s2d = cKDTree(data_in).query(stl_up)[0]
    acc = float(d2s[d2s < max_dist].mean())
    comp = float(s2d[s2d < max_dist].mean())
    return {"accuracy_mm": acc, "completeness_mm": comp, "chamfer_mm": 0.5 * (acc + comp),
            "n_pred_points": int(len(data_obs)), "n_gt_points": int(len(stl_up))}
