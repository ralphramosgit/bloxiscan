"""Visual hull by space carving (own implementation).

Course concepts: pinhole projection, silhouettes, voxel grids, robust voting.

A smooth single-colour object (a bottle) gives feature matching and stereo
almost nothing to lock onto, so its point cloud has big gaps. Its outline,
though, is easy to cut out in every frame (scripts/make_masks.swift). Every
point of the object projects inside the object's outline in every view, so
the object lies inside the intersection of the silhouettes' viewing cones:
the visual hull (Laurentini 1994). Space carving builds it on a voxel grid:
  1. Fill a box standing on the table with voxels.
  2. Project every voxel centre into every view (x = K [R | t] X).
  3. A view votes "outside" when the voxel lands outside the silhouette;
     views that do not see the voxel (off the image) do not vote.
  4. Score = fraction of the views that see the voxel which put it inside
     the silhouette. Keep voxels scoring at least `keep_ratio`, so one bad
     mask cannot punch a hole in the object.
  5. A coarse pass finds the object, a fine pass carves it in detail.
The surface is where the (lightly smoothed) score crosses keep_ratio. Using
the score rather than a hard in/out cut matters: every mask is off by a pixel
or two somewhere, and hard cuts turn that into grooves along the object.
Normals are minus the score's gradient, so they point outward.

Limits: dents no outline reveals (concavities) stay filled in, and a flat top
seen only from low angles gets a shallow cone on it (height about
radius * tan(lowest camera elevation)).
"""
import numpy as np
from scipy import ndimage


def table_basis(up):
    """Rows e1, e2, up: an orthonormal frame with the table normal as 3rd axis."""
    up = np.asarray(up, float) / np.linalg.norm(up)
    e1 = np.cross(up, [1.0, 0.0, 0.0])
    if np.linalg.norm(e1) < 0.1:
        e1 = np.cross(up, [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    return np.stack([e1, np.cross(up, e1), up])


def carve_scores(points, views, floor=0.5, min_seen=3):
    """Silhouette score of 3D points: the fraction of the views that see a
    point (it projects onto the image) which put it inside the mask.

    views: list of (P 3x4, mask bool HxW), mask True on the object.
    Points seen by fewer than min_seen views score 0.
    A point with more 'outside' votes than (1 - floor) * len(views) can never
    score above `floor`, so it is dropped early and scores 0 (most of the box
    goes after a few views, which makes carving fast).
    """
    X = np.c_[np.asarray(points, float), np.ones(len(points))]
    seen = np.zeros(len(X), np.int32)
    outside = np.zeros(len(X), np.int32)
    budget = (1.0 - floor) * len(views)
    active = np.arange(len(X))
    for P, mask in views:
        x = X[active] @ np.asarray(P, float).T
        h, w = mask.shape
        with np.errstate(divide="ignore", invalid="ignore"):
            u = np.floor(x[:, 0] / x[:, 2])       # COLMAP: pixel i spans [i, i+1)
            v = np.floor(x[:, 1] / x[:, 2])
        vis = (x[:, 2] > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        idx = active[vis]
        seen[idx] += 1
        outside[idx[~mask[v[vis].astype(np.int64), u[vis].astype(np.int64)]]] += 1
        active = active[outside[active] <= budget]
    score = np.zeros(len(X))
    ok = active[seen[active] >= min_seen]
    score[ok] = 1.0 - outside[ok] / seen[ok]
    return score


def _grid(lo, hi, n_long):
    """Voxel centres filling the box [lo, hi], n_long voxels along its longest side."""
    voxel = float(np.max(hi - lo)) / n_long
    shape = np.maximum(np.ceil((hi - lo) / voxel - 1e-9).astype(int), 1)
    axes = [lo[i] + (np.arange(shape[i]) + 0.5) * voxel for i in range(3)]
    return np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3), shape, voxel


def _largest_blob(occ):
    lab, n = ndimage.label(occ)
    if n <= 1:
        return occ
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return lab == sizes.argmax()


def surface_samples(score, level, lo, voxel, sigma=1.0):
    """Points on the surface score = level of a voxel grid (box-local
    coordinates), with outward normals.

    The score is Gaussian-smoothed (sigma in voxels) first. Each voxel at the
    edge of the largest region scoring >= level is then moved along its normal
    to where the score crosses `level` (one Newton step, s - level = |grad s| t).
    Without that step the points sit on the voxel grid, and a round object
    gets a staircase that shows up as ridges along the grid axes.
    """
    smooth = ndimage.gaussian_filter(np.pad(score, 2), sigma)
    solid = _largest_blob(smooth >= level)
    border = solid & ~ndimage.binary_erosion(solid)
    grad = np.stack(np.gradient(smooth), -1)[border]
    g = np.maximum(np.linalg.norm(grad, axis=1, keepdims=True), 1e-12)
    nrm = -grad / g
    step = np.clip((smooth[border][:, None] - level) / g, 0.0, 1.5)   # in voxels
    ijk = np.argwhere(border) - 2 + 0.5 + step * nrm                  # undo the padding
    return lo + ijk * voxel, nrm


def visual_hull(views, up, d, axis_point, half_width, height,
                res=160, coarse=64, keep_ratio=0.9, min_seen=None):
    """Carve the object standing on the table plane up . x + d = 0.

    axis_point: a point on (or near) the object's vertical axis.
    half_width, height: search box around that axis, above the table.
    res: voxels along the longest side of the object's box in the fine pass.
    Returns (points (N,3), outward normals (N,3), info dict), world coords.
    """
    B = table_basis(up)
    up = B[2]
    origin = axis_point - (axis_point @ up + d) * up          # axis foot on the table
    if min_seen is None:
        min_seen = max(3, len(views) // 2)

    def carve(lo, hi, n):
        L, shape, voxel = _grid(lo, hi, n)
        floor = max(0.0, keep_ratio - 0.3)       # keep the score's ramp near the surface
        return carve_scores(origin + L @ B, views, floor, min_seen).reshape(shape), voxel

    # coarse pass: forgiving threshold, only to find the object's box
    lo = np.array([-half_width, -half_width, 0.0])
    hi = np.array([half_width, half_width, height])
    score, voxel = carve(lo, hi, coarse)
    occ = _largest_blob(score >= min(keep_ratio, 0.75))
    if not occ.any():
        raise ValueError("carving removed everything: check that the masks match "
                         "the images the cameras were computed for")
    ijk = np.argwhere(occ)
    # object reaching the sides or top of the search box = box too small
    touches = bool(np.any(ijk[:, :2].min(0) == 0)
                   or np.any(ijk[:, :2].max(0) == np.array(occ.shape[:2]) - 1)
                   or ijk[:, 2].max() == occ.shape[2] - 1)
    lo2 = lo + ijk.min(0) * voxel - 3 * voxel
    hi2 = lo + (ijk.max(0) + 1) * voxel + 3 * voxel
    lo2[2] = 0.0                                               # stand on the table

    # fine pass
    score, voxel = carve(lo2, hi2, res)
    pts, nrm = surface_samples(score, keep_ratio, lo2, voxel)
    info = {"views": len(views), "voxel": voxel, "grid": list(score.shape),
            "voxels_kept": int((score >= keep_ratio).sum()), "touches_box": touches,
            "height": float(np.ptp(pts[:, 2]) + voxel)}
    return origin + pts @ B, nrm @ B, info
