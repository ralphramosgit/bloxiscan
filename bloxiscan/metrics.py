"""Geometry accuracy metrics (own implementation).

* sample_surface: uniform random points on a triangle mesh
* chamfer: accuracy (pred -> gt), completeness (gt -> pred) and their mean
* fscore: precision / recall / F at a distance threshold tau

Nearest neighbours use SciPy's KD-tree (a data structure, not the metric).
"""
import numpy as np
from scipy.spatial import cKDTree


def triangle_areas(V, F):
    return 0.5 * np.linalg.norm(
        np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)


def sample_surface(V, F, n, rng=None):
    """n points uniformly distributed over the mesh surface.

    Pick faces with probability proportional to area, then a uniform point in
    each triangle with  p = (1 - sqrt(r1)) a + sqrt(r1)(1 - r2) b + sqrt(r1) r2 c
    (the sqrt makes the density uniform instead of piling up at vertex a).
    """
    rng = np.random.default_rng(rng)
    V = np.asarray(V, float)
    F = np.asarray(F, np.int64)
    area = triangle_areas(V, F)
    face = rng.choice(len(F), size=n, p=area / area.sum())
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    a, b, c = V[F[face, 0]], V[F[face, 1]], V[F[face, 2]]
    return (1 - s)[:, None] * a + (s * (1 - r2))[:, None] * b + (s * r2)[:, None] * c


def nn_dist(src, dst):
    """Distance from every point in src to its nearest point in dst."""
    return cKDTree(dst).query(src, k=1)[0]


def chamfer(pred, gt, max_dist=None):
    """Chamfer distance between point sets (same units as the input).

    accuracy     = mean over pred points of distance to nearest gt point
    completeness = mean over gt points of distance to nearest pred point
    chamfer      = (accuracy + completeness) / 2
    With max_dist, distances above it are left out of the means (DTU style).
    """
    d_pg = nn_dist(pred, gt)
    d_gp = nn_dist(gt, pred)
    if max_dist is not None:
        d_pg = d_pg[d_pg < max_dist]
        d_gp = d_gp[d_gp < max_dist]
    acc, comp = float(d_pg.mean()), float(d_gp.mean())
    return {"accuracy": acc, "completeness": comp, "chamfer": 0.5 * (acc + comp)}


def fscore(pred, gt, tau):
    """Precision = share of pred within tau of gt; recall = share of gt within
    tau of pred; F = 2PR / (P + R)."""
    precision = float((nn_dist(pred, gt) < tau).mean())
    recall = float((nn_dist(gt, pred) < tau).mean())
    f = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "fscore": f, "tau": tau}


def mesh_vs_points(V, F, gt_points, tau, n_samples=200_000, rng=0):
    """Sample the mesh, then compute Chamfer + F-score against gt points."""
    pred = sample_surface(V, F, n_samples, rng)
    out = chamfer(pred, gt_points)
    out.update(fscore(pred, gt_points, tau))
    return out
