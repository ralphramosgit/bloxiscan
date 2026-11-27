"""Multi-view geometry, written from scratch with NumPy.

Course concepts: pinhole projection, camera centers, linear triangulation (DLT),
SVD, similarity transforms (Umeyama), and robust fitting with RANSAC.
Reference: Hartley & Zisserman, Multiple View Geometry, 2nd ed., ch. 6, 12.
"""
import numpy as np


# ---------------------------------------------------------------- projection
def projection_matrix(K, R, t):
    """P = K [R | t]  (3x4). R, t map world -> camera coordinates."""
    return np.asarray(K, float) @ np.hstack([np.asarray(R, float),
                                              np.asarray(t, float).reshape(3, 1)])


def project(P, X):
    """Project 3D points X (N,3) with P (3x4). Returns pixels (N,2)."""
    X = np.atleast_2d(np.asarray(X, float))
    Xh = np.hstack([X, np.ones((len(X), 1))])
    x = Xh @ np.asarray(P, float).T
    return x[:, :2] / x[:, 2:3]


def camera_center(P):
    """Camera center C solves P [C; 1] = 0. With P = [M | p4]:  C = -M^-1 p4."""
    P = np.asarray(P, float)
    return -np.linalg.solve(P[:, :3], P[:, 3])


# ------------------------------------------------------------- triangulation
def triangulate_dlt(Ps, xs):
    """Linear triangulation of ONE 3D point seen in N >= 2 views.

    Each view gives two equations (rows of A), from x cross (P X) = 0:
        x * P[2] - P[0]
        y * P[2] - P[1]
    Stack them into A (2N x 4) and solve A X = 0 in the least-squares sense:
    X is the right singular vector of A with the smallest singular value.
    Rows are scaled to unit length, which improves conditioning.
    """
    rows = []
    for P, (x, y) in zip(Ps, xs):
        P = np.asarray(P, float)
        rows.append(x * P[2] - P[0])
        rows.append(y * P[2] - P[1])
    A = np.array(rows)
    A /= np.linalg.norm(A, axis=1, keepdims=True)
    _, _, Vt = np.linalg.svd(A)
    Xh = Vt[-1]
    return Xh[:3] / Xh[3]


def reprojection_errors(Ps, xs, X):
    """Pixel distance between observations xs and the projection of X, per view."""
    return np.array([np.linalg.norm(project(P, X)[0] - np.asarray(x, float))
                     for P, x in zip(Ps, xs)])


def triangulate_robust(Ps, xs, max_err_px=3.0, min_views=3):
    """DLT with greedy outlier removal.

    Triangulate with the current views; if the worst view's reprojection
    error is above max_err_px, drop that view and triangulate again.
    (One bad detection drags the DLT solution, so every view looks a bit off;
    removing only the worst one each round avoids throwing away good views.)
    Returns (X, kept_view_indices, errors_of_kept_views).
    """
    idx = list(range(len(Ps)))
    while True:
        P_k = [Ps[i] for i in idx]
        x_k = [xs[i] for i in idx]
        X = triangulate_dlt(P_k, x_k)
        err = reprojection_errors(P_k, x_k, X)
        worst = int(np.argmax(err))
        if err[worst] <= max_err_px or len(idx) <= min_views:
            return X, np.array(idx), err
        idx.pop(worst)


# ---------------------------------------------------- similarity alignment
def umeyama(src, dst, with_scale=True):
    """Least-squares similarity transform with dst ~ s * R @ src + t.

    Umeyama (1991): center both point sets, take the SVD of their
    cross-covariance, and fix the sign so R is a rotation (det = +1).
    src, dst: (N,3) corresponding points, N >= 3, not all collinear.
    Returns s (float), R (3x3), t (3,).
    """
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    n = len(src)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    if with_scale:
        var_s = (xs ** 2).sum() / n
        s = float(np.trace(np.diag(D) @ S) / var_s)
    else:
        s = 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def apply_similarity(s, R, t, X):
    return s * np.asarray(X, float) @ np.asarray(R, float).T + np.asarray(t, float)


def similarity_matrix(s, R, t):
    """4x4 matrix of x -> s R x + t."""
    T = np.eye(4)
    T[:3, :3] = s * np.asarray(R, float)
    T[:3, 3] = t
    return T


# ------------------------------------------------------- RANSAC plane fit
def ransac_trials(confidence, inlier_ratio, sample_size):
    """N = log(1 - p) / log(1 - w^s): trials needed to draw one all-inlier
    sample with probability p, when a fraction w of the data are inliers."""
    w_s = inlier_ratio ** sample_size
    if w_s <= 0.0:
        return np.inf
    if w_s >= 1.0:
        return 1
    return int(np.ceil(np.log(1.0 - confidence) / np.log(1.0 - w_s)))


def fit_plane_lstsq(points):
    """Total-least-squares plane: normal = smallest right singular vector of
    the centered points. Returns (n, d) with n . x + d = 0 and |n| = 1."""
    c = points.mean(0)
    _, _, Vt = np.linalg.svd(points - c, full_matrices=False)
    n = Vt[-1]
    return n, -float(n @ c)


def ransac_plane(points, threshold, confidence=0.999, max_iters=5000, rng=None,
                 normal_hint=None, max_angle_deg=30.0):
    """Robustly fit the dominant plane (e.g. the table) to a point cloud.

    Each trial: 3 random points -> plane -> count points within `threshold`.
    The number of trials adapts with ransac_trials() as better models appear.
    The best inlier set is refit with least squares.
    normal_hint: if given, only planes whose normal is within max_angle_deg of
    it (either sign) are considered, so a wall behind the object cannot win
    over the table it stands on.
    Returns (normal (3,), d, inlier_mask), or None if no plane fits the hint.
    """
    rng = np.random.default_rng(rng)
    pts = np.asarray(points, float)
    n_pts = len(pts)
    if normal_hint is not None:
        hint = np.asarray(normal_hint, float) / np.linalg.norm(normal_hint)
        min_cos = np.cos(np.radians(max_angle_deg))
    best_mask, best_count = None, -1
    needed, it = max_iters, 0
    while it < min(needed, max_iters):
        it += 1
        a, b, c = pts[rng.choice(n_pts, 3, replace=False)]
        normal = np.cross(b - a, c - a)
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            continue  # collinear sample
        normal /= norm
        if normal_hint is not None and abs(normal @ hint) < min_cos:
            continue  # too steep to be the table
        d = -normal @ a
        mask = np.abs(pts @ normal + d) < threshold
        count = int(mask.sum())
        if count > best_count:
            best_mask, best_count = mask, count
            needed = ransac_trials(confidence, count / n_pts, 3)
    if best_mask is None:
        return None
    n, d = fit_plane_lstsq(pts[best_mask])
    mask = np.abs(pts @ n + d) < threshold
    return n, d, mask


# ------------------------------------------------------------- rotations
def rotation_between(a, b):
    """Rotation matrix R with R @ a_hat = b_hat (Rodrigues' formula)."""
    a = np.asarray(a, float) / np.linalg.norm(a)
    b = np.asarray(b, float) / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(a @ b)
    if np.linalg.norm(v) < 1e-12:
        if c > 0:
            return np.eye(3)
        # 180 degrees: rotate about any axis perpendicular to a
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


# ------------------------------------------------------------ orbit center
def orbit_center(centers, directions):
    """The point the camera kept looking at while circling the object.

    Each frame's viewing axis is a line (camera center c, direction d). The
    point x closest to all lines in the least-squares sense minimizes
    sum_i ||(I - d_i d_i^T)(x - c_i)||^2, which gives the 3x3 linear system
    [sum_i (I - d_i d_i^T)] x = sum_i (I - d_i d_i^T) c_i.
    """
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for c, d in zip(np.asarray(centers, float), np.asarray(directions, float)):
        d = d / np.linalg.norm(d)
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ c
    return np.linalg.solve(A, b)


def orbit_up(centers, directions):
    """'Up' from the way the camera circled the object.

    The camera centers lie roughly on a horizontal circle, so the normal of
    their best-fit plane (fit_plane_lstsq) is vertical. Its sign is chosen so
    the cameras look down at the object on average (mean direction . up < 0).
    Independent of how the phone was held, unlike the image's 'up' axis.
    """
    n, _ = fit_plane_lstsq(np.asarray(centers, float))
    if np.mean(np.asarray(directions, float), axis=0) @ n > 0:
        n = -n
    return n
