"""Unit tests: each of our implementations vs a library or a known answer.

Run:  python -m pytest -q
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bloxiscan import blur, geometry as geo, meshops, metrics, qem  # noqa: E402
from bloxiscan.colmap_io import read_model_text  # noqa: E402

RNG = np.random.default_rng(0)


def random_rotation(rng):
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q))


def sphere(res=60):
    m = o3d.geometry.TriangleMesh.create_sphere(1.0, res)
    return np.asarray(m.vertices).copy(), np.asarray(m.triangles).copy()


# ------------------------------------------------------------------ blur
def test_laplacian_matches_opencv():
    img = RNG.integers(0, 256, (120, 160)).astype(np.float64)
    ours = blur.laplacian(img)
    ref = cv2.Laplacian(img, cv2.CV_64F, ksize=1)[1:-1, 1:-1]
    assert np.allclose(ours, ref)


def test_blur_lowers_sharpness():
    img = RNG.integers(0, 256, (200, 200, 3)).astype(np.uint8)
    blurred = cv2.GaussianBlur(img, (0, 0), 3)
    assert blur.sharpness(img) > 10 * blur.sharpness(blurred)


def test_select_sharp_keeps_best_per_window():
    scores = [5, 9, 1, 2, 8, 7, 0.1, 0.2]
    assert blur.select_sharp(scores, window=2, min_ratio=0.0) == [1, 3, 4, 7]
    # kept scores 9, 2, 8, 0.2 -> median 5 -> threshold 1.75 drops only 0.2
    assert blur.select_sharp(scores, window=2, min_ratio=0.35) == [1, 3, 4]


# -------------------------------------------------------------- geometry
def make_camera(rng, center, target=np.zeros(3)):
    z = target - center
    z /= np.linalg.norm(z)
    x = np.cross([0, 0, 1.0], z)
    if np.linalg.norm(x) < 1e-6:
        x = np.array([1.0, 0, 0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    t = -R @ center
    K = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1.0]])
    return K, R, t


def test_camera_center():
    K, R, t = make_camera(RNG, np.array([1.0, 2.0, 3.0]))
    assert np.allclose(geo.camera_center(geo.projection_matrix(K, R, t)), [1, 2, 3])


def test_triangulation_exact_and_noisy():
    X = np.array([0.1, -0.2, 0.3])
    Ps = [geo.projection_matrix(*make_camera(RNG, c))
          for c in [[3, 0, 1], [0, 3, 1], [-3, 0, 1.5], [0, -3, 2]]]
    xs = [geo.project(P, X)[0] for P in Ps]
    assert np.allclose(geo.triangulate_dlt(Ps, xs), X, atol=1e-9)
    noisy = [x + RNG.normal(0, 0.5, 2) for x in xs]
    assert np.linalg.norm(geo.triangulate_dlt(Ps, noisy) - X) < 5e-3


def test_triangulation_rejects_outlier_view():
    X = np.array([0.0, 0.1, 0.2])
    cams = [[3, 0, 1], [0, 3, 1], [-3, 0, 1.5], [0, -3, 2], [2, 2, 2]]
    Ps = [geo.projection_matrix(*make_camera(RNG, c)) for c in cams]
    xs = [geo.project(P, X)[0] for P in Ps]
    xs[2] = xs[2] + 60.0  # a bad detection
    X_hat, kept, _ = geo.triangulate_robust(Ps, xs)
    assert 2 not in kept and np.linalg.norm(X_hat - X) < 1e-6


def test_umeyama_recovers_similarity_and_matches_open3d():
    src = RNG.normal(size=(50, 3))
    s, R, t = 2.5, random_rotation(RNG), np.array([1.0, -2.0, 0.5])
    dst = geo.apply_similarity(s, R, t, src)
    s2, R2, t2 = geo.umeyama(src, dst)
    assert np.isclose(s2, s) and np.allclose(R2, R) and np.allclose(t2, t)
    est = o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True)
    corr = o3d.utility.Vector2iVector(np.c_[np.arange(50), np.arange(50)])
    T = est.compute_transformation(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(src)),
                                   o3d.geometry.PointCloud(o3d.utility.Vector3dVector(dst)), corr)
    assert np.allclose(T, geo.similarity_matrix(s2, R2, t2), atol=1e-6)


def test_ransac_trials_formula():
    assert geo.ransac_trials(0.99, 0.5, 8) == 1177


def test_ransac_plane_with_outliers():
    n_true = np.array([0.2, -0.1, 1.0])
    n_true /= np.linalg.norm(n_true)
    basis = np.linalg.svd(n_true[None])[2][1:]
    plane = RNG.uniform(-1, 1, (600, 2)) @ basis + 0.3 * n_true + RNG.normal(0, 0.002, (600, 1)) * n_true
    outliers = RNG.uniform(-1, 1, (400, 3))
    pts = np.vstack([plane, outliers])
    n, d, mask = geo.ransac_plane(pts, 0.01, rng=1)
    assert abs(abs(n @ n_true) - 1) < 1e-3 and abs(abs(d) - 0.3) < 0.01
    assert mask[:600].mean() > 0.97 and mask[600:].mean() < 0.05


def test_rotation_between():
    a, b = RNG.normal(size=3), RNG.normal(size=3)
    R = geo.rotation_between(a, b)
    assert np.allclose(R @ (a / np.linalg.norm(a)), b / np.linalg.norm(b))
    assert np.isclose(np.linalg.det(R), 1.0)
    R2 = geo.rotation_between([0, 0, 1.0], [0, 0, -1.0])
    assert np.allclose(R2 @ [0, 0, 1.0], [0, 0, -1.0]) and np.isclose(np.linalg.det(R2), 1.0)


# ------------------------------------------------------------------- QEM
def test_qem_keeps_sphere_closed_and_accurate():
    V, F = sphere(60)
    V2, F2, st = qem.simplify(V, F, 1000)
    assert len(F2) <= 1000 and len(F2) >= 990
    assert meshops.is_watertight(F2) and meshops.euler_characteristic(F2) == 2
    assert np.abs(np.linalg.norm(V2, axis=1) - 1).max() < 0.01


def test_qem_close_to_open3d():
    V, F = sphere(60)
    gt = metrics.sample_surface(V, F, 50_000, 1)
    V2, F2, _ = qem.simplify(V, F, 800)
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F))
    m = m.simplify_quadric_decimation(800)
    ours = metrics.mesh_vs_points(V2, F2, gt, 0.01, 50_000)["chamfer"]
    lib = metrics.mesh_vs_points(np.asarray(m.vertices), np.asarray(m.triangles), gt, 0.01, 50_000)["chamfer"]
    assert ours < 1.5 * lib, (ours, lib)


def test_qem_preserves_open_border():
    g = o3d.geometry.TriangleMesh.create_box(1, 1, 1e-9).subdivide_midpoint(4)
    V, F = np.asarray(g.vertices), np.asarray(g.triangles)
    top = V[:, 2] > 0.5e-9  # keep only the top face of the thin box: a flat square
    keep = top[F].all(axis=1)
    V, F = meshops.remove_unreferenced(V, F[keep])
    V2, F2, _ = qem.simplify(V, F, 20)
    assert np.allclose(V2[:, :2].min(0), [0, 0]) and np.allclose(V2[:, :2].max(0), [1, 1])
    assert np.isclose(metrics.triangle_areas(V2, F2).sum(), 1.0)


# --------------------------------------------------------------- meshops
def test_fill_holes_restores_watertight():
    V, F = sphere(30)
    F_holes = np.delete(F, [10, 11, 500, 900], axis=0)
    assert not meshops.is_watertight(F_holes)
    V2, F2, n = meshops.fill_holes(V, F_holes)
    assert n >= 2 and meshops.is_watertight(F2)


def test_largest_component():
    V, F = sphere(20)
    V_small, F_small = sphere(5)
    Vall = np.vstack([V, V_small * 0.1 + 5])
    Fall = np.vstack([F, F_small + len(V)])
    V2, F2 = meshops.keep_largest_component(Vall, Fall)
    assert len(F2) == len(F) and len(V2) == len(V)


# --------------------------------------------------------------- metrics
def test_chamfer_and_fscore_on_spheres():
    V, F = sphere(80)
    a = metrics.sample_surface(V, F, 40_000, 0)
    b = metrics.sample_surface(V * 1.1, F, 40_000, 1)
    c = metrics.chamfer(a, b)
    assert abs(c["chamfer"] - 0.1) < 0.01
    assert metrics.fscore(a, b, 0.05)["fscore"] == 0.0
    assert metrics.fscore(a, b, 0.2)["fscore"] == 1.0


def test_sampling_is_uniform():
    V, F = sphere(80)
    p = metrics.sample_surface(V, F, 100_000, 0)
    assert np.abs(p.mean(0)).max() < 0.01                 # no side favoured
    assert np.abs(np.linalg.norm(p, axis=1) - 1).max() < 0.01


# ------------------------------------------------------------- COLMAP IO
def test_colmap_text_reader(tmp_path):
    K, R, t = make_camera(RNG, np.array([2.0, 1.0, 1.0]))
    from scipy.spatial.transform import Rotation
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    (tmp_path / "cameras.txt").write_text(
        "# Camera list\n1 PINHOLE 640 480 800 800 320 240\n")
    (tmp_path / "images.txt").write_text(
        "# Image list\n"
        f"1 {w} {x} {y} {z} {t[0]} {t[1]} {t[2]} 1 frame_000001.jpg\n"
        "\n"
        f"2 {w} {x} {y} {z} {t[0]} {t[1]} {t[2]} 1 frame_000002.jpg\n"
        "100.0 200.0 -1\n")
    cams, imgs = read_model_text(tmp_path)
    assert len(imgs) == 2 and imgs[2].name == "frame_000002.jpg"
    assert np.allclose(imgs[1].R, R) and np.allclose(imgs[1].center, [2, 1, 1])
    X = np.array([[0.1, 0.2, 0.3]])
    assert np.allclose(geo.project(imgs[1].P(cams[1]), X),
                       geo.project(geo.projection_matrix(K, R, t), X))
