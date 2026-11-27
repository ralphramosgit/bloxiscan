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
from bloxiscan import blur, carve, geometry as geo, meshops, metrics, qem, texture  # noqa: E402
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


def test_ransac_plane_hint_ignores_wall():
    """A wall with 3x the points of the floor wins plain RANSAC; with the
    camera orbit's 'up' as a hint the floor is found."""
    floor = np.c_[RNG.uniform(-1, 1, (500, 2)), np.zeros(500)]
    wall = np.c_[np.full(1500, -1.0), RNG.uniform(-1, 1, 1500), RNG.uniform(0, 2, 1500)]
    pts = np.vstack([floor, wall]) + RNG.normal(0, 0.002, (2000, 3))
    n, _, _ = geo.ransac_plane(pts, 0.01, rng=0)
    assert abs(n[0]) > 0.99                                   # the wall
    n, d, mask = geo.ransac_plane(pts, 0.01, rng=0, normal_hint=[0.1, 0, 1])
    assert abs(n[2]) > 0.999 and abs(d) < 0.01 and mask[:500].mean() > 0.97
    assert geo.ransac_plane(floor, 0.01, rng=0, normal_hint=[1, 0, 0]) is None


def test_orbit_up():
    """Cameras circling at different heights, looking down: up is +z,
    whatever the orbit's direction of travel."""
    target = np.array([0.0, 0.0, 0.2])
    for sign in (1, -1):
        ang = sign * np.linspace(0, 2 * np.pi, 20, endpoint=False)
        C = np.c_[np.cos(ang), np.sin(ang), 0.5 + 0.05 * np.sin(3 * ang)]
        up = geo.orbit_up(C, target - C)
        assert up[2] > 0.99


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


def test_fill_holes_leaves_winding_rim_open():
    """A flat hole is filled; a rim that winds around in 3D (here: a band cut
    out of the sphere, like Poisson around a gap in the data) is left open."""
    V, F = sphere(30)
    c = V[F].mean(1)
    flat = c[:, 2] < -0.8                                # bottom cap: flat rim
    band = (np.abs(c[:, 0]) < 0.15) & (c[:, 2] > 0)      # half a slot: winding rim
    V2, F2, n = meshops.fill_holes(V, F[~(flat | band)], max_flatness=0.1)
    assert n == 1 and len(meshops.boundary_loops(F2)) == 1


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


# ------------------------------------------------------------- carving
def cylinder_views(r, h, n_views=24, elevations=(30, 60), size=(480, 640)):
    """Cameras circling a cylinder (axis z, standing on z = 0) and its exact
    silhouettes (the convex hull of the projected cylinder)."""
    ang = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    rim = np.c_[r * np.cos(ang), r * np.sin(ang)]
    solid = np.vstack([np.c_[rim, np.zeros(64)], np.c_[rim, np.full(64, h)]])
    views, rng = [], np.random.default_rng(0)
    for i, a in enumerate(np.linspace(0, 2 * np.pi, n_views, endpoint=False)):
        e = np.radians(elevations[i % len(elevations)])
        target = np.array([0, 0, h / 2])
        C = target + 1.5 * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
        K, R, t = make_camera(rng, C, target)
        P = geo.projection_matrix(K, R, t)
        mask = np.zeros(size, np.uint8)
        hull = cv2.convexHull(geo.project(P, solid).astype(np.float32)).astype(np.int32)
        cv2.fillConvexPoly(mask, hull, 1)
        views.append((P, mask.astype(bool)))
    return views


def test_visual_hull_of_cylinder():
    r, h = 0.15, 0.4
    views = cylinder_views(r, h)
    pts, nrm, info = carve.visual_hull(views, [0, 0, 1.0], 0.0, np.array([0.03, -0.02, 0.2]),
                                       0.45, 0.9, res=80)
    assert not info["touches_box"]
    side = (pts[:, 2] > 0.3 * h) & (pts[:, 2] < 0.7 * h)
    rho = np.linalg.norm(pts[side, :2], axis=1)
    assert abs(np.median(rho) - r) < 0.01, np.median(rho)
    assert np.all(np.abs(rho - r) < 1.5 * info["voxel"])        # no grooves or bumps
    assert rho.std() < 0.2 * info["voxel"]                      # no voxel staircase
    radial = np.c_[pts[side, :2] / rho[:, None], np.zeros(side.sum())]
    assert np.mean(np.einsum("ij,ij->i", nrm[side], radial)) > 0.95   # outward
    # the bottom sits on the table; the top may gain a shallow cone
    # (about r * tan(lowest elevation)) because no camera looks straight down
    assert pts[:, 2].min() < 0.01
    assert h - 0.01 < pts[:, 2].max() < h + r * np.tan(np.radians(30)) + 0.01


def test_carving_tolerates_a_bad_mask():
    """One empty mask (a missed detection) must not carve the object away."""
    views = cylinder_views(0.15, 0.4)
    views[3] = (views[3][0], np.zeros_like(views[3][1]))
    pts, _, _ = carve.visual_hull(views, [0, 0, 1.0], 0.0, np.array([0, 0, 0.2]),
                                  0.45, 0.9, res=60)
    rho = np.linalg.norm(pts[(pts[:, 2] > 0.1) & (pts[:, 2] < 0.3), :2], axis=1)
    assert abs(np.median(rho) - 0.15) < 0.015


# ------------------------------------------------------------- texture
RED, BLUE, GRAY = (220, 40, 40), (40, 40, 220), (128, 128, 128)


def two_colour_frames(views, r, h, size=(480, 640)):
    """Ray-cast images of a cylinder whose top half is red and bottom half is
    blue (background gray), seen by the cameras of cylinder_views."""
    import trimesh
    import open3d as o3d
    cyl = trimesh.creation.cylinder(r, h, sections=96)
    cyl.apply_translation([0, 0, h / 2])
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(cyl.vertices.astype(np.float32)),
                        o3d.core.Tensor(cyl.faces.astype(np.uint32)))
    K = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1.0]])
    ys, xs = np.mgrid[0:size[0], 0:size[1]]
    frames = []
    for P, _ in views:
        Rt = np.linalg.inv(K) @ P
        R, t = Rt[:, :3], Rt[:, 3]
        C = -R.T @ t
        d = np.c_[(xs.ravel() + 0.5 - 320) / 800, (ys.ravel() + 0.5 - 240) / 800,
                  np.ones(xs.size)] @ R
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        hit = scene.cast_rays(o3d.core.Tensor(
            np.c_[np.repeat(C[None], len(d), 0), d].astype(np.float32)))["t_hit"].numpy()
        z = C[2] + hit * d[:, 2]
        img = np.full((xs.size, 3), GRAY, np.uint8)
        on = np.isfinite(hit)
        img[on & (z >= h / 2)] = RED
        img[on & (z < h / 2)] = BLUE
        frames.append({"K": K, "R": R, "t": t, "image": img.reshape(*size, 3),
                       "mask": on.reshape(size)})
    return frames


def face_colours(uv, F_uv, tex):
    c = uv[F_uv].mean(1)
    n = len(tex)
    return tex[np.clip(((1 - c[:, 1]) * (n - 1)).round().astype(int), 0, n - 1),
               np.clip((c[:, 0] * (n - 1)).round().astype(int), 0, n - 1)].astype(int)


def test_bake_texture_puts_colours_in_the_right_place():
    """Bake a texture onto the cylinder from its ray-cast frames: faces on the
    top half must come out red, faces on the bottom half blue, and the bottom
    cap (never seen) must still get a colour from its neighbours."""
    import trimesh
    r, h = 0.15, 0.4
    frames = two_colour_frames(cylinder_views(r, h), r, h)
    cyl = trimesh.creation.cylinder(r, h, sections=64)
    cyl.apply_translation([0, 0, h / 2])
    cyl = cyl.subdivide().subdivide()
    vmap, F_uv, uv, tex, info = texture.bake_texture(cyl.vertices, cyl.faces, frames, size=512)
    assert len(F_uv) == len(cyl.faces)                    # one UV face per face
    z = cyl.vertices[vmap][F_uv].mean(1)[:, 2]
    col = face_colours(uv, F_uv, tex)
    top, bottom = z > h / 2 + 0.03, (z < h / 2 - 0.03) & (z > 0.01)
    assert (col[top, 0] > col[top, 2] + 100).mean() > 0.97
    assert (col[bottom, 2] > col[bottom, 0] + 100).mean() > 0.97
    assert np.all(np.abs(col - GRAY).sum(1) > 60)          # no background leaked in
    assert 0.7 < info["texels_seen"] < 1.0                  # the bottom cap was not seen
