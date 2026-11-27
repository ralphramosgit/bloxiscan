"""End-to-end tests: run the scripts on synthetic data with known answers."""
import csv
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import trimesh
from scipy.io import savemat
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bloxiscan import dtu, meshops  # noqa: E402
from bloxiscan.metrics import sample_surface  # noqa: E402
from tests.test_core import make_camera, sphere  # noqa: E402

PY = sys.executable


def run(*args):
    r = subprocess.run([PY, *map(str, args)], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


def test_aruco_scale_recovers_known_scale(tmp_path):
    """Render a marker of side 0.37 (model units) into 10 views, write a COLMAP
    model, and check that L = 0.10 m gives scale 0.10 / 0.37."""
    ell, S, b = 0.37, 400, 100
    marker = cv2.aruco.generateImageMarker(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), 0, S)
    canvas = np.full((S + 2 * b, S + 2 * b), 255, np.uint8)
    canvas[b:b + S, b:b + S] = marker
    # marker pixel (u right, v down) -> world (x, y) on z = 0, seen from above.
    # v maps to -y, otherwise the camera sees a mirror image (not a valid marker).
    A = np.array([[ell / S, 0, -b * ell / S], [0, -ell / S, b * ell / S], [0, 0, 1]])
    img_dir, model = tmp_path / "images", tmp_path / "model"
    img_dir.mkdir()
    model.mkdir()
    (model / "cameras.txt").write_text("1 PINHOLE 640 480 800 800 320 240\n")
    lines = []
    target = np.array([ell / 2, -ell / 2, 0.0])
    rng = np.random.default_rng(3)
    for i, ang in enumerate(np.linspace(0, 2 * np.pi, 10, endpoint=False)):
        C = target + np.array([0.7 * np.cos(ang), 0.7 * np.sin(ang), 0.9])
        K, R, t = make_camera(rng, C, target)
        H = K @ np.c_[R[:, 0], R[:, 1], t] @ A
        img = cv2.warpPerspective(canvas, H, (640, 480), borderValue=255)
        name = f"frame_{i:06d}.jpg"
        cv2.imwrite(str(img_dir / name), img)
        x, y, z, w = Rotation.from_matrix(R).as_quat()
        lines += [f"{i + 1} {w} {x} {y} {z} {t[0]} {t[1]} {t[2]} 1 {name}", ""]
    (model / "images.txt").write_text("\n".join(lines) + "\n")
    out = tmp_path / "scale.json"
    run("scripts/aruco_scale.py", "--model", model, "--images", img_dir,
        "--length", 0.10, "--out", out)
    s = json.loads(out.read_text())
    assert abs(s["scale"] / (0.10 / ell) - 1) < 0.01, s["scale"]
    assert abs(s["diagonal_over_side"] - np.sqrt(2)) < 0.01


def test_make_mesh_on_fake_colmap_cloud(tmp_path):
    """A 0.6 m ball resting above a table, moved into a random 'COLMAP' frame
    (scale 3.7). The output must be upright, watertight, <= budget, ~0.6 m."""
    rng = np.random.default_rng(0)
    d = rng.normal(size=(40000, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    ball = d * 0.3 + [0, 0, 0.4]
    table = np.c_[rng.uniform(-1, 1, (40000, 2)), np.zeros(40000)]
    pts = np.vstack([ball, table]) + rng.normal(0, 0.002, (80000, 3))
    nrm = np.vstack([d, np.tile([0, 0, 1.0], (40000, 1))])
    R = Rotation.random(random_state=1).as_matrix()
    pts = 3.7 * pts @ R.T + [5, -2, 1]
    nrm = nrm @ R.T
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pcd.normals = o3d.utility.Vector3dVector(nrm)
    o3d.io.write_point_cloud(str(tmp_path / "fused.ply"), pcd)
    out = tmp_path / "out"
    run("scripts/make_mesh.py", tmp_path / "fused.ply", "--out", out, "--depth", "8",
        "--target", "3000", "--scale", 1 / 3.7)
    st = json.loads((out / "stats.json").read_text())
    assert st["triangles"] <= 3000 and st["watertight"]
    assert np.allclose(st["size_xyz"], [0.6, 0.6, 0.6], atol=0.06), st["size_xyz"]
    m = trimesh.load(out / "model.glb", force="mesh")
    assert m.bounds[0][1] > -1e-6          # sits on y = 0 (upright, glTF +Y up)
    assert meshops.is_watertight(np.asarray(m.faces))


def test_extract_frames_skips_blurry(tmp_path):
    rng = np.random.default_rng(0)
    base = cv2.resize(rng.integers(0, 256, (120, 160, 3)).astype(np.uint8), (640, 480),
                      interpolation=cv2.INTER_NEAREST)
    video = tmp_path / "clip.avi"
    vw = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 30, (640, 480))
    for i in range(60):
        f = np.roll(base, 4 * i, axis=1)
        if i % 3 == 0:
            f = cv2.GaussianBlur(f, (0, 0), 6)
        vw.write(f)
    vw.release()
    out = tmp_path / "frames" / "images"
    run("scripts/extract_frames.py", video, out, "--fps", 15, "--window", 2)
    kept = [int(p.stem.split("_")[1]) for p in out.glob("*.jpg")]
    assert len(kept) >= 25 and all(i % 3 != 0 for i in kept)
    rows = list(csv.DictReader(open(tmp_path / "frames" / "scores.csv")))
    assert len(rows) == 60


def test_budget_sweep_runs(tmp_path):
    V, F = sphere(60)
    trimesh.Trimesh(V, F, process=False).export(tmp_path / "full.ply")
    run("scripts/budget_sweep.py", tmp_path / "full.ply", "--tau", 0.01, "--units", "m",
        "--budgets", 2000, 500, "--out", tmp_path / "res")
    rows = list(csv.DictReader(open(tmp_path / "res" / "sweep.csv")))
    assert len(rows) == 5 and (tmp_path / "res" / "sweep.png").exists()
    ours = [r for r in rows if r["simplifier"] == "ours"]
    assert all(r["watertight"] == "True" for r in ours)


def test_dtu_protocol_on_fake_scan(tmp_path):
    (tmp_path / "ObsMask").mkdir()
    (tmp_path / "Points" / "stl").mkdir(parents=True)
    savemat(tmp_path / "ObsMask" / "ObsMask1_10.mat",
            {"ObsMask": np.ones((41, 41, 41), bool), "BB": np.array([[-1.2] * 3, [1.2] * 3]),
             "Res": np.array([[0.06]])})
    savemat(tmp_path / "ObsMask" / "Plane1.mat", {"P": np.array([[0], [0], [1], [10.0]])})
    V, F = sphere(80)
    gt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(sample_surface(V, F, 200_000, 5)))
    o3d.io.write_point_cloud(str(tmp_path / "Points" / "stl" / "stl001_total.ply"), gt)
    same = dtu.evaluate_dtu(V, F, 1, tmp_path, density=0.02, max_dist=0.5, patch=0.1)
    bigger = dtu.evaluate_dtu(V * 1.05, F, 1, tmp_path, density=0.02, max_dist=0.5, patch=0.1)
    assert same["chamfer_mm"] < 0.01
    assert abs(bigger["chamfer_mm"] - 0.05) < 0.01


def test_make_mesh_without_normals(tmp_path):
    """Sparse COLMAP points / scanner clouds have no normals: estimate them."""
    rng = np.random.default_rng(2)
    d = rng.normal(size=(20000, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(d * 0.3 + rng.normal(0, 0.002, (20000, 3))))
    o3d.io.write_point_cloud(str(tmp_path / "sparse.ply"), pcd)
    out = tmp_path / "out"
    run("scripts/make_mesh.py", tmp_path / "sparse.ply", "--out", out, "--no-plane",
        "--keep-frame", "--depth", "7", "--target", "2000", "--voxel", "0.01")
    st = json.loads((out / "stats.json").read_text())
    assert st["watertight"] and st["triangles"] <= 2000
    assert np.allclose(st["size_xyz"], [0.6, 0.6, 0.6], atol=0.05), st["size_xyz"]


def test_dtu_mask_points_and_scale_mat(tmp_path):
    """Object = ball at the origin of the normalized frame; background = a wide
    floor. Masks are the ball's silhouette. Masking must keep the ball, drop
    the floor; --scale-mat must map normalized -> world coordinates."""
    rng = np.random.default_rng(4)
    scan = tmp_path / "scan1"
    (scan / "mask").mkdir(parents=True)
    s, t = 200.0, np.array([10.0, -20.0, 600.0])
    S = np.eye(4)
    S[:3, :3] *= s
    S[:3, 3] = t
    d = rng.normal(size=(5000, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    ball = d * 0.3
    floor = np.c_[rng.uniform(-1, 1, (5000, 2)), np.full(5000, -0.8)]  # in view
    cams = {}
    for i, ang in enumerate(np.linspace(0, 2 * np.pi, 8, endpoint=False)):
        C = t + s * np.array([2.5 * np.cos(ang), 2.5 * np.sin(ang), 1.0])
        K, R, tt = make_camera(rng, C, t)
        W = np.eye(4)
        W[:3, :4] = K @ np.c_[R, tt]
        cams[f"world_mat_{i}"], cams[f"scale_mat_{i}"] = W, S
        P = (W @ S)[:3]
        x = np.c_[ball, np.ones(len(ball))] @ P.T
        uv = np.round(x[:, :2] / x[:, 2:]).astype(int)
        m = np.zeros((480, 640), np.uint8)
        for u, v in uv:
            cv2.circle(m, (int(u), int(v)), 6, 255, -1)
        cv2.imwrite(str(scan / "mask" / f"{i:03d}.png"), m)
    np.savez(scan / "cameras.npz", **cams)
    pts = np.vstack([ball, floor])
    o3d.io.write_point_cloud(str(tmp_path / "fused.ply"),
                             o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts)))
    run("scripts/dtu_mask_points.py", "--scan-dir", scan, "--input", tmp_path / "fused.ply",
        "--out", tmp_path / "masked.ply", "--dilate", 4)
    kept = np.asarray(o3d.io.read_point_cloud(str(tmp_path / "masked.ply")).points)
    r = np.linalg.norm(kept, axis=1)
    assert (r < 0.35).sum() >= 0.99 * len(ball)       # ball kept
    assert (r >= 0.35).sum() <= 0.02 * len(floor)     # floor dropped
    V, F = sphere(20)
    trimesh.Trimesh(V * 0.3, F, process=False).export(tmp_path / "m.ply")
    run("scripts/align_to_reference.py", "--scale-mat", scan / "cameras.npz",
        "--mesh", tmp_path / "m.ply", "--out", tmp_path / "m_world.ply")
    Vw = trimesh.load(tmp_path / "m_world.ply", process=False).vertices
    assert np.allclose(Vw, V * 0.3 * s + t)


def test_make_mesh_real_height(tmp_path):
    """No marker: --real-height 0.6 must make the upright ball 0.6 m tall."""
    rng = np.random.default_rng(7)
    d = rng.normal(size=(30000, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    ball = d * 0.3 + [0, 0, 0.4]
    table = np.c_[rng.uniform(-1, 1, (30000, 2)), np.zeros(30000)]
    pts = np.vstack([ball, table])
    nrm = np.vstack([d, np.tile([0, 0, 1.0], (30000, 1))])
    R = Rotation.random(random_state=3).as_matrix()
    pts = 5.0 * pts @ R.T          # unknown COLMAP scale
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pcd.normals = o3d.utility.Vector3dVector(nrm @ R.T)
    o3d.io.write_point_cloud(str(tmp_path / "fused.ply"), pcd)
    out = tmp_path / "out"
    run("scripts/make_mesh.py", tmp_path / "fused.ply", "--out", out, "--depth", "8",
        "--target", "3000", "--real-height", "0.6")
    st = json.loads((out / "stats.json").read_text())
    assert abs(st["size_xyz"][1] - 0.6) < 0.01, st["size_xyz"]


def test_orbit_center_and_background_crop(tmp_path):
    """Cameras circle a ball on a table in a cluttered room. The orbit-center
    crop must cut the clutter away so the mesh is just the ball."""
    from bloxiscan.geometry import orbit_center
    rng = np.random.default_rng(11)
    target = np.array([0.0, 0.0, 0.4])
    model = tmp_path / "model"
    model.mkdir()
    (model / "cameras.txt").write_text("1 PINHOLE 640 480 500 500 320 240\n")
    lines, C, D = [], [], []
    for i, ang in enumerate(np.linspace(0, 2 * np.pi, 16, endpoint=False)):
        c = target + np.array([1.2 * np.cos(ang), 1.2 * np.sin(ang), 0.5 + 0.1 * (i % 2)])
        K, R, t = make_camera(rng, c, target)
        x, y, z, w = Rotation.from_matrix(R).as_quat()
        lines += [f"{i + 1} {w} {x} {y} {z} {t[0]} {t[1]} {t[2]} 1 f{i:03d}.jpg", ""]
        C.append(c)
        D.append(R.T @ np.array([0, 0, 1.0]))
    (model / "images.txt").write_text("\n".join(lines) + "\n")
    assert np.linalg.norm(orbit_center(C, D) - target) < 1e-6

    d = rng.normal(size=(30000, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    ball = d * 0.15 + target
    table = np.c_[rng.uniform(-0.6, 0.6, (20000, 2)), np.full(20000, 0.24)]
    clutter = rng.uniform(-3, 3, (15000, 3))
    clutter = clutter[np.linalg.norm(clutter[:, :2], axis=1) > 1.5]   # around the room
    pts = np.vstack([ball, table, clutter])
    nrm = np.vstack([d, np.tile([0, 0, 1.0], (20000, 1)),
                     rng.normal(size=(len(clutter), 3))])
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pcd.normals = o3d.utility.Vector3dVector(nrm / np.linalg.norm(nrm, axis=1, keepdims=True))
    o3d.io.write_point_cloud(str(tmp_path / "fused.ply"), pcd)
    out = tmp_path / "out"
    run("scripts/make_mesh.py", tmp_path / "fused.ply", "--out", out, "--depth", "8",
        "--target", "3000", "--no-cluster", "--cameras", model, "--real-height", "0.3")
    st = json.loads((out / "stats.json").read_text())
    assert st["watertight"]
    assert np.allclose(st["size_xyz"], [0.3, 0.3, 0.3], atol=0.03), st["size_xyz"]
