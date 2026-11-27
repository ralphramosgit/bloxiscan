"""Stages 5-6: COLMAP point cloud -> Roblox-ready .glb (<= 20,000 triangles).

Your object (phone video), scaled by the ArUco marker:
    python scripts/make_mesh.py work/obj/dense/0/fused.ply --out out/obj \
        --scale-json out/obj/scale.json
No marker: give the object's real height in metres instead:
    python scripts/make_mesh.py work/obj/dense/0/fused.ply --out out/obj --real-height 0.21

DTU scan (keep DTU's coordinates for evaluation, no table removal):
    python scripts/make_mesh.py scan24/dense/fused.ply --out out/scan24 --no-plane --keep-frame

Steps (ours = written in this repo, lib = library call):
  1. ours  RANSAC plane fit finds the table; table points are removed and the
           table normal becomes "up" (+Y, which glTF and Roblox use)
  2. ours  with --cameras: keep points near where the camera circled, then cut the
           object out with a cylinder around its axis (bloxiscan/isolate.py);
     lib   otherwise DBSCAN (Open3D) -> crop to the biggest cluster
  3. lib   Screened Poisson reconstruction (Open3D), trim low-density vertices
  4. ours  keep the largest connected piece, fill holes, check watertight
  5. ours  QEM simplification to --target triangles (or --simplifier open3d)
  6. ours  metric scale (ArUco result, real height, or a number), export .glb in metres
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bloxiscan.colmap_io import read_model_text  # noqa: E402
from bloxiscan.geometry import orbit_center, ransac_plane, rotation_between  # noqa: E402
from bloxiscan.isolate import isolate_object  # noqa: E402
from bloxiscan.meshops import (euler_characteristic, fill_holes,  # noqa: E402
                               is_watertight, keep_largest_component)
from bloxiscan.qem import simplify  # noqa: E402


def robust_size(pts):
    lo, hi = np.percentile(pts, 1, axis=0), np.percentile(pts, 99, axis=0)
    return float(np.linalg.norm(hi - lo))


def biggest_cluster_box(pts, size):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    small = pcd.voxel_down_sample(0.005 * size)
    labels = np.asarray(small.cluster_dbscan(eps=0.02 * size, min_points=10))
    if labels.max() < 0:
        return None
    best = np.bincount(labels[labels >= 0]).argmax()
    p = np.asarray(small.points)[labels == best]
    pad = 0.03 * size
    return p.min(0) - pad, p.max(0) + pad


def simplify_open3d(V, F, target):
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F))
    m = m.simplify_quadric_decimation(target_number_of_triangles=target)
    return np.asarray(m.vertices), np.asarray(m.triangles)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="point cloud with normals, e.g. dense/0/fused.ply")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--crop", type=float, nargs=6, metavar=("XMIN", "YMIN", "ZMIN", "XMAX", "YMAX", "ZMAX"))
    ap.add_argument("--no-plane", action="store_true", help="skip table removal (DTU)")
    ap.add_argument("--plane-thresh", type=float, default=0.01,
                    help="RANSAC inlier distance, as a fraction of the cloud size")
    ap.add_argument("--no-cluster", action="store_true", help="skip the DBSCAN crop")
    ap.add_argument("--keep-frame", action="store_true", help="do not rotate/move the model (DTU)")
    ap.add_argument("--depth", type=int, default=9, help="Poisson octree depth")
    ap.add_argument("--trim", type=float, default=0.02, help="density quantile to trim")
    ap.add_argument("--target", type=int, default=20000, help="triangle budget")
    ap.add_argument("--simplifier", choices=["ours", "open3d"], default="ours")
    ap.add_argument("--scale", type=float, default=1.0, help="multiply coordinates (to metres)")
    ap.add_argument("--scale-json", help="scale.json from scripts/aruco_scale.py")
    ap.add_argument("--cameras", help="COLMAP text model: keep only points near the spot the "
                    "camera circled (cuts away the background)")
    ap.add_argument("--focus-radius", type=float, default=1.0,
                    help="crop radius, in half-heights of the view at the object's distance")
    ap.add_argument("--radius-scale", type=float, default=1.35,
                    help="object cut-out radius vs. the measured outline (raise for long objects)")
    ap.add_argument("--no-isolate", action="store_true", help="skip the object cut-out")
    ap.add_argument("--real-height", type=float,
                    help="no marker: the object's real height in metres (e.g. 0.21)")
    ap.add_argument("--voxel", type=float, default=0.0,
                    help="downsample the input to one point per voxel of this size first")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    log = {"input": a.input}
    t_start = time.time()

    pcd = o3d.io.read_point_cloud(a.input)
    print(f"Loaded {len(pcd.points):,} points")
    if a.voxel > 0:
        pcd = pcd.voxel_down_sample(a.voxel)
        print(f"Voxel downsampled to {len(pcd.points):,} points")
    need_normals = not pcd.has_normals()
    pts = np.asarray(pcd.points)
    nrm = np.asarray(pcd.normals) if not need_normals else np.zeros_like(pts)

    center = view_half = None
    if a.cameras:
        # every frame looks at the object, so the viewing axes meet there
        cams, imgs = read_model_text(a.cameras)
        C = np.array([im.center for im in imgs.values()])
        D = np.array([im.R.T @ np.array([0.0, 0.0, 1.0]) for im in imgs.values()])
        center = orbit_center(C, D)
        dist = float(np.median(np.linalg.norm(C - center, axis=1)))
        cam = next(iter(cams.values()))
        view_half = dist * (cam.height / 2) / cam.K[1, 1]
        radius = a.focus_radius * view_half
        keep = np.linalg.norm(pts - center, axis=1) < radius
        pts, nrm = pts[keep], nrm[keep]
        print(f"Kept {keep.sum():,} points near the orbit center (radius {radius:.3f})")

    if a.crop:
        lo, hi = np.array(a.crop[:3]), np.array(a.crop[3:])
        keep = np.all((pts >= lo) & (pts <= hi), axis=1)
        pts, nrm = pts[keep], nrm[keep]

    size = robust_size(pts)
    up_R, plane = None, None
    if not a.no_plane:
        n, d, inl = ransac_plane(pts, a.plane_thresh * size, rng=0)
        frac = inl.mean()
        side = np.sign(np.median(pts[~inl] @ n + d))
        n, d = n * side, d * side                  # object on the positive side
        above = (pts @ n + d) > a.plane_thresh * size
        print(f"Table plane: {frac:.0%} of points; keeping {above.sum():,} above it")
        if frac < 0.05:
            print("  warning: no clear table plane found; check the result")
        pts, nrm = pts[above], nrm[above]
        plane = (n, d)
        log["plane"] = {"normal": n.tolist(), "d": d, "inlier_fraction": float(frac)}

    iso = None
    if center is not None and plane is not None and not a.no_isolate:
        mask, iso = isolate_object(pts, plane[0], plane[1], center, view_half,
                                   a.plane_thresh * size, a.radius_scale)
        pts, nrm = pts[mask], nrm[mask]
        if iso:
            print(f"Object cut-out: {len(pts):,} points inside a cylinder of radius "
                  f"{iso['radius']:.3f} around the object's axis")

    if not a.no_cluster and iso is None:
        box = biggest_cluster_box(pts, size)
        if box is not None:
            keep = np.all((pts >= box[0]) & (pts <= box[1]), axis=1)
            pts, nrm = pts[keep], nrm[keep]
            print(f"Biggest cluster: {len(pts):,} points")

    if need_normals:
        # sparse COLMAP points and scanner point clouds have no normals:
        # estimate them from 30 neighbours, make neighbours agree, then point
        # them away from the object's center (outward) on average
        print("No normals in the input: estimating them...")
        tmp = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        tmp.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(30))
        tmp.orient_normals_consistent_tangent_plane(15)
        nrm = np.asarray(tmp.normals).copy()
        if iso:   # outward = away from the object's vertical axis
            out = pts - iso["axis"]
            out -= np.outer(out @ plane[0], plane[0])
        else:     # outward = away from the middle of the points
            out = pts - np.median(pts, axis=0)
        if np.mean(np.einsum("ij,ij->i", out, nrm)) < 0:
            nrm = -nrm

    if not a.keep_frame and plane is not None:
        up_R = rotation_between(plane[0], [0.0, 1.0, 0.0])
        pts, nrm = pts @ up_R.T, nrm @ up_R.T
        pts[:, 1] += plane[1]                     # table at y = 0
        c = 0.5 * (pts.min(0) + pts.max(0))
        pts[:, 0] -= c[0]
        pts[:, 2] -= c[2]

    # --- Poisson (library)
    t0 = time.time()
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    cloud.normals = o3d.utility.Vector3dVector(nrm)
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(cloud, depth=a.depth)
    dens = np.asarray(dens)
    mesh.remove_vertices_by_mask(dens < np.quantile(dens, a.trim))
    lo, hi = pts.min(0), pts.max(0)
    pad = 0.02 * size
    if not a.keep_frame and plane is not None:
        lo[1] = 0.0                               # nothing below the table
    V = np.asarray(mesh.vertices)
    outside = np.any((V < lo - pad) | (V > hi + pad), axis=1)
    if not a.keep_frame and plane is not None:
        outside |= V[:, 1] < 0.0
    mesh.remove_vertices_by_mask(outside)
    V, F = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    print(f"Poisson depth {a.depth}: {len(F):,} triangles ({time.time() - t0:.1f}s)")

    # --- cleanup (ours)
    V, F = keep_largest_component(V, F)
    V, F, n_holes = fill_holes(V, F)
    print(f"Largest piece: {len(F):,} triangles, filled {n_holes} holes, "
          f"watertight={is_watertight(F)}")
    s = a.scale
    if a.scale_json:
        with open(a.scale_json) as f:
            s = json.load(f)["scale"]
    if a.real_height:
        # no marker: the object's measured height sets the scale.
        # After table removal the model stands upright, so height = y extent.
        if a.keep_frame or plane is None:
            print("  warning: no table found, so 'up' is unknown; using the y extent as height")
        s = a.real_height / float(np.ptp(V[:, 1]))
        print(f"Scale from real height {a.real_height} m: s = {s:.5f}")
    log.update(full_triangles=int(len(F)), holes_filled=n_holes, scale=s)

    full = trimesh.Trimesh(V * s, F, process=False)
    full.export(os.path.join(a.out, "model_full.ply"))

    # --- simplification (ours or Open3D)
    t0 = time.time()
    if a.simplifier == "ours":
        V2, F2, st = simplify(V, F, a.target, verbose=True)
        log["qem"] = st
    else:
        V2, F2 = simplify_open3d(V, F, a.target)
    secs = time.time() - t0
    wt = is_watertight(F2)
    print(f"Simplified ({a.simplifier}) to {len(F2):,} triangles in {secs:.1f}s, "
          f"watertight={wt}, euler={euler_characteristic(F2)}")

    V2 = V2 * s
    dims = (V2.max(0) - V2.min(0)).tolist()
    trimesh.Trimesh(V2, F2, process=False).export(os.path.join(a.out, "model.glb"))
    trimesh.Trimesh(V2, F2, process=False).export(os.path.join(a.out, "model.ply"))
    log.update(simplifier=a.simplifier, triangles=int(len(F2)), watertight=bool(wt),
               simplify_seconds=round(secs, 2), size_xyz=dims,
               total_seconds=round(time.time() - t_start, 1))
    with open(os.path.join(a.out, "stats.json"), "w") as f:
        json.dump(log, f, indent=2)
    unit = "m" if (a.scale_json or a.real_height or a.scale != 1.0) else "model units"
    print(f"Size (x, y, z): {', '.join(f'{v:.3f}' for v in dims)} {unit}")
    print(f"Wrote {a.out}/model.glb  (import into Roblox Studio with units = meters)")
    if len(F2) > 20000:
        print("  warning: over Roblox's 20,000-triangle limit")


if __name__ == "__main__":
    main()
