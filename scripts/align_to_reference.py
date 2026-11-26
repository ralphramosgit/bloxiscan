"""Move a mesh from COLMAP's frame into a reference frame (e.g. DTU's) using
our Umeyama similarity fit on camera centers.

    python scripts/align_to_reference.py --model scan24/sparse_txt \
        --idr-npz scan24/cameras.npz --mesh out/scan24/model_full.ply \
        --out out/scan24/model_full_dtu.ply

Reference camera centers come from either
  --reference centers.txt   lines of "image_name x y z", or
  --idr-npz cameras.npz     IDR/NeuS format with world_mat_<i>; image i is
                            matched to the i-th COLMAP image in name order.

Shortcut for the 2DGS DTU download, whose COLMAP poses live in a normalized
frame: apply that scan's scale_mat_0 directly (what 2DGS's own evaluator does):
    python scripts/align_to_reference.py --scale-mat dtu/scan24/cameras.npz \
        --mesh out/scan24/model_full.ply --out out/scan24/model_full_world.ply
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bloxiscan.colmap_io import read_model_text  # noqa: E402
from bloxiscan.geometry import (apply_similarity, camera_center,  # noqa: E402
                                similarity_matrix, umeyama)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="COLMAP text model (not needed with --scale-mat)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--reference")
    g.add_argument("--idr-npz")
    g.add_argument("--scale-mat", help="cameras.npz: apply its scale_mat_0")
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if a.scale_mat:
        S = np.load(a.scale_mat)["scale_mat_0"]
        mesh = trimesh.load(a.mesh, process=False)
        V = np.asarray(mesh.vertices) * S[0, 0] + S[:3, 3]
        trimesh.Trimesh(V, np.asarray(mesh.faces), process=False).export(a.out)
        print(f"Applied scale_mat_0 (scale {S[0, 0]:.4f}); wrote {a.out}")
        return
    if not a.model:
        sys.exit("--model is required with --reference or --idr-npz")

    _, imgs = read_model_text(a.model)
    by_name = {im.name: im for im in imgs.values()}
    src, dst = [], []
    if a.reference:
        for line in open(a.reference):
            p = line.split()
            if len(p) == 4 and p[0] in by_name:
                src.append(by_name[p[0]].center)
                dst.append([float(v) for v in p[1:]])
    else:
        cams = np.load(a.idr_npz)
        names = sorted(by_name)  # COLMAP lists only the images it registered
        n_world = len([k for k in cams.files if k.startswith("world_mat_") and "inv" not in k])
        if len(names) != n_world:
            print(f"warning: {len(names)} COLMAP images vs {n_world} world_mats; "
                  "check that images are matched correctly")
        for i, name in enumerate(names):
            stem = Path(name).stem       # "000012.png" -> world_mat_12
            key = f"world_mat_{int(stem) if stem.isdigit() else i}"
            if key in cams.files:
                src.append(by_name[name].center)
                dst.append(camera_center(cams[key][:3, :4]))
    src, dst = np.array(src), np.array(dst)
    if len(src) < 3:
        sys.exit("Need at least 3 matched cameras.")

    s, R, t = umeyama(src, dst)
    res = np.linalg.norm(apply_similarity(s, R, t, src) - dst, axis=1)
    print(f"{len(src)} cameras matched; scale {s:.5f}; residual RMS {np.sqrt((res**2).mean()):.4f}, "
          f"max {res.max():.4f} (reference units)")

    mesh = trimesh.load(a.mesh, process=False)
    V = apply_similarity(s, R, t, np.asarray(mesh.vertices))
    trimesh.Trimesh(V, np.asarray(mesh.faces), process=False).export(a.out)
    with open(Path(a.out).with_suffix(".transform.json"), "w") as f:
        json.dump({"scale": s, "R": R.tolist(), "t": t.tolist(),
                   "matrix": similarity_matrix(s, R, t).tolist(),
                   "residual_rms": float(np.sqrt((res**2).mean()))}, f, indent=2)
    print(f"Wrote {a.out}")


if __name__ == "__main__":
    main()
