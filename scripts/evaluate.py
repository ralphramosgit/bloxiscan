"""Accuracy of a mesh against ground truth.

DTU protocol (mesh must be in DTU's frame, millimetres):
    python scripts/evaluate.py out/scan24/model.ply --dtu-dir DTU_GT --scan 24 --tau 1.0

Any ground truth point cloud or mesh (same units as your mesh):
    python scripts/evaluate.py out/x/model.ply --gt gt.ply --tau 0.002

Prints JSON with accuracy, completeness, Chamfer, and precision/recall/F-score.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bloxiscan import dtu, metrics  # noqa: E402


def load_points(path, n=500_000):
    g = trimesh.load(path, process=False)
    if isinstance(g, trimesh.Trimesh) and len(g.faces):
        return metrics.sample_surface(g.vertices, g.faces, n, rng=1)
    return np.asarray(g.vertices)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh")
    ap.add_argument("--gt")
    ap.add_argument("--dtu-dir")
    ap.add_argument("--scan", type=int)
    ap.add_argument("--tau", type=float, required=True, help="F-score threshold (mesh units)")
    ap.add_argument("--samples", type=int, default=300_000)
    ap.add_argument("--out", help="write the JSON here too")
    a = ap.parse_args()

    m = trimesh.load(a.mesh, process=False)
    V, F = np.asarray(m.vertices), np.asarray(m.faces)
    result = {"mesh": a.mesh, "triangles": int(len(F))}
    if a.dtu_dir:
        result.update(dtu.evaluate_dtu(V, F, a.scan, a.dtu_dir))
        gt = dtu.above_plane(dtu.load_stl(a.dtu_dir, a.scan), dtu.load_plane(a.dtu_dir, a.scan))
        ob, bb, res = dtu.load_obsmask(a.dtu_dir, a.scan)
        pred = metrics.sample_surface(V, F, a.samples, rng=0)
        _, pred_obs = dtu.mask_points(pred, ob, bb, res)
        result.update(metrics.fscore(pred_obs, gt, a.tau))
    elif a.gt:
        result.update(metrics.mesh_vs_points(V, F, load_points(a.gt), a.tau, a.samples))
    else:
        sys.exit("Give --gt or --dtu-dir/--scan")
    txt = json.dumps(result, indent=2)
    print(txt)
    if a.out:
        Path(a.out).write_text(txt)


if __name__ == "__main__":
    main()
