"""Main experiment: how much accuracy survives each triangle budget?

    python scripts/budget_sweep.py out/scan24/model_full.ply --dtu-dir DTU_GT --scan 24 \
        --tau 1.0 --out results/scan24
    python scripts/budget_sweep.py out/sneaker/model_full.ply --tau 0.002 --out results/sneaker

Simplifies the full mesh to each budget with our QEM and with Open3D's, then
measures Chamfer distance and F-score against the ground truth (DTU) or, if no
ground truth is given, against the full-resolution mesh itself.
Writes sweep.csv and sweep.png (one panel per metric, Roblox's limit marked).
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bloxiscan import dtu, metrics  # noqa: E402
from bloxiscan.meshops import is_watertight  # noqa: E402
from bloxiscan.qem import simplify  # noqa: E402

COLORS = {"ours": "#257de0", "open3d": "#f47230"}
NAMES = {"ours": "Our QEM", "open3d": "Open3D QEM"}
INK, MUTED, GRID = "#191919", "#666666", "#e6e6e6"


def run_simplifier(name, V, F, target):
    if name == "ours":
        V2, F2, _ = simplify(V, F, target)
        return V2, F2
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F))
    m = m.simplify_quadric_decimation(target_number_of_triangles=target)
    return np.asarray(m.vertices), np.asarray(m.triangles)


def plot(rows, full_row, path, units):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150)
    panels = [("chamfer", f"Chamfer distance ({units}), lower is better"),
              ("fscore", "F-score, higher is better")]
    for ax, (key, title) in zip(axes, panels):
        series = {}
        for name in COLORS:
            r = sorted([x for x in rows if x["simplifier"] == name], key=lambda x: x["triangles"])
            if r:
                series[name] = ([x["triangles"] for x in r], [x[key] for x in r])
        # direct labels at the smallest budget, placed on the side away from the
        # full-mesh line (which the lines approach) so labels never sit on it
        ref = full_row[key] if full_row is not None else 0.0
        firsts = {n: s[1][0] for n, s in series.items()}
        away = 1.0
        if firsts:
            near = min(firsts, key=lambda n: abs(firsts[n] - ref))
            away = np.sign(firsts[near] - ref) or 1.0
        for name, (xs, ys) in series.items():
            ax.plot(xs, ys, color=COLORS[name], lw=1.8, marker="o", ms=6,
                    markeredgecolor="white", markeredgewidth=1.5, label=NAMES[name], zorder=3)
            ax.annotate(NAMES[name], (xs[0], ys[0]), xytext=(6, 8 if away > 0 else -14),
                        textcoords="offset points", fontsize=8, color=INK)
        if full_row is not None:
            ax.axhline(full_row[key], color=MUTED, lw=1, ls=":", zorder=1)
            ax.annotate("full mesh", (0, full_row[key]), xycoords=("axes fraction", "data"),
                        xytext=(4, 4), textcoords="offset points", fontsize=8, color=MUTED)
        ax.axvline(20000, color=MUTED, lw=1, ls="--", zorder=1)
        # the lines meet the full-mesh value at large budgets: put the limit label
        # on the opposite side (top for Chamfer, bottom for F-score)
        at_top = key == "chamfer"
        ax.annotate("Roblox limit (20k)", (20000, 1 if at_top else 0),
                    xycoords=("data", "axes fraction"), xytext=(4, -12 if at_top else 6),
                    textcoords="offset points", fontsize=8, color=MUTED)
        ax.set_xscale("log")
        ax.set_xlabel("Triangles (log scale)", color=MUTED, fontsize=9)
        ax.set_title(title, color=INK, fontsize=10, loc="left")
        ax.grid(True, color=GRID, lw=0.8, zorder=0)
        ax.tick_params(colors=MUTED, labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK, loc="center right")
    fig.tight_layout()
    fig.savefig(path, facecolor="white")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("full_mesh")
    ap.add_argument("--gt", help="ground truth points/mesh; default = the full mesh itself")
    ap.add_argument("--dtu-dir")
    ap.add_argument("--scan", type=int)
    ap.add_argument("--tau", type=float, required=True)
    ap.add_argument("--budgets", type=int, nargs="+", default=[100000, 20000, 10000, 5000, 2000])
    ap.add_argument("--simplifiers", nargs="+", default=["ours", "open3d"])
    ap.add_argument("--samples", type=int, default=200_000)
    ap.add_argument("--units", default="mm")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    m = trimesh.load(a.full_mesh, process=False)
    V, F = np.asarray(m.vertices, float), np.asarray(m.faces, np.int64)
    if a.dtu_dir:
        ob, bb, res = dtu.load_obsmask(a.dtu_dir, a.scan)
        gt = dtu.above_plane(dtu.load_stl(a.dtu_dir, a.scan), dtu.load_plane(a.dtu_dir, a.scan))
    elif a.gt:
        g = trimesh.load(a.gt, process=False)
        gt = (metrics.sample_surface(g.vertices, g.faces, a.samples, 1)
              if isinstance(g, trimesh.Trimesh) and len(g.faces) else np.asarray(g.vertices))
    else:
        gt = metrics.sample_surface(V, F, a.samples, rng=1)

    def score(V_, F_):
        pred = metrics.sample_surface(V_, F_, a.samples, rng=0)
        if a.dtu_dir:
            _, pred = dtu.mask_points(pred, ob, bb, res)
            out = metrics.chamfer(pred, gt, max_dist=20.0)
        else:
            out = metrics.chamfer(pred, gt)
        out.update(metrics.fscore(pred, gt, a.tau))
        return out

    full_row = {"simplifier": "full", "triangles": len(F), "seconds": 0.0,
                "watertight": is_watertight(F), **score(V, F)}
    rows = [full_row]
    print(f"full: {len(F):,} tris, chamfer {full_row['chamfer']:.4f}, F {full_row['fscore']:.3f}")
    for name in a.simplifiers:
        for b in sorted(a.budgets, reverse=True):
            if b >= len(F):
                continue
            t0 = time.time()
            V2, F2 = run_simplifier(name, V, F, b)
            secs = time.time() - t0
            r = {"simplifier": name, "triangles": len(F2), "seconds": round(secs, 2),
                 "watertight": is_watertight(F2), **score(V2, F2)}
            rows.append(r)
            print(f"{name:7s} {len(F2):>7,} tris  {secs:6.1f}s  chamfer {r['chamfer']:.4f}  "
                  f"F {r['fscore']:.3f}  watertight={r['watertight']}")

    keys = ["simplifier", "triangles", "seconds", "watertight", "accuracy", "completeness",
            "chamfer", "precision", "recall", "fscore", "tau"]
    with open(os.path.join(a.out, "sweep.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    plot([r for r in rows if r["simplifier"] != "full"], full_row,
         os.path.join(a.out, "sweep.png"), a.units)
    print(f"Wrote {a.out}/sweep.csv and sweep.png")


if __name__ == "__main__":
    main()
