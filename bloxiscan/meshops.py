"""Mesh cleanup for Roblox, written with NumPy/SciPy.

* keep_largest_component: drop floating debris
* boundary_loops + fill_holes: close holes with a fan of triangles
* is_watertight / euler_characteristic: checks before export
"""
from collections import defaultdict

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def _directed_edges(F):
    return np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])


def undirected_edge_counts(F):
    """Unique undirected edges and how many faces use each."""
    return np.unique(np.sort(_directed_edges(F), axis=1), axis=0, return_counts=True)


def is_watertight(F):
    """Closed, consistently oriented 2-manifold: every undirected edge is used
    by exactly two faces, and every directed edge appears once (the two faces
    traverse a shared edge in opposite directions)."""
    if len(F) == 0:
        return False
    _, counts = undirected_edge_counts(F)
    if not np.all(counts == 2):
        return False
    e = _directed_edges(F)
    return len(np.unique(e, axis=0)) == len(e)


def euler_characteristic(F):
    """V - E + F over referenced vertices. A closed genus-0 surface gives 2."""
    edges, _ = undirected_edge_counts(F)
    return len(np.unique(F)) - len(edges) + len(F)


def remove_unreferenced(V, F):
    used = np.unique(F)
    remap = -np.ones(len(V), dtype=np.int64)
    remap[used] = np.arange(len(used))
    return V[used], remap[F]


def keep_largest_component(V, F):
    """Keep the connected piece with the most faces."""
    e = _directed_edges(F)
    n = len(V)
    adj = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    n_comp, labels = connected_components(adj, directed=False)
    if n_comp == 1:
        return remove_unreferenced(V, F)
    face_label = labels[F[:, 0]]
    best = np.bincount(face_label).argmax()
    return remove_unreferenced(V, F[face_label == best])


def boundary_loops(F):
    """Ordered vertex loops around each hole.

    A boundary edge is used by one face only. We follow boundary edges in the
    direction they run inside their face, so each loop is ordered consistently.
    """
    e = _directed_edges(F)
    key = np.sort(e, axis=1)
    _, inv, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    b = e[counts[inv.reshape(-1)] == 1]
    out = defaultdict(list)
    for a, c in b.tolist():
        out[a].append(c)
    used, loops = set(), []
    for a0, c0 in b.tolist():
        if (a0, c0) in used:
            continue
        used.add((a0, c0))
        loop, nxt = [a0], c0
        closed = True
        while nxt != a0:
            loop.append(nxt)
            options = [w for w in out[nxt] if (nxt, w) not in used]
            if not options:
                closed = False
                break
            used.add((nxt, options[0]))
            nxt = options[0]
        if closed and len(loop) >= 3:
            loops.append(loop)
    return loops


def loop_flatness(P):
    """How far a hole's rim is from flat: smallest / largest singular value of
    the centred rim points. ~0 for a flat rim (a fan closes it well), large
    for a rim that winds around in 3D (a fan makes long crossing slivers)."""
    s = np.linalg.svd(P - P.mean(axis=0), compute_uv=False)
    return float(s[2] / max(s[0], 1e-12))


def fill_holes(V, F, max_hole_edges=None, max_flatness=None, small_hole_edges=30):
    """Close holes with a fan of triangles around the hole's centroid.

    For a boundary edge a->b (direction inside its existing face), the new face
    must use b->a to keep orientations consistent: triangle (b, a, centroid).
    Holes with more than max_hole_edges edges are left open, and so are holes
    with more than small_hole_edges edges whose rim is less flat than
    max_flatness (see loop_flatness). Small holes are always filled: their
    fan is small, and their flatness is mostly noise.
    Returns (V, F, number_of_holes_filled).
    """
    V = np.asarray(V, float)
    F = np.asarray(F, np.int64)
    new_v, new_f, filled = [], [], 0
    for loop in boundary_loops(F):
        if max_hole_edges is not None and len(loop) > max_hole_edges:
            continue
        if (max_flatness is not None and len(loop) > small_hole_edges
                and loop_flatness(V[loop]) > max_flatness):
            continue
        c = len(V) + len(new_v)
        new_v.append(V[loop].mean(axis=0))
        for i in range(len(loop)):
            a, b = loop[i], loop[(i + 1) % len(loop)]
            new_f.append([b, a, c])
        filled += 1
    if filled:
        V = np.vstack([V, np.array(new_v)])
        F = np.vstack([F, np.array(new_f, dtype=np.int64)])
    return V, F, filled
