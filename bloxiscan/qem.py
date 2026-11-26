"""Quadric Error Metric (QEM) mesh simplification, written from scratch.

Garland & Heckbert, "Surface Simplification Using Quadric Error Metrics",
SIGGRAPH 1997. This is what gets a 300k-triangle scan under Roblox's
20,000-triangle limit.

Idea
----
* Every face lies on a plane p = [a, b, c, d] with a^2 + b^2 + c^2 = 1, so the
  squared distance of a point v (homogeneous [x, y, z, 1]) to it is (p . v)^2
  = v^T (p p^T) v.  K_p = p p^T is the face's "fundamental quadric".
* Each vertex starts with Q = sum of K_p over its faces (here weighted by face
  area, so big faces count more).
* Collapsing edge (a, b) into one point v costs  v^T (Q_a + Q_b) v , the summed
  squared distance to all the original planes around a and b. The best v solves
  a 3x3 linear system; if that system is singular we try a, b and the midpoint.
* Repeatedly collapse the cheapest edge (priority queue) until the face budget
  is met. Flat regions are cheap, sharp creases are expensive, so detail
  survives where it matters.

Safety checks added to the basic algorithm (also standard practice):
* Link condition: only collapse if the neighbors that a and b share are exactly
  the opposite corners of their shared faces. This keeps a closed (watertight)
  mesh closed and manifold, which Roblox needs.
* Flip check: reject a collapse that turns any surrounding face upside down.
* Boundary preservation: open borders get extra "wall" planes so they don't
  shrink (Garland & Heckbert, section 5).
"""
import heapq
import time

import numpy as np


# ---------------------------------------------------------------- quadrics
def face_planes(V, F):
    """Unit-normal plane [a, b, c, d] and area of every face."""
    v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    double_area = np.linalg.norm(n, axis=1)
    ok = double_area > 1e-30
    n[ok] /= double_area[ok, None]
    n[~ok] = 0.0
    d = -np.einsum("ij,ij->i", n, v0)
    return np.c_[n, d], 0.5 * double_area


def _edge_table(F):
    """Directed edges (as they appear in faces), their face index, and how
    many faces use each undirected edge."""
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    fidx = np.tile(np.arange(len(F)), 3)
    _, inv, counts = np.unique(np.sort(e, axis=1), axis=0,
                               return_inverse=True, return_counts=True)
    return e, fidx, counts[inv.reshape(-1)]


def initial_quadrics(V, F, area_weighted=True, boundary_weight=100.0):
    """Q (n_vertices, 4, 4): sum of plane quadrics around each vertex."""
    planes, areas = face_planes(V, F)
    w = areas if area_weighted else np.ones(len(F))
    K = np.einsum("fi,fj->fij", planes, planes) * w[:, None, None]
    Q = np.zeros((len(V), 4, 4))
    for k in range(3):
        np.add.at(Q, F[:, k], K)

    if boundary_weight > 0:
        e, fidx, count = _edge_table(F)
        is_b = count == 1
        if is_b.any():
            be, bf = e[is_b], fidx[is_b]
            a, b = V[be[:, 0]], V[be[:, 1]]
            m = np.cross(b - a, planes[bf, :3])  # in-face, perpendicular to edge
            ln = np.linalg.norm(m, axis=1)
            ok = ln > 1e-30
            m = m[ok] / ln[ok, None]
            p = np.c_[m, -np.einsum("ij,ij->i", m, a[ok])]
            len2 = np.einsum("ij,ij->i", b[ok] - a[ok], b[ok] - a[ok])
            Kb = np.einsum("fi,fj->fij", p, p) * (boundary_weight * len2)[:, None, None]
            np.add.at(Q, be[ok, 0], Kb)
            np.add.at(Q, be[ok, 1], Kb)
    return Q


def optimal_positions(Qs, pa, pb):
    """Best collapse point and its cost for a batch of edges.

    Minimizing v^T Q v over v = [x, y, z, 1] gives  A v_xyz = -q,  with
    A = Q[:3, :3] and q = Q[:3, 3]. Fall back to the cheapest of a, b and the
    midpoint when A is (nearly) singular or the solution lands far away.
    """
    k = len(Qs)
    A = Qs[:, :3, :3]
    rhs = -Qs[:, :3, 3]
    pos = 0.5 * (pa + pb)
    scale = np.abs(A).reshape(k, -1).max(axis=1)
    det = np.linalg.det(A)
    solved = np.abs(det) > 1e-9 * scale ** 3 + 1e-300
    if solved.any():
        idx = np.flatnonzero(solved)
        sol = np.linalg.solve(A[idx], rhs[idx][..., None])[..., 0]
        edge_len = np.linalg.norm(pa[idx] - pb[idx], axis=1)
        near = np.linalg.norm(sol - pos[idx], axis=1) <= 2.0 * edge_len + 1e-12
        pos[idx[near]] = sol[near]
        solved[idx[~near]] = False
    fb = ~solved
    if fb.any():
        cands = np.stack([pa[fb], pb[fb], 0.5 * (pa[fb] + pb[fb])], axis=1)
        ch = np.concatenate([cands, np.ones(cands.shape[:2] + (1,))], axis=-1)
        c = np.einsum("kci,kij,kcj->kc", ch, Qs[fb], ch)
        pos[fb] = cands[np.arange(len(c)), c.argmin(axis=1)]
    ph = np.c_[pos, np.ones(k)]
    cost = np.einsum("ki,kij,kj->k", ph, Qs, ph)
    return pos, np.maximum(cost, 0.0)


# ---------------------------------------------------------------- simplify
def simplify(V, F, target_faces, area_weighted=True, boundary_weight=100.0,
             flip_cos=0.0, verbose=False):
    """Simplify a triangle mesh to at most `target_faces` faces.

    V: (n, 3) float vertices, F: (m, 3) int faces.
    Returns (V_new, F_new, stats).
    """
    t0 = time.time()
    V = np.array(V, dtype=np.float64)
    F = np.array(F, dtype=np.int64)
    n_v, n_f = len(V), len(F)
    stats = {"faces_in": n_f, "collapses": 0, "rejected_link": 0,
             "rejected_flip": 0}
    if target_faces >= n_f:
        stats.update(faces_out=n_f, seconds=0.0)
        return V, F, stats

    Q = initial_quadrics(V, F, area_weighted, boundary_weight)
    faces = F.tolist()                     # python lists: fast single edits
    face_alive = [True] * n_f
    v_alive = [True] * n_v
    version = [0] * n_v
    vf = [set() for _ in range(n_v)]       # vertex -> faces around it
    for fi, (a, b, c) in enumerate(faces):
        vf[a].add(fi)
        vf[b].add(fi)
        vf[c].add(fi)

    _, _, count = _edge_table(F)
    has_boundary = bool((count == 1).any())

    def neighbors(v):
        out = set()
        for f in vf[v]:
            out.update(faces[f])
        out.discard(v)
        return out

    def is_boundary_vertex(v):
        return any(len(vf[v] & vf[w]) == 1 for w in neighbors(v))

    def flips(ring, a, b, p):
        tris = np.array([faces[f] for f in ring])
        old = V[tris]
        new = old.copy()
        new[(tris == a) | (tris == b)] = p
        n_old = np.cross(old[:, 1] - old[:, 0], old[:, 2] - old[:, 0])
        n_new = np.cross(new[:, 1] - new[:, 0], new[:, 2] - new[:, 0])
        dot = np.einsum("ij,ij->i", n_old, n_new)
        mag = np.linalg.norm(n_old, axis=1) * np.linalg.norm(n_new, axis=1)
        return bool(np.any(dot <= flip_cos * mag))

    # initial priority queue over all unique edges
    edges = np.unique(np.sort(np.concatenate(
        [F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1), axis=0)
    pos, cost = optimal_positions(Q[edges[:, 0]] + Q[edges[:, 1]],
                                  V[edges[:, 0]], V[edges[:, 1]])
    heap = [(float(cost[i]), i, int(edges[i, 0]), int(edges[i, 1]), 0, 0,
             float(pos[i, 0]), float(pos[i, 1]), float(pos[i, 2]))
            for i in range(len(edges))]
    heapq.heapify(heap)
    counter = len(heap)
    alive_faces = n_f

    while alive_faces > target_faces and heap:
        _, _, a, b, ver_a, ver_b, px, py, pz = heapq.heappop(heap)
        # lazy deletion: skip entries made stale by earlier collapses
        if not (v_alive[a] and v_alive[b]) or version[a] != ver_a or version[b] != ver_b:
            continue
        shared = vf[a] & vf[b]
        if not shared:
            continue

        # link condition (keeps the mesh manifold / watertight)
        opposite = set()
        for f in shared:
            opposite.update(faces[f])
        opposite -= {a, b}
        if (neighbors(a) & neighbors(b)) != opposite:
            stats["rejected_link"] += 1
            continue
        if has_boundary and len(shared) == 2 and is_boundary_vertex(a) and is_boundary_vertex(b):
            stats["rejected_link"] += 1   # interior edge joining two borders
            continue

        p = np.array([px, py, pz])
        ring = (vf[a] | vf[b]) - shared
        if ring and flips(ring, a, b, p):
            stats["rejected_flip"] += 1
            continue

        # --- collapse b into a
        for f in shared:
            face_alive[f] = False
            for v in faces[f]:
                vf[v].discard(f)
        for f in vf[b]:
            tri = faces[f]
            tri[tri.index(b)] = a
            vf[a].add(f)
        vf[b] = set()
        v_alive[b] = False
        V[a] = p
        Q[a] += Q[b]
        version[a] += 1
        version[b] += 1
        alive_faces -= len(shared)
        stats["collapses"] += 1

        # costs changed only for edges touching a
        nbrs = list(neighbors(a))
        if nbrs:
            Qs = Q[a][None] + Q[nbrs]
            P2, C2 = optimal_positions(Qs, np.repeat(V[a][None], len(nbrs), 0), V[nbrs])
            for w, c, pp in zip(nbrs, C2.tolist(), P2.tolist()):
                counter += 1
                heapq.heappush(heap, (c, counter, a, w, version[a], version[w], *pp))

        if verbose and stats["collapses"] % 20000 == 0:
            print(f"  QEM: {alive_faces} faces left")

    keep = [f for f in range(n_f) if face_alive[f]]
    F_new = np.array([faces[f] for f in keep], dtype=np.int64).reshape(-1, 3)
    used = np.unique(F_new)
    remap = -np.ones(n_v, dtype=np.int64)
    remap[used] = np.arange(len(used))
    stats.update(faces_out=len(F_new), seconds=round(time.time() - t0, 2))
    return V[used], remap[F_new], stats
