"""Colour texture from the video frames (own implementation; xatlas lays out UVs).

Course concepts: pinhole projection, visibility (ray casting), barycentric
coordinates, weighted multi-view blending.

The mesh only has shape. The frames have colour, and the cameras say where
each frame was taken from, so every point on the surface can be projected
into the frames that see it and coloured from them:
  1. lib   xatlas cuts the mesh into charts and packs them into a square
           image: every triangle gets a place in the texture (UV coordinates).
  2. ours  rasterize the charts: for every texel, the triangle that covers it
           and the barycentric coordinates -> its 3D point and normal.
  3. ours  visibility: a triangle is seen by a frame if it faces the camera
           and a ray from the camera to it hits nothing first (Open3D's ray
           casting is the library part).
  4. ours  colour: project each texel into every frame that sees it and
           average the pixels, weighted by cos(view angle)^power so head-on
           views dominate over grazing ones. Masks, if given, reject samples
           that land on the background.
  5. ours  texels no frame saw (the bottom, on the table) take the colour of
           the nearest seen texel in 3D; then each chart is padded by a few
           texels so filtering in the game engine does not bleed in black.
"""
import cv2
import numpy as np
import open3d as o3d
import xatlas
from scipy.spatial import cKDTree


def sample_bilinear(img, x, y):
    """Colours at real-valued array coordinates (x right, y down; integers are
    pixel centres), interpolated between the 4 nearest pixels."""
    h, w = img.shape[:2]
    x = np.clip(x, 0, w - 1)
    y = np.clip(y, 0, h - 1)
    x0 = np.minimum(np.floor(x).astype(np.int64), w - 2)
    y0 = np.minimum(np.floor(y).astype(np.int64), h - 2)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    I = img.astype(np.float32)
    return ((1 - fx) * (1 - fy) * I[y0, x0] + fx * (1 - fy) * I[y0, x0 + 1]
            + (1 - fx) * fy * I[y0 + 1, x0] + fx * fy * I[y0 + 1, x0 + 1])


def uv_layout(V, F, size):
    """xatlas charts. Returns (vmap, F_uv, uv): new vertex i is old vertex
    vmap[i]; uv in [0, 1] with v up (OpenGL / trimesh convention)."""
    atlas = xatlas.Atlas()
    atlas.add_mesh(np.asarray(V, np.float32), np.asarray(F, np.uint32))
    pack = xatlas.PackOptions()
    pack.resolution = size
    pack.padding = 4
    pack.bilinear = True
    atlas.generate(pack_options=pack)
    vmap, F_uv, uv = atlas[0]
    return vmap.astype(np.int64), F_uv.astype(np.int64), uv.astype(np.float64)


def rasterize(uv, F_uv, size):
    """For every texel: the face covering it (-1 = none) and barycentric
    coordinates. Texel (x, y) has uv = (x / (size-1), 1 - y / (size-1))."""
    px = np.c_[uv[:, 0] * (size - 1), (1.0 - uv[:, 1]) * (size - 1)]
    face_id = np.full((size, size), -1, np.int32)
    shift = 4                                           # sub-pixel precision
    tri = np.round(px[F_uv] * (1 << shift)).astype(np.int32)
    for f in range(len(F_uv)):
        cv2.fillConvexPoly(face_id, tri[f], int(f), lineType=cv2.LINE_8, shift=shift)
    ys, xs = np.nonzero(face_id >= 0)
    f = face_id[ys, xs]
    p0, p1, p2 = px[F_uv[f, 0]], px[F_uv[f, 1]], px[F_uv[f, 2]]
    q = np.c_[xs, ys].astype(float)
    # solve q - p0 = b1 (p1 - p0) + b2 (p2 - p0) per texel (2x2, Cramer's rule)
    e1, e2, r = p1 - p0, p2 - p0, q - p0
    det = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    det = np.where(np.abs(det) < 1e-12, 1e-12, det)
    b1 = (r[:, 0] * e2[:, 1] - r[:, 1] * e2[:, 0]) / det
    b2 = (e1[:, 0] * r[:, 1] - e1[:, 1] * r[:, 0]) / det
    bary = np.clip(np.c_[1 - b1 - b2, b1, b2], 0.0, 1.0)
    bary /= bary.sum(1, keepdims=True)
    return face_id, ys, xs, f, bary


def visible_faces(V, F, centers_cam, eps):
    """(n_faces, n_views) bool: face centre seen from each camera centre."""
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(np.asarray(V, np.float32)),
                        o3d.core.Tensor(np.asarray(F, np.uint32)))
    tri = V[F]
    ctr = tri.mean(1)
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    target = ctr + eps * nrm                     # just outside, to avoid self-hits
    vis = np.zeros((len(F), len(centers_cam)), bool)
    for j, C in enumerate(centers_cam):
        d = target - C
        dist = np.linalg.norm(d, axis=1)
        d /= dist[:, None]
        front = np.einsum("ij,ij->i", nrm, -d) > 0
        rays = np.c_[np.repeat(C[None], len(F), 0), d].astype(np.float32)
        t = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
        vis[:, j] = front & (t >= dist - eps)
    return vis


def bake_texture(V, F, views, size=1024, power=4.0, pad=6):
    """Texture for mesh (V, F) from calibrated frames.

    V: (n,3) vertices in the cameras' (COLMAP) frame. views: list of dicts
    with K (3x3), R, t (world -> camera), image (HxWx3 RGB uint8) and
    optional mask (HxW bool); image and mask may also be functions that
    load them, so only one frame is in memory at a time.
    Returns (vmap, F_uv, uv, texture HxWx3 uint8, info).
    """
    V = np.asarray(V, float)
    F = np.asarray(F, np.int64)
    vmap, F_uv, uv = uv_layout(V, F, size)
    face_id, ys, xs, f, bary = rasterize(uv, F_uv, size)
    Vn = V[vmap]
    P3 = np.einsum("ij,ijk->ik", bary, Vn[F_uv[f]])          # texel 3D points
    tri = V[F]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
    vn = np.zeros_like(V)                                      # smooth vertex normals
    for k in range(3):
        np.add.at(vn, F[:, k], fn)
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
    N3 = np.einsum("ij,ijk->ik", bary, vn[vmap][F_uv[f]])
    N3 /= np.maximum(np.linalg.norm(N3, axis=1, keepdims=True), 1e-12)

    centers = [-v["R"].T @ v["t"] for v in views]
    eps = 1e-3 * float(np.linalg.norm(np.ptp(V, axis=0)))
    vis = visible_faces(Vn, F_uv, centers, eps)              # same surface, xatlas order

    acc = np.zeros((len(P3), 3))
    wsum = np.zeros(len(P3))
    for j, v in enumerate(views):
        sel = np.nonzero(vis[f, j])[0]
        if len(sel) == 0:
            continue
        X = P3[sel] @ v["R"].T + v["t"]
        x = X @ v["K"].T
        u, w_ = x[:, 0] / x[:, 2], x[:, 1] / x[:, 2]           # COLMAP pixel coords
        img = v["image"]() if callable(v["image"]) else v["image"]
        mask = v.get("mask")
        mask = mask() if callable(mask) else mask
        h, w = img.shape[:2]
        ok = (x[:, 2] > 0) & (u >= 0) & (u < w) & (w_ >= 0) & (w_ < h)
        if mask is not None:
            ok[ok] = mask[w_[ok].astype(int), u[ok].astype(int)]
        sel, u, w_ = sel[ok], u[ok], w_[ok]
        view_dir = centers[j] - P3[sel]
        view_dir /= np.linalg.norm(view_dir, axis=1, keepdims=True)
        cos = np.clip(np.einsum("ij,ij->i", N3[sel], view_dir), 0.0, 1.0)
        weight = cos ** power
        col = sample_bilinear(img, u - 0.5, w_ - 0.5)          # COLMAP centres are at +0.5
        acc[sel] += weight[:, None] * col
        wsum[sel] += weight

    seen = wsum > 1e-6
    colors = np.zeros((len(P3), 3))
    colors[seen] = acc[seen] / wsum[seen, None]
    if (~seen).any() and seen.any():                       # never seen: nearest seen texel
        _, nn = cKDTree(P3[seen]).query(P3[~seen])
        colors[~seen] = colors[seen][nn]

    tex = np.zeros((size, size, 3), np.float32)
    tex[ys, xs] = colors
    filled = np.zeros((size, size), np.float32)
    filled[ys, xs] = 1.0
    for _ in range(pad):                                   # grow charts outward
        num = cv2.blur(tex * filled[..., None], (3, 3))
        den = cv2.blur(filled, (3, 3))
        grow = (den > 0) & (filled == 0)
        tex[grow] = num[grow] / den[grow, None]
        filled[grow] = 1.0
    info = {"texture_size": size, "texels": int(len(P3)),
            "texels_seen": float(seen.mean()), "views": len(views)}
    return vmap, F_uv, uv, np.clip(tex, 0, 255).astype(np.uint8), info
