"""Cut one object out of a scanned scene (own implementation).

Course concepts: RANSAC plane fitting, least-squares line intersection,
robust statistics (medians, histogram peaks).

The object stands on a surface (table, shirt, floor) and the camera circled
it. So:
  1. The surface is the dominant plane (RANSAC) -> its normal is "up".
  2. Every frame looks at the object -> the viewing axes meet at the object
     (orbit_center). Dropping that point onto the plane gives a vertical axis.
  3. Re-center the axis on the points themselves (iterated median of their
     horizontal offsets), because people rarely circle exactly around the
     middle of the object.
  4. The object's outline shows up as a peak in the histogram of horizontal
     distances from that axis. Keep points inside a cylinder slightly wider
     than that peak, above the surface and below the object's top.
Background clutter and wrinkles in the cloth fall outside that cylinder.
"""
import numpy as np


def horizontal_offsets(points, axis_point, up):
    v = points - axis_point
    return v - np.outer(v @ up, up)


def isolate_object(points, up, d, orbit_point, view_half, plane_thresh,
                   radius_scale=1.35):
    """Return a boolean mask of points that belong to the object.

    points: (N,3); up, d: plane with up . x + d = 0, object on the positive
    side; orbit_point: where the camera kept looking; view_half: half the
    view height at the object's distance (sets the search size);
    plane_thresh: points closer than this to the plane are surface.
    """
    h = points @ up + d
    axis = orbit_point - (orbit_point @ up + d) * up
    search = 0.3 * view_half

    r = np.linalg.norm(horizontal_offsets(points, axis, up), axis=1)
    near = (h > plane_thresh) & (r < 0.5 * view_half)
    if near.sum() < 20:
        return near, None
    top = np.percentile(h[near & (r < search)], 99)

    for _ in range(5):                       # re-center on the object
        off = horizontal_offsets(points, axis, up)
        r = np.linalg.norm(off, axis=1)
        band = (h > 0.4 * top) & (h < 0.9 * top) & (r < search)
        if band.sum() < 10:
            break
        axis = axis + np.median(off[band], axis=0)

    r = np.linalg.norm(horizontal_offsets(points, axis, up), axis=1)
    band = (h > 0.4 * top) & (h < 0.9 * top) & (r < search)
    hist, edges = np.histogram(r[band], bins=30, range=(0, search))
    peak = 0.5 * (edges[hist.argmax()] + edges[hist.argmax() + 1])
    radius = radius_scale * peak
    mask = (h > plane_thresh) & (h < 1.05 * top) & (r < radius)
    return mask, {"axis": axis, "radius": float(radius), "top": float(top)}
