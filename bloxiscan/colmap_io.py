"""Read a COLMAP model saved as text (cameras.txt, images.txt).

COLMAP saves binary by default; convert first:
    colmap model_converter --input_path dense/0/sparse \
        --output_path dense/0/sparse_txt --output_type TXT
Use the *undistorted* model (dense/0/sparse) so cameras are plain pinholes.
"""
import os
from dataclasses import dataclass

import numpy as np

from .geometry import projection_matrix


@dataclass
class Camera:
    id: int
    model: str
    width: int
    height: int
    params: np.ndarray

    @property
    def K(self):
        p = self.params
        if self.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL",
                          "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"):
            fx = fy = p[0]
            cx, cy = p[1], p[2]
        else:  # PINHOLE, OPENCV, FULL_OPENCV, OPENCV_FISHEYE, ...
            fx, fy, cx, cy = p[:4]
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)

    @property
    def has_distortion(self):
        return self.model not in ("SIMPLE_PINHOLE", "PINHOLE")


@dataclass
class Image:
    id: int
    qvec: np.ndarray   # (qw, qx, qy, qz), world -> camera rotation
    tvec: np.ndarray   # world -> camera translation
    camera_id: int
    name: str

    @property
    def R(self):
        return qvec_to_rotmat(self.qvec)

    @property
    def center(self):
        """Camera center in world coordinates: C = -R^T t."""
        return -self.R.T @ self.tvec

    def P(self, camera):
        return projection_matrix(camera.K, self.R, self.tvec)


def qvec_to_rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * z * x + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * z * x - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])


def read_cameras_text(path):
    cams = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            el = line.split()
            cams[int(el[0])] = Camera(int(el[0]), el[1], int(el[2]), int(el[3]),
                                      np.array([float(v) for v in el[4:]]))
    return cams


def read_images_text(path):
    """images.txt stores two lines per image; the second (2D points) may be empty."""
    imgs = {}
    with open(path) as f:
        lines = iter(f.read().splitlines())
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        el = line.split()
        img = Image(int(el[0]), np.array([float(v) for v in el[1:5]]),
                    np.array([float(v) for v in el[5:8]]), int(el[8]),
                    " ".join(el[9:]))
        imgs[img.id] = img
        next(lines, None)  # skip the 2D-points line
    return imgs


def read_model_text(model_dir):
    return (read_cameras_text(os.path.join(model_dir, "cameras.txt")),
            read_images_text(os.path.join(model_dir, "images.txt")))
