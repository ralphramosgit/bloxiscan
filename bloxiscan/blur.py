"""Blur detection with the variance of the Laplacian (own implementation).

Course concepts: image filtering / convolution, second derivatives, edges.

A sharp frame has strong edges, so its Laplacian has large positive and
negative values -> high variance. A blurry frame has a weak, flat Laplacian
-> low variance. OpenCV is only used elsewhere to *check* this code
(see tests/test_blur.py), not to compute it.
"""
import numpy as np

# Discrete Laplacian: d2I/dx2 + d2I/dy2 with central differences.
LAPLACIAN_KERNEL = np.array([[0, 1, 0],
                             [1, -4, 1],
                             [0, 1, 0]], dtype=np.float64)


def to_gray(img: np.ndarray) -> np.ndarray:
    """BGR (OpenCV channel order) or gray uint8 image -> float64 luminance."""
    if img.ndim == 2:
        return img.astype(np.float64)
    b, g, r = img[..., 0], img[..., 1], img[..., 2]
    return 0.114 * b + 0.587 * g + 0.299 * r  # ITU-R BT.601 weights


def convolve2d_valid(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """2D convolution, 'valid' region only, written with array slicing.

    out[y, x] = sum_{i,j} k[i, j] * img[y + i, x + j]   (k flipped = convolution)
    Looping over the (small) kernel instead of the (large) image keeps it fast.
    """
    kh, kw = kernel.shape
    h, w = img.shape
    oh, ow = h - kh + 1, w - kw + 1
    k = kernel[::-1, ::-1]
    out = np.zeros((oh, ow), dtype=np.float64)
    for i in range(kh):
        for j in range(kw):
            if k[i, j] != 0.0:
                out += k[i, j] * img[i:i + oh, j:j + ow]
    return out


def laplacian(gray: np.ndarray) -> np.ndarray:
    """Laplacian of a gray image (interior pixels only)."""
    return convolve2d_valid(gray, LAPLACIAN_KERNEL)


def sharpness(img: np.ndarray) -> float:
    """Blur score = Var(Laplacian). Higher = sharper."""
    return float(laplacian(to_gray(img)).var())


def select_sharp(scores, window: int = 2, min_ratio: float = 0.35):
    """Pick frames to keep from a list of blur scores.

    1. In every run of `window` consecutive samples keep only the sharpest one
       (this turns e.g. 6 fps of samples into 3 fps of the best frames).
    2. Drop any kept frame whose score is below `min_ratio` x the median of the
       kept scores (catches frames that are blurry for the whole window).
    Returns the sorted list of kept indices.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if len(scores) == 0:
        return []
    window = max(1, int(window))
    best = []
    for start in range(0, len(scores), window):
        chunk = scores[start:start + window]
        best.append(start + int(np.argmax(chunk)))
    best = np.array(best)
    med = np.median(scores[best])
    keep = best[scores[best] >= min_ratio * med]
    return sorted(int(i) for i in keep)
