"""Make a printable ArUco marker (DICT_4X4_50, id 0 by default).

    python scripts/make_marker.py --cm 10 --out aruco_id0.png

Print at 100% scale (no "fit to page"), then measure the black square's side
with a ruler: that measured length is L for scripts/aruco_scale.py.
"""
import argparse

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", type=int, default=0)
    ap.add_argument("--dict", default="DICT_4X4_50")
    ap.add_argument("--cm", type=float, default=10.0, help="side length of the black square")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--out", default="aruco_id0.png")
    a = ap.parse_args()

    side_px = int(round(a.cm / 2.54 * a.dpi))
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, a.dict))
    marker = cv2.aruco.generateImageMarker(d, a.id, side_px)
    margin = side_px // 4  # white quiet zone so the detector finds the border
    page = np.full((side_px + 2 * margin, side_px + 2 * margin), 255, np.uint8)
    page[margin:margin + side_px, margin:margin + side_px] = marker
    # PNG with a DPI tag so most print dialogs print it at true size
    ok, buf = cv2.imencode(".png", page)
    with open(a.out, "wb") as f:
        f.write(_set_png_dpi(buf.tobytes(), a.dpi))
    print(f"Wrote {a.out}: {a.cm} cm marker at {a.dpi} dpi. Print at 100% and measure it.")


def _set_png_dpi(png: bytes, dpi: int) -> bytes:
    """Insert a pHYs chunk (pixels per metre) after IHDR."""
    import struct
    import zlib
    ppm = int(round(dpi / 0.0254))
    data = struct.pack(">IIB", ppm, ppm, 1)
    chunk = struct.pack(">I", len(data)) + b"pHYs" + data
    chunk += struct.pack(">I", zlib.crc32(b"pHYs" + data) & 0xFFFFFFFF)
    ihdr_end = 8 + 4 + 4 + 13 + 4
    return png[:ihdr_end] + chunk + png[ihdr_end:]


if __name__ == "__main__":
    main()
