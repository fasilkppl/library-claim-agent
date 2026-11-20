"""Printable reference marker: ArUco 4x4 id 0, exactly 10 cm wide on an A4 page at 300 dpi.

Print at 100% (no 'fit to page'), check with a ruler that the black square is 10.0 cm, and place it flat
on a shelf front at the start of the sweep. Only needed for camera-only sweeps; WebXR sweeps get metric
scale from ARCore depth.
"""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

DPI = 300
SIDE_CM = 10.0


def main() -> None:
    px = round(SIDE_CM / 2.54 * DPI)
    a4 = (round(21.0 / 2.54 * DPI), round(29.7 / 2.54 * DPI))
    page = np.full((a4[1], a4[0]), 255, np.uint8)
    marker = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), 0, px)
    x0, y0 = (a4[0] - px) // 2, round(4 / 2.54 * DPI)
    page[y0 : y0 + px, x0 : x0 + px] = marker
    cv2.putText(
        page,
        "Claim agent reference marker - print at 100%: black square = 10.0 cm",
        (x0 - 260, y0 + px + 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.4,
        0,
        3,
    )
    out = Path(__file__).parents[1] / "docs" / "marker_10cm.pdf"
    Image.fromarray(page).save(out, resolution=DPI)
    print(out)


if __name__ == "__main__":
    main()
