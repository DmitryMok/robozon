"""Увеличенный кроп силуэта Ручки с ракурса 90 (сверху): показывает, сколько пикселей
реально занимает тонкий объект в боевом кадре 2592x1944. Выход: exports/report_img/sil_pen_zoom.png
"""
from pathlib import Path

import cv2
import numpy as np

from roboson_tools.geometry.mesh_io import Mesh, load_stl
from roboson_tools.silhouette import camera as camera_silhouette

FOV_DEG = 68.0
RESOLUTION = (2592, 1944)

mesh = load_stl(Path("assets/stl/Ручка.stl"))
zmin = mesh.vertices[:, 2].min()
g = Mesh(vertices=mesh.vertices - np.array([0.0, 0.0, zmin]), faces=mesh.faces)

tiles = []
for angle, d in [(90.0, 2000.0), (0.0, 2000.0)]:
    sil = camera_silhouette.build_silhouette(g, angle, RESOLUTION, FOV_DEG, d, "center")
    ys, xs = np.nonzero(sil.mask)
    m = 12
    crop = sil.mask[ys.min() - m : ys.max() + m, xs.min() - m : xs.max() + m]
    img = 255 - crop.astype(np.uint8) * 255
    scale = 6
    img = cv2.resize(img, (img.shape[1] * scale, img.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h_px, w_px = crop.shape[0] - 2 * m, crop.shape[1] - 2 * m
    label = f"{angle:g} deg, crop x{scale}: {w_px}x{h_px} px"
    cv2.rectangle(img, (0, 0), (img.shape[1] - 1, 36), (240, 240, 240), -1)
    cv2.putText(img, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2)
    cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), (180, 180, 180), 1)
    tiles.append(img)

h = max(t.shape[0] for t in tiles)
tiles = [
    cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    for t in tiles
]
out = np.hstack(tiles)
cv2.imwrite("exports/report_img/sil_pen_zoom.png", out)
print("ok", out.shape)
