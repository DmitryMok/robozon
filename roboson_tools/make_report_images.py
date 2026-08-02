"""Генерация иллюстраций для справки по выбору метода замера габаритов.

Для набора показательных кейсов (простая коробка, пуфик на ребре, тонкая ручка,
шлем, бутылка) строит полосу из 4 перспективных силуэтов (ракурсы рига 0/45/90/135,
боевые параметры камеры 2592x1944 / FOV 68 / 2м, зеркала 2.6м) и параллельно
замеряет ошибку габаритов v1 (check_simple_camera_dims) и v3
(check_simple_camera_dims_v3, supersample=3, параллакс start/center/end).

Запуск (Windows venv из WSL, из корня проекта):
    python make_report_images.py
Выход: exports/report_img/*.png + exports/report_img/errors.json
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from roboson_tools.core.experiment import (
    Orientation,
    check_simple_camera_dims,
    check_simple_camera_dims_v3,
    mesh_true_dims,
)
from roboson_tools.geometry.mesh_io import Mesh, load_stl
from roboson_tools.geometry.transform import apply_orientation
from roboson_tools.silhouette import camera as camera_silhouette

FOV_DEG = 68.0
CAMERA_DISTANCE = 2000.0
RESOLUTION = (2592, 1944)
DISTANCES = {45.0: 2600.0, 135.0: 2600.0}
VIEW_ANGLES = [0.0, 45.0, 90.0, 135.0]
SUPERSAMPLE = 3
BELT_POSITIONS = ["start", "center", "end"]

ASSETS = Path("assets/stl")
OUT = Path("exports/report_img")

# (id, stl, ориентация, подпись)
CASES = [
    ("box300", "Короб 300х200х200.stl", Orientation(0, 0, 0), "Korob 300x200x200, kak est"),
    ("pouf_edge_roll90", "Пуфик.stl", Orientation(90, 0, 0), "Pufik na rebre (roll=90)"),
    ("pouf_edge_pitch90", "Пуфик.stl", Orientation(0, 90, 0), "Pufik na rebre (pitch=90)"),
    ("pouf_flat", "Пуфик.stl", Orientation(0, 0, 0), "Pufik plashmya"),
    ("pen", "Ручка.stl", Orientation(0, 0, 0), "Ruchka (tonkaya)"),
    ("helmet", "Шлем.stl", Orientation(0, 0, 0), "Shlem"),
    ("bottle_yaw0", "Бутылка.stl", Orientation(0, 0, 0), "Butylka yaw=0"),
    ("bottle_yaw90", "Бутылка.stl", Orientation(0, 0, 90), "Butylka yaw=90"),
]

STRIP_H = 420  # высота одного кадра в полосе, px


def grounded_mesh(stl_path: Path, o: Orientation) -> Mesh:
    mesh = load_stl(stl_path)
    oriented = apply_orientation(mesh, o.roll_deg, o.pitch_deg, o.yaw_deg)
    zmin = oriented.vertices[:, 2].min()
    return Mesh(vertices=oriented.vertices - np.array([0.0, 0.0, zmin]), faces=oriented.faces)


def strip_image(mesh: Mesh) -> np.ndarray:
    tiles = []
    for angle in VIEW_ANGLES:
        d = DISTANCES.get(angle, CAMERA_DISTANCE)
        sil = camera_silhouette.build_silhouette(
            mesh, angle, RESOLUTION, FOV_DEG, d, "center", supersample=1
        )
        img = 255 - sil.mask.astype(np.uint8) * 255  # объект чёрным на белом
        h, w = img.shape
        scale = STRIP_H / h
        img = cv2.resize(img, (int(w * scale), STRIP_H), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        label = f"{angle:g} deg  D={d / 1000:g}m" + ("  (mirror)" if angle in DISTANCES else "")
        cv2.rectangle(img, (0, 0), (img.shape[1] - 1, 36), (240, 240, 240), -1)
        cv2.putText(img, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2)
        cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), (180, 180, 180), 1)
        tiles.append(img)
    return np.hstack(tiles)


def max_err_pct(dims, true_dims) -> float:
    d = sorted(dims, reverse=True)
    t = sorted(true_dims, reverse=True)
    return max(abs(100.0 * (a - b) / b) for a, b in zip(d, t))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    for case_id, stl_name, orient, note in CASES:
        stl_path = ASSETS / stl_name
        mesh = load_stl(stl_path)
        true_dims = mesh_true_dims(mesh)
        g = grounded_mesh(stl_path, orient)

        cv2.imwrite(str(OUT / f"sil_{case_id}.png"), strip_image(g))

        v1 = check_simple_camera_dims(
            mesh, orient, FOV_DEG, CAMERA_DISTANCE, RESOLUTION,
            camera_distances=DISTANCES, supersample=SUPERSAMPLE,
        )
        v3 = check_simple_camera_dims_v3(
            mesh, orient, FOV_DEG, CAMERA_DISTANCE, RESOLUTION,
            side_angles_deg=[0.0, 45.0, 135.0], belt_positions=BELT_POSITIONS,
            camera_distances=DISTANCES, supersample=SUPERSAMPLE,
        )
        entry = {
            "note": note,
            "true_dims": [round(x, 1) for x in true_dims] if true_dims else None,
            "grounded_extents": [
                round(float(x), 1)
                for x in (g.vertices.max(axis=0) - g.vertices.min(axis=0))
            ],
        }
        if v1 is not None:
            d1 = sorted([v1.length, v1.width, v1.height], reverse=True)
            entry["v1_dims"] = [round(x, 1) for x in d1]
            entry["v1_err_pct"] = round(max_err_pct(d1, true_dims), 1)
        if v3 is not None:
            d3 = sorted([v3.length, v3.width, v3.height], reverse=True)
            entry["v3_dims"] = [round(x, 1) for x in d3]
            entry["v3_err_pct"] = round(max_err_pct(d3, true_dims), 1)
        results[case_id] = entry
        print(case_id, json.dumps(entry, ensure_ascii=False))

    (OUT / "errors.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print("done ->", OUT)


if __name__ == "__main__":
    main()
