#!/usr/bin/env python3
"""Независимый веб-сервер математического CV-отчёта по загруженному STL.

Не использует Webots и SAM3. Сервер специально запускается отдельным процессом:
ошибка в чужом STL не влияет на демонстрацию исполнительного механизма.
"""
from __future__ import annotations

import argparse
import cgi
import json
import sys
import time
import traceback
import uuid
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = ROOT.parent / "roboson_tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from roboson_tools.core.experiment import Orientation, check_model_roll_g4, check_simple_camera_dims_v3
from roboson_tools.geometry.mesh_io import Mesh, load_stl
from roboson_tools.silhouette.analytical import PRINCIPAL_AXIS
from roboson_tools.silhouette.camera import belt_offset, build_silhouette, camera_frame, frame_shape
from roboson_tools.visual_hull.exact_polyhedral_hull import carve

STATIC_DIR = ROOT / "tools" / "cv_report_static"
DEFAULT_RUNS_DIR = Path("/tmp/robozon_cv_reports")
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
# Габаритный гейт (см. tools/prepare_stl.py::MAX_DIMS/MIN_DIMS/classify — тот же порог,
# постановка задачи требует 10×10×10мм минимум и 450×320×320мм максимум без исключений).
# ОТСУТСТВОВАЛ в этом отчёте до 2026-07-31 — category считалась только по G4 (round/not_round/
# uncertain), ни один размер не проверялся вовсе (найдено пользователем на "Ручка.stl": одна
# сторона 9мм < 10мм должна давать oversize, но category молча оставалась по одной круглости).
MAX_DIMS_MM = (450.0, 320.0, 320.0)
MIN_DIMS_MM = (10.0, 10.0, 10.0)


def _gauge_violation(dims_mm: list[float]) -> str | None:
    """Сортировка по убыванию и поосное сравнение (как в prepare_stl.py::classify) — не
    предполагает, что оси входа совпадают с осями габаритного ограничения. None — в допуске."""
    d = sorted(dims_mm, reverse=True)
    mx = sorted(MAX_DIMS_MM, reverse=True)
    mn = sorted(MIN_DIMS_MM, reverse=True)
    for i in range(3):
        if d[i] >= mx[i]:
            return f"сторона {d[i]:.1f}мм ≥ максимума {mx[i]:.0f}мм"
        if d[i] <= mn[i]:
            return f"сторона {d[i]:.1f}мм ≤ минимума {mn[i]:.0f}мм"
    return None


def _load_rig() -> tuple[list[dict], float, int]:
    preset_path = TOOLS_ROOT / "config" / "camera_rig_presets.yaml"
    app_path = TOOLS_ROOT / "config" / "app_settings.yaml"
    presets = yaml.safe_load(preset_path.read_text(encoding="utf-8"))["presets"]
    preset = next(p for p in presets if p["name"] == "Симулятор Webots (cv_rig, 5 ракурсов)")
    settings = yaml.safe_load(app_path.read_text(encoding="utf-8"))
    # G4 работает с квадратным растром. Геометрия рига (FOV, углы и
    # дистанции) при этом сохраняется, а 1024 px — штатное разрешение ядра.
    resolution = int(settings.get("raster_resolution_px", 1024))
    return preset["cameras"], float(settings.get("camera_fov_deg", 68)), resolution


def _write_png(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"Не удалось закодировать изображение {path.name}")
    path.write_bytes(encoded.tobytes())


def _plane_quad(center: np.ndarray, normal: np.ndarray, half_size: float) -> np.ndarray:
    """4 угла квадрата половинного размера `half_size`, лежащего в плоскости с данной
    нормалью через `center` — та же плоскость среза, что показывает круглую проекцию
    (roboson_tools рисует её как `set_slice_plane(center, normal=ось_переката)`)."""
    normal = normal / (np.linalg.norm(normal) or 1.0)
    ref = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, ref); u = u / np.linalg.norm(u)
    v = np.cross(normal, u)
    return np.array([
        center - half_size * u - half_size * v,
        center - half_size * u + half_size * v,
        center + half_size * u + half_size * v,
        center + half_size * u - half_size * v,
    ])


def _render_mesh(
    vertices: np.ndarray, faces: np.ndarray | None, out: Path, title: str,
    axis_dir: list[float] | None = None,
) -> None:
    """`axis_dir` — единичный вектор найденной оси переката (см. G4): рисуется ЛИНИЕЙ через
    центр реконструкции + полупрозрачной ПЛОСКОСТЬЮ среза перпендикулярно ей — то же
    сочетание (`set_roll_axis` + `set_slice_plane(center, normal=ось)`), что показывает
    Panel 1 в roboson_tools при включённом показе круглой проекции."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(projection="3d")
    if faces is not None:
        collection = Poly3DCollection(vertices[faces], alpha=0.55, edgecolor="#17324d", linewidths=0.15)
        collection.set_facecolor("#4da3d9")
        ax.add_collection3d(collection)
    else:
        ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], s=1, alpha=0.45, color="#1976d2")
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    extent = np.maximum(hi - lo, 1e-6)
    pad = extent * 0.04
    center = (lo + hi) / 2.0
    if axis_dir is not None:
        d = np.asarray(axis_dir, dtype=np.float64)
        d = d / (np.linalg.norm(d) or 1.0)
        half_len = float(np.linalg.norm(extent)) * 0.6
        p0, p1 = center - d * half_len, center + d * half_len
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                color="#d94f2b", linewidth=2.2, label="Ось переката")
        quad = _plane_quad(center, d, float(np.linalg.norm(extent)) * 0.55)
        plane = Poly3DCollection([quad], alpha=0.22, linewidths=0.8)
        plane.set_facecolor("#f2b134"); plane.set_edgecolor("#c98a12")
        ax.add_collection3d(plane)
        ax.legend(loc="upper left", fontsize=8, framealpha=0.6)
    ax.set(xlim=(lo[0]-pad[0], hi[0]+pad[0]), ylim=(lo[1]-pad[1], hi[1]+pad[1]), zlim=(lo[2]-pad[2], hi[2]+pad[2]))
    ax.set_box_aspect(extent)
    ax.set_title(title)
    ax.set_xlabel("X, мм"); ax.set_ylabel("Y, мм"); ax.set_zlabel("Z, мм")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def _render_projection(coords: list[tuple[float, float]] | None, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 5))
    if coords:
        pts = np.asarray(coords); ax.fill(pts[:, 0], pts[:, 1], "#4da3d9", alpha=.55); ax.plot(pts[:, 0], pts[:, 1], "#17324d")
    ax.set_aspect("equal"); ax.set_title("Круглая проекция по найденной оси"); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


_CELL_PX = 256
_CROP_MARGIN_FACTOR = 0.15  # тот же запас, что sim/cv_grid.py::bbox_center_and_size
_PAD_GRAY = 128

_LIGHT_DIR = np.array([0.35, -0.45, 0.82])
_LIGHT_DIR = _LIGHT_DIR / np.linalg.norm(_LIGHT_DIR)
_BASE_COLOR = np.array([70.0, 150.0, 210.0])  # BGR, тон близкий к #4da3d9 из _render_mesh
_AMBIENT = 0.35


def _mask_bbox_natural(mask: np.ndarray, margin_factor: float = _CROP_MARGIN_FACTOR) -> tuple[float, float, float]:
    """(cx, cy, natural_size) bbox маски с запасом — формула 1:1 с
    `sim/cv_grid.py::bbox_center_and_size`, чтобы окно кропа синтетического отчёта считалось
    так же, как окно кропа реальных кадров симулятора (Фаза 3.2)."""
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        h, w = mask.shape
        return w / 2.0, h / 2.0, float(max(w, h))
    y0, y1 = int(rows.min()), int(rows.max())
    x0, x1 = int(cols.min()), int(cols.max())
    w, h = (x1 - x0 + 1), (y1 - y0 + 1)
    cx, cy = x0 + w / 2.0, y0 + h / 2.0
    natural_size = max(w, h) * (1.0 + 2.0 * margin_factor)
    return float(cx), float(cy), float(natural_size)


def _crop_square_padded(frame: np.ndarray, cx: float, cy: float, side: float, pad_value) -> np.ndarray:
    """Квадратный кроп вокруг (cx, cy) с серым паддингом за краем кадра — 1:1 с
    `sim/cv_grid.py::crop_square_padded` (не импортируется напрямую, чтобы отчёт оставался
    независимым от sim/*, который требует конфигурацию рига симулятора)."""
    side_i = max(1, round(side))
    h, w = frame.shape[:2]
    x0 = round(cx - side_i / 2.0)
    y0 = round(cy - side_i / 2.0)
    shape = (side_i, side_i, frame.shape[2]) if frame.ndim == 3 else (side_i, side_i)
    canvas = np.full(shape, pad_value, dtype=frame.dtype)
    src_x0, src_y0 = max(0, x0), max(0, y0)
    src_x1, src_y1 = min(w, x0 + side_i), min(h, y0 + side_i)
    if src_x1 > src_x0 and src_y1 > src_y0:
        dst_x0, dst_y0 = src_x0 - x0, src_y0 - y0
        canvas[dst_y0:dst_y0 + (src_y1 - src_y0), dst_x0:dst_x0 + (src_x1 - src_x0)] = frame[src_y0:src_y1, src_x0:src_x1]
    return canvas


def _crop_and_resize_square(frame: np.ndarray, cx: float, cy: float, native_side: float,
                             cell_px: int, pad_value) -> np.ndarray:
    square = _crop_square_padded(frame, cx, cy, native_side, pad_value)
    scale = cell_px / native_side
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(square, (cell_px, cell_px), interpolation=interp)


def _render_camera_crop(mesh: Mesh, view_angle_deg: float, fov_deg: float, camera_distance: float,
                         belt_position, resolution_px: int) -> np.ndarray:
    """Плоский теневой (Lambert + ambient) рендер объекта из позы камеры рига — похоже на
    кроп реальной камеры, а не на бинарный силуэт. Проекция — ТА ЖЕ математика, что
    `silhouette.camera.build_silhouette` (`camera_frame`), поэтому масштаб и кадрирование
    идентичны маске, на которой считаются контуры/габариты."""
    camera_pos, forward, r_theta, f_px, cx, cy = camera_frame(view_angle_deg, fov_deg, camera_distance, resolution_px)
    n_rows, n_cols = frame_shape(resolution_px)
    dx = belt_offset(mesh, fov_deg, camera_distance, belt_position)
    verts = mesh.vertices + dx * PRINCIPAL_AXIS

    tri_verts = verts[mesh.faces]
    rel = tri_verts - camera_pos
    depth = rel @ forward
    valid = depth.min(axis=1) > 1e-6
    tri_verts, rel, depth = tri_verts[valid], rel[valid], depth[valid]

    canvas = np.full((n_rows, n_cols, 3), _PAD_GRAY, dtype=np.uint8)
    if len(tri_verts) == 0:
        return canvas

    u = rel @ r_theta
    x = rel @ PRINCIPAL_AXIS
    px_col = f_px * u / depth + cx
    px_row = cy - f_px * x / depth
    tri_px = np.round(np.stack([px_col, px_row], axis=-1)).astype(np.int32)

    v0, v1, v2 = tri_verts[:, 0], tri_verts[:, 1], tri_verts[:, 2]
    normals = np.cross(v1 - v0, v2 - v0)
    norm_len = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.where(norm_len > 1e-12, norm_len, 1.0)
    to_cam = camera_pos - tri_verts.mean(axis=1)
    flip = np.sum(normals * to_cam, axis=1) < 0.0
    normals[flip] *= -1.0
    shade = _AMBIENT + (1.0 - _AMBIENT) * np.clip(normals @ _LIGHT_DIR, 0.0, 1.0)

    order = np.argsort(-depth.mean(axis=1))  # дальние сначала — красим ближние поверх (painter's algorithm)
    for idx in order:
        color = tuple(int(c) for c in np.clip(_BASE_COLOR * shade[idx], 0, 255))
        cv2.fillConvexPoly(canvas, tri_px[idx], color)
    return canvas


def build_report(stl_path: Path, run_dir: Path, unit: str = "mm") -> dict:
    """Строит полный отчёт по синтетическим геометрическим силуэтам."""
    cameras, fov_deg, resolution_px = _load_rig()
    mesh = load_stl(stl_path)
    # STL не хранит единицы. Геометрия CV-рига задана в миллиметрах, поэтому
    # приводим загруженную модель к ним явно, не делая рискованных догадок.
    if unit == "m":
        mesh = Mesh(vertices=mesh.vertices * 1000.0, faces=mesh.faces)
    elif unit != "mm":
        raise ValueError("Поддерживаются только единицы мм или м")
    if len(mesh.vertices) < 4 or len(mesh.faces) < 4 or not np.isfinite(mesh.vertices).all():
        raise ValueError("STL не содержит корректного замкнутого треугольного меша")
    angles = [float(item["angle_deg"]) for item in cameras]
    distances = {float(item["angle_deg"]): float(item["distance_mm"]) for item in cameras}

    # Ровно 9 кадров: пять центральных камер + start/end у top и side.
    by_angle = {float(c["angle_deg"]): c for c in cameras}
    grid_spec = [("start top", 90., "start"), ("центр top", 90., "center"), ("end top", 90., "end"),
                 ("start side", 5., "start"), ("центр diag", 70., "center"), ("end side", 5., "end"),
                 ("центр top mirror", 157.26, "center"), ("центр side", 5., "center"), ("центр diag mirror", 171.95, "center")]
    # Шаг 1: bbox каждой ячейки по бинарному силуэту полного кадра (дёшево — нужен только
    # bbox), затем ЕДИНОЕ native-окно кропа на всю сетку — max(natural_size) по всем 9
    # ячейкам, как в `sim/cv_grid_v2.py::assemble_grid_v2` (Шаг 3): маленький в кадре объект
    # не раздувается до целого кадра, а крупный не обрезается — сохраняется РЕАЛЬНОЕ
    # относительное соотношение размеров между ракурсами, а не одинаковый zoom на каждый.
    grid_start = time.perf_counter()
    raw_masks: list[np.ndarray] = []
    centers: list[tuple[float, float]] = []
    natural_sizes: list[float] = []
    for label, angle, position in grid_spec:
        sil = build_silhouette(mesh, angle, resolution_px, fov_deg, float(by_angle[angle]["distance_mm"]), position, crop_to_object=False)
        cx, cy, natural = _mask_bbox_natural(sil.mask)
        raw_masks.append(sil.mask); centers.append((cx, cy)); natural_sizes.append(natural)
    native_side = max(natural_sizes)

    thumbs = []
    for (label, angle, position), (cx, cy) in zip(grid_spec, centers):
        render = _render_camera_crop(mesh, angle, fov_deg, float(by_angle[angle]["distance_mm"]), position, resolution_px)
        thumbs.append(_crop_and_resize_square(render, cx, cy, native_side, _CELL_PX, _PAD_GRAY))
    _write_png(run_dir / "grid.png", np.vstack([np.hstack(thumbs[i:i+3]) for i in range(0, 9, 3)]))
    # Время шага 1 — реальная стоимость формирования этих 9 кропов + сборки сетки (включая
    # 3D-рендеринг силуэтов/теней), а не только финальной компоновки изображения.
    grid_s = time.perf_counter() - grid_start

    # Шаг 2: контуры — на ТЕХ ЖЕ кропах (тот же центр/native_side/масштаб, что и сетка выше),
    # а не на кадре целиком — чтобы контур был нормализован по масштабу так же, как в сетке.
    seg_start = time.perf_counter()
    contours_imgs = []
    for mask, (cx, cy) in zip(raw_masks, centers):
        mask_u8 = mask.astype(np.uint8) * 255
        crop = _crop_and_resize_square(mask_u8, cx, cy, native_side, _CELL_PX, 0)
        fg = cv2.threshold(crop, 127, 255, cv2.THRESH_BINARY)[1]
        canvas = cv2.cvtColor(fg, cv2.COLOR_GRAY2BGR)
        cs, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, cs, -1, (40, 210, 80), 2)
        contours_imgs.append(canvas)
    _write_png(run_dir / "contours.png", np.vstack([np.hstack(contours_imgs[i:i+3]) for i in range(0, 9, 3)]))
    segmentation_s = time.perf_counter() - seg_start

    reconstruction_start = time.perf_counter()
    # Exact Polyhedral Visual Hull — сохраняет вогнутости (см. модуль), НЕ выпуклый как
    # прежний polytope_hull.carve (пересечение полупространств математически всегда выпукло).
    # Только для визуализации — габариты по-прежнему считает check_simple_camera_dims_v3 ниже.
    hull = carve(mesh, [(angle, "center") for angle in angles], fov_deg, distances, resolution_px,
                 contour_epsilon_px=1.5, crop_to_object=False)
    reconstruction_s = time.perf_counter() - reconstruction_start
    if hull is None:
        raise RuntimeError("Не удалось построить visual hull по пяти геометрическим силуэтам")
    dims_v3 = check_simple_camera_dims_v3(mesh, Orientation(), fov_deg, distances[90.], resolution_px, top_angle_deg=90., side_angles_deg=[5., 70., 157.26, 171.95], belt_positions=["start", "center", "end"], camera_distances=distances, supersample=1)
    dims = (hull.vertices.max(axis=0) - hull.vertices.min(axis=0)).tolist()

    g4_start = time.perf_counter()
    g4 = check_model_roll_g4(
        mesh, Orientation(), angles, resolution_px=resolution_px,
        use_camera_silhouettes=True, camera_fov_deg=fov_deg, camera_distances=distances,
        torchhull_parallax=True,
    )
    g4_s = time.perf_counter() - g4_start
    best_axis = list(g4.best_axis_dir) if g4.best_axis_dir is not None else None

    # Устойчивая локальная круглость вдоль оси переката (см. вики [[Устойчивая локальная
    # круглость вдоль оси переката (G4)]]) — заменяет отклонённый подход через
    # `trimesh.compute_stable_poses` (не разделял короб и круглые объекты, см. история задачи).
    # `torchhull_parallax=True` выше делает признак РЕАЛЬНЫМ (не только диагностикой):
    # на GPU (torchhull, `dense_points=True`) `sustained_demoted` уже понизил `g4.verdict`
    # round→uncertain внутри `check_model_roll_g4`, если локальная круглость не держится на
    # достаточной доле длины оси (короб/куб по диагонали) — сюда просто прокидываем цифры для
    # отображения, отдельного пересчёта category не нужно (она уже определяется `g4.verdict`
    # ниже, который эту демоцию уже учёл).
    dims_mm = [round(v, 2) for v in ((dims_v3.length, dims_v3.width, dims_v3.height) if dims_v3 else dims)]
    # `tri_height` (roboson_tools/core/experiment.py::check_simple_camera_dims_v3) теперь
    # обобщён на ПРОИЗВОЛЬНУЮ пару азимутов (см. [[Триангуляция высоты на риге без чисто
    # бокового ракурса (roboson_tools)]]) — на этом 5-камерном риге (все боковые ракурсы
    # наклонные, чисто бокового v_z=0 нет) height для тонких объектов больше не уходит в 0.0.
    # `triangulated_points` (per-point триангуляция) при этом всё ещё может остаться 0 для
    # тонкого сечения — независимая, не исправленная этим фиксом проблема (см. ТЗ, «Не входит
    # в скоуп») — но она больше не означает недостоверную ЦИФРУ height, только то, что
    # per-point путь не подтвердил её независимо. Флаг переведён на прямую проверку самой
    # цифры: недостоверно, только если ВСЕ источники высоты (triangulated_height,
    # fallback_height, tri_height) дали 0.0.
    dims_unreliable = dims_v3 is not None and dims_v3.height == 0.0
    # Габаритный гейт — ПЕРВЫМ (см. tools/prepare_stl.py::classify, постановка задачи требует
    # именно такой порядок: сначала размер, затем круглость). Раньше отсутствовал в этом
    # отчёте — category считалась только по G4, ни один размер не проверялся.
    gauge = _gauge_violation(dims_mm)
    if gauge is not None:
        category = "C"
    else:
        category = {"round": "D", "not_round": "B", "uncertain": "C"}.get(g4.verdict, "C")
    # Ось+плоскость среза рисуются ПОСЛЕ G4 (нужен best_axis) — тот же вид, что Panel 1 в
    # roboson_tools при включённом показе круглой проекции (see _render_mesh docstring).
    _render_mesh(hull.vertices, hull.faces, run_dir / "reconstruction.png",
                 "3D-реконструкция Visual Hull (невыпуклый, Exact Polyhedral Hull)", axis_dir=best_axis)
    _render_projection(g4.best_projection_coords, run_dir / "round_projection.png")
    total_s = grid_s + segmentation_s + reconstruction_s + g4_s
    sustained_note = (
        "Локальная круглость не держится вдоль оси (короб/куб по диагонали) — verdict понижен "
        "round → uncertain этим признаком."
        if g4.sustained_demoted else
        "Ось переката найдена среди альтернативных базовых кандидатов после того, как основная "
        "не прошла проверку устойчивости — verdict остался round."
        if g4.sustained_promoted_from_alt_axis else
        "Признак ЭКСПЕРИМЕНТАЛЬНО указывает на not_round (низкая устойчивая круглость и у "
        "лучшей оси, и у остальных базовых кандидатов), но НЕ влияет на category: регрессия "
        "нашла реальный катящийся объект (cyl_skewed) с такими же числами на этом риге — "
        "калибровка порога не разделяет случаи, см. заметку задачи. category остаётся uncertain "
        "→ оператору, не понижается автоматически."
        if g4.sustained_confirmed_not_round else
        "Проверка не запускалась (verdict не был round на момент проверки)."
        if g4.sustained_fraction is None else
        "Локальная круглость держится на достаточной доле длины оси — verdict не понижен."
    )
    return {
        "source": stl_path.name,
        "input_unit": unit,
        "notice": "Сегментация в этом отчёте — точный геометрический силуэт STL, не результат сенсорной CV-сегментации и не SAM3.",
        "grid": {"image": "grid.png", "seconds": round(grid_s, 4), "frames": 9},
        "segmentation": {"image": "contours.png", "seconds": round(segmentation_s, 4)},
        "reconstruction": {"algorithm": "multi-side triangulation v3 (габариты); Exact Polyhedral Visual Hull, невыпуклый (визуализация)", "seconds": round(reconstruction_s, 4), "image": "reconstruction.png", "dimensions_mm": dims_mm,
                           "dims_unreliable": dims_unreliable,
                           "dims_note": ("Высоту не удалось определить (все методы триангуляции дали 0) — "
                                         "возможно, объект вне поля зрения бокового рига или вырожденный силуэт."
                                         if dims_unreliable else None)},
        "gauge": {
            "violation": gauge,
            "min_mm": list(MIN_DIMS_MM),
            "max_mm": list(MAX_DIMS_MM),
            "note": (f"Вне габарита: {gauge} — категория C независимо от круглости."
                     if gauge else "В допуске."),
        },
        "g4": {"verdict": g4.verdict, "category": category, "k": round(g4.best.k, 4) if g4.best else None,
               "axis": [round(v, 5) for v in best_axis] if best_axis else None, "seconds": round(g4_s, 4), "projection": "round_projection.png",
               "note": "Категория чувствительна к ориентации STL (Camera Mode — реальный разреженный риг из 5 фиксированных ракурсов, а не идеализированная ортографика): поворот объекта относительно рига может дать другую k. Здесь используется ориентация STL как есть (без поворота)."},
        "sustained": {
            "fraction": round(g4.sustained_fraction, 4) if g4.sustained_fraction is not None else None,
            "demoted": g4.sustained_demoted,
            "promoted_from_alt_axis": g4.sustained_promoted_from_alt_axis,
            "confirmed_not_round": g4.sustained_confirmed_not_round,  # диагностика, на category НЕ влияет — см. note
            "seconds": round(g4.sustained_seconds, 4) if g4.sustained_seconds is not None else None,
            "note": sustained_note,
        },
        "total_seconds": round(total_s, 4),
    }


class Handler(SimpleHTTPRequestHandler):
    runs_dir: Path

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def _json(self, status: int, body: dict) -> None:
        encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status); self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded))); self.end_headers(); self.wfile.write(encoded)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path.startswith("/reports/"):
            relative = Path(path.removeprefix("/reports/")).as_posix()
            requested = (self.runs_dir / relative).resolve()
            if self.runs_dir.resolve() not in requested.parents or not requested.is_file():
                self.send_error(HTTPStatus.NOT_FOUND); return
            self.send_response(HTTPStatus.OK); self.send_header("Content-Type", "image/png"); self.end_headers(); self.wfile.write(requested.read_bytes()); return
        if path == "/api/health":
            self._json(200, {"ok": True}); return
        if path == "/":
            self.path = "/index.html"
        super().do_GET()

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/report":
            self.send_error(HTTPStatus.NOT_FOUND); return
        if int(self.headers.get("Content-Length", "0")) > MAX_UPLOAD_BYTES:
            self._json(413, {"error": "STL больше 100 МБ"}); return
        try:
            form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": self.headers.get("Content-Type", "")})
            upload = form["stl"] if "stl" in form else None
            if upload is None or not getattr(upload, "file", None):
                raise ValueError("Поле stl не найдено")
            name = Path(upload.filename or "model.stl").name
            if Path(name).suffix.lower() != ".stl":
                raise ValueError("Нужен файл с расширением .stl")
            run_id = uuid.uuid4().hex
            run_dir = self.runs_dir / run_id
            run_dir.mkdir(parents=True, exist_ok=False)
            source = run_dir / name
            payload = upload.file.read(MAX_UPLOAD_BYTES + 1)
            if len(payload) > MAX_UPLOAD_BYTES:
                raise ValueError("STL больше 100 МБ")
            source.write_bytes(payload)
            result = build_report(source, run_dir, form.getfirst("unit", "mm"))
            result["id"] = run_id
            self._json(200, result)
        except Exception as exc:
            traceback.print_exc()
            self._json(422, {"error": str(exc)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    args = parser.parse_args()
    args.runs_dir.mkdir(parents=True, exist_ok=True)
    Handler.runs_dir = args.runs_dir.resolve()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"CV-отчёт: http://{args.host}:{args.port}/ (артефакты: {args.runs_dir})")
    server.serve_forever()


if __name__ == "__main__":
    main()
