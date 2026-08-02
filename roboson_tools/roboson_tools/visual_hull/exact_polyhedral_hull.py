"""Exact Polyhedral Visual Hull — точное пересечение ОБОБЩЁННЫХ (не обязательно выпуклых)
конусов обзора, альтернатива `polytope_hull.py` для НЕВЫПУКЛЫХ объектов.

Диагностированная причина завести этот модуль: `polytope_hull.py` перед построением
полупространств берёт `cv2.convexHull` контура силуэта — любая вогнутость (например, резкий
переход "толстая зажимаемая часть -> тонкий стержень без зажима" у ручки) стирается ещё на
этом шаге. Пересечение полупространств математически ВСЕГДА выпукло, поэтому такой уступ
восстанавливается как плавный конус, а не цилиндр с уступом (подтверждено синтетически, см.
`tests/test_exact_polyhedral_hull.py`) — то же исходное допущение "объект выпуклый", что
изначально было принято для круглой проекции (`metrics/roundness.py`) и повторено в
`polytope_hull.py`.

Идея (Franco & Boyer, "Exact Polyhedral Visual Hull", 2003): конус обзора одной камеры — это
замкнутое тело с апексом в камере и ОСНОВАНИЕМ — реальным (не выпрямленным) контуром силуэта,
триангулированное ear-clipping'ом (`trimesh.creation.triangulate_polygon`, движок
`mapbox_earcut`) на "дальней крышке" плюс боковой веер треугольников апекс-ребро. Пересечение
(булево, `trimesh.boolean.intersection`, движок `manifold3d`) таких тел по всем ракурсам даёт
ТОЧНЫЙ (без сеточной аппроксимации) visual hull, включая вогнутости — то, что `voxel_carving.py`
даёт лишь приближённо (и то только при достаточно мелкой, дорогой сетке — см. заметку задачи).

Требует `manifold3d`+`mapbox_earcut` (см. requirements.txt) — оба ставятся как готовые wheel'ы,
компилятор не нужен.

ИСКЛЮЧИТЕЛЬНО для визуализации/сравнения на Panel 1, как и `polytope_hull.py`/`voxel_carving.py`.
НЕ подключено к габаритам/круглой проекции.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from ..geometry.mesh_io import Mesh
from ..silhouette.analytical import PRINCIPAL_AXIS
from ..silhouette.camera import BeltPosition, belt_offset, build_silhouette, camera_frame
from ._cone_geometry import cone_mesh_from_mask


@dataclass
class ExactHull:
    vertices: np.ndarray  # (V, 3) вершины восстановленного (возможно невыпуклого) многогранника
    faces: np.ndarray  # (F, 3) индексы треугольной триангуляции граней


def _view_cone_mesh(
    mesh: Mesh,
    view_angle_deg: float,
    belt_position: BeltPosition,
    fov_deg: float,
    camera_distance: float,
    resolution_px: int,
    far_distance: float,
    contour_epsilon_px: float,
    crop_to_object: bool = False,
) -> trimesh.Trimesh | None:
    """Обобщённый конус обзора одной камеры — замкнутый треугольный меш (апекс + невыпуклая
    дальняя крышка на глубине far_distance от апекса), в НЕсдвинутой (локальной) системе
    координат объекта — та же логика сдвига апекса на -dx, что в `polytope_hull._view_halfspaces`.
    None — вырожденный силуэт (пустая маска/контур < 3 точек) или ошибка триангуляции.

    `crop_to_object=True` — передаётся в `build_silhouette` для ROI (см.
    `polytope_hull._view_halfspaces` и `silhouette/camera.build_silhouette`)."""
    dx = belt_offset(mesh, fov_deg, camera_distance, belt_position)
    silhouette = build_silhouette(
        mesh, view_angle_deg, resolution_px,
        fov_deg=fov_deg, camera_distance=camera_distance, belt_position=belt_position,
        crop_to_object=crop_to_object,
    )
    camera_pos, forward, r_theta, f_px, _cx, _cy = camera_frame(
        view_angle_deg, fov_deg, camera_distance, resolution_px
    )
    cx = silhouette.cx if silhouette.cx is not None else _cx
    cy = silhouette.cy if silhouette.cy is not None else _cy
    apex = camera_pos - dx * PRINCIPAL_AXIS

    # Кольцевая модель — частный случай общей геометрии конуса (`_cone_geometry.py`, там же
    # используется произвольной позой камеры для реальных фото): "up" = PRINCIPAL_AXIS общий на
    # все камеры => "down" = -PRINCIPAL_AXIS, квадратный пиксель => fx = fy = f_px.
    return cone_mesh_from_mask(
        silhouette.mask, apex, forward, r_theta, -PRINCIPAL_AXIS, f_px, f_px, cx, cy,
        far_distance, contour_epsilon_px,
    )


def carve(
    mesh: Mesh,
    views: list[tuple[float, BeltPosition]],
    fov_deg: float,
    camera_distances: dict[float, float],
    resolution_px: int = 1024,
    far_distance: float | None = None,
    contour_epsilon_px: float = 1.0,
    crop_to_object: bool = False,
) -> ExactHull | None:
    """Точное пересечение обобщённых конусов обзора всех `views` — тот же формат (азимут +
    позиция на ленте), что у `polytope_hull.carve`/`voxel_carving.carve`. `camera_distances` —
    обязательная карта {азимут -> дистанция}, по одной на каждый азимут в `views` (см.
    `core/camera_rig.py`); `KeyError` при отсутствии азимута — намеренно, без фоллбека (см.
    `voxel_carving.carve`). В отличие от `polytope_hull`, сохраняет вогнутости силуэта (не
    берёт `cv2.convexHull` контура). None — вырожденный случай (пустой силуэт, вырожденная
    триангуляция, пустое/неограниченное пересечение).

    `contour_epsilon_px` — допуск упрощения контура через `cv2.approxPolyDP` ДО
    триангуляции; передаётся в `cone_mesh_from_mask`. ДЕФОЛТ 1.0 (исходное поведение до Плана A)
    — обратная совместимость со всеми существующими вызывающими. Для G4 рекомендуется
    `contour_epsilon_px=1.5` (согласовано с `polytope_hull.carve`) — упрощает earcut-
    триангуляцию и manifold boolean. `0` отключает упрощение.

    `crop_to_object=True` — обрезать кадр до bbox маски + margin в каждом силуэте (см.
    `polytope_hull.carve` и `silhouette/camera.build_silhouette`)."""
    if len(views) < 2:
        # Один конус — не "неограниченное полупространство", как у polytope_hull (тело здесь
        # всегда конечное, обрезано по far_distance), но эта граница чисто числовая, не несёт
        # информации о реальном протяжении объекта вдоль луча взгляда — тот же вырожденный
        # случай по смыслу, что и "один азимут" у polytope_hull, возвращаем None по той же
        # причине, а не молча отдаём произвольно обрезанный конус.
        return None
    if far_distance is None:
        # Самая длинная дистанция рига — с запасом, чтобы дальняя крышка конуса гарантированно
        # была за объектом при ЛЮБОЙ из (теперь разных) дистанций камер, включая самую длинную
        # (зеркальный ракурс).
        far_distance = max(camera_distances[a] for a, _ in views) * 2.0

    cones = []
    for view_angle_deg, belt_position in views:
        cone = _view_cone_mesh(
            mesh, view_angle_deg, belt_position, fov_deg, camera_distances[view_angle_deg],
            resolution_px, far_distance, contour_epsilon_px,
            crop_to_object=crop_to_object,
        )
        if cone is None:
            return None
        cones.append(cone)

    if len(cones) == 1:
        result = cones[0]
    else:
        try:
            result = trimesh.boolean.intersection(cones, engine="manifold")
        except Exception:
            return None

    if result is None or len(result.vertices) == 0 or len(result.faces) == 0:
        return None
    return ExactHull(vertices=np.asarray(result.vertices), faces=np.asarray(result.faces))
