"""Точное пересечение конусов обзора (полупространств) — альтернатива voxel carving
(`visual_hull/voxel_carving.py`) для ВЫПУКЛЫХ объектов. См. заметку задачи "Добавить Camera
Mode..." — обсуждение обоих подходов.

Идея: конус обзора одной камеры — это пирамида с апексом в камере и основанием, заданным
ВЫПУКЛОЙ ОБОЛОЧКОЙ силуэта (тот же приём, что уже используется для круглой проекции в
`metrics/roundness.py` — выпуклая оболочка задаёт контакт при перекате, конкретный невыпуклый
контур не важен). Каждая грань такой пирамиды — это полупространство, ограниченное плоскостью
через апекс и ребро контура. Форма объекта, видимая одновременно из N камер, лежит в
пересечении ВСЕХ этих полупространств по всем камерам — то есть в пересечении полупространств
единой линейной системы. Это ТОЧНАЯ (без сеточной аппроксимации) выпуклая оболочка формы,
вычисляемая через `scipy.spatial.HalfspaceIntersection` (обёртка над qhull — той же
библиотекой, что неявно используется через `cv2.convexHull`/shapely).

Ограничение: метод восстанавливает именно ВЫПУКЛУЮ форму (пересечение полупространств always
convex) — как и voxel carving, это Visual Hull (верхняя оценка, содержит объект), но здесь
без дискретизации на сетке; для объекта с вогнутостями (не встречается в тестовом наборе
коробок/бутылок) даст выпуклую оболочку истинной формы, не точный вогнутый контур — то же
допущение, что уже принято в `check_model_roll` для круглой проекции.

ИСКЛЮЧИТЕЛЬНО для визуализации на Panel 1 (сравнение с voxel carving), см.
`gui/panels/model_panel.set_camera_hull_polytope`. НЕ подключено к габаритам/круглой проекции.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, HalfspaceIntersection, QhullError

from ..geometry.mesh_io import Mesh
from ..silhouette.analytical import PRINCIPAL_AXIS
from ..silhouette.camera import BeltPosition, belt_offset, build_silhouette, camera_frame
from ._cone_geometry import halfspaces_from_mask

_INTERIOR_MARGIN = 1e-6  # запас, чтобы точка считалась строго внутри (не на границе)


@dataclass
class PolytopeHull:
    vertices: np.ndarray  # (V, 3) вершины восстановленного выпуклого многогранника
    faces: np.ndarray  # (F, 3) индексы треугольной триангуляции граней (из qhull)


def _view_halfspaces(
    mesh: Mesh,
    view_angle_deg: float,
    belt_position: BeltPosition,
    fov_deg: float,
    camera_distance: float,
    resolution_px: int,
    contour_epsilon_px: float = 0.0,
    crop_to_object: bool = False,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Полупространства (нормали + смещения) одного конуса обзора, в НЕсдвинутой (локальной)
    системе координат объекта. Возвращает (normals (K,3) единичные, offsets (K,)) такие, что
    точка p внутри конуса <=> normals @ p + offsets <= 0 для всех K. None — вырожденный силуэт
    (пустая маска/контур < 3 точек).

    `crop_to_object=True` — передаётся в `build_silhouette` для ROI (обрезки кадра до bbox
    объекта + margin): маска плотнее покрывает объект, оптический центр берётся из
    `Silhouette.cx`/`Silhouette.cy` (а не из `camera_frame`, которая считает полный кадр).
    См. докстринг `silhouette/camera.build_silhouette`."""
    dx = belt_offset(mesh, fov_deg, camera_distance, belt_position)
    silhouette = build_silhouette(
        mesh,
        view_angle_deg,
        resolution_px,
        fov_deg=fov_deg,
        camera_distance=camera_distance,
        belt_position=belt_position,
        crop_to_object=crop_to_object,
    )
    camera_pos, forward, r_theta, f_px, _cx, _cy = camera_frame(
        view_angle_deg, fov_deg, camera_distance, resolution_px
    )
    # Оптический центр: при ROI (cx/cy не None) — из силуэта (сдвинут в координаты обрезанной
    # маски), иначе — из camera_frame (полный кадр, cx/cy = cols/2/rows/2).
    cx = silhouette.cx if silhouette.cx is not None else _cx
    cy = silhouette.cy if silhouette.cy is not None else _cy
    # Апекс в системе координат НЕсдвинутого объекта: сдвиг объекта на +dx эквивалентен
    # наблюдению несдвинутого объекта камерой, сдвинутой на -dx (та же логика, что в
    # voxel_carving.carve — там сдвигается сетка, здесь эквивалентно сдвигаем апекс).
    apex = camera_pos - dx * PRINCIPAL_AXIS

    # Кольцевая модель — частный случай общей геометрии конуса (`_cone_geometry.py`, там же
    # используется произвольной позой камеры для реальных фото): "up" = PRINCIPAL_AXIS общий на
    # все камеры => "down" = -PRINCIPAL_AXIS, квадратный пиксель => fx = fy = f_px.
    return halfspaces_from_mask(
        silhouette.mask, apex, forward, r_theta, -PRINCIPAL_AXIS, f_px, f_px, cx, cy,
        contour_epsilon_px=contour_epsilon_px,
    )


def _find_interior_point(A: np.ndarray, b: np.ndarray, centroid: np.ndarray) -> np.ndarray | None:
    """Точка, строго внутри всех полупространств A@p + b <= 0 — нужна scipy для
    HalfspaceIntersection. Сначала пробуем центроид объекта (дёшево, обычно работает — объект
    по построению виден во всех конусах с запасом). Если нет — решаем LP на центр Чебышева
    (максимальный вписанный шар) — надёжный запасной путь."""
    if np.all(A @ centroid + b < -_INTERIOR_MARGIN):
        return centroid

    n = A.shape[0]
    a_ub = np.hstack([A, np.ones((n, 1))])
    b_ub = -b
    c = np.array([0.0, 0.0, 0.0, -1.0])  # maximize radius r == minimize -r
    res = linprog(c, A_ub=a_ub, b_ub=b_ub, bounds=[(None, None)] * 3 + [(0, None)], method="highs")
    if not res.success or res.x[3] <= _INTERIOR_MARGIN:
        return None
    return res.x[:3]


def carve(
    mesh: Mesh,
    views: list[tuple[float, BeltPosition]],
    fov_deg: float,
    camera_distances: dict[float, float],
    resolution_px: int = 1024,
    contour_epsilon_px: float = 0.0,
    crop_to_object: bool = False,
) -> PolytopeHull | None:
    """Точное пересечение конусов обзора всех `views` (список пар азимут+позиция на ленте —
    тот же формат, что у `voxel_carving.carve`). `camera_distances` — обязательная карта
    {азимут -> дистанция}, по одной на каждый азимут в `views` (см. `core/camera_rig.py`);
    `KeyError` при отсутствии азимута — намеренно, без фоллбека (см. `voxel_carving.carve`).
    None — вырожденный случай (пустой силуэт хотя бы в одном кадре, либо пустое/неограниченное
    пересечение полупространств).

    `contour_epsilon_px` — допуск упрощения контура через `cv2.approxPolyDP` (после выпуклой
    оболочки); передаётся в `halfspaces_from_mask`. ДЕФОЛТ 0.0 (упрощение отключено) — обратная
    совместимость со всеми существующими вызывающими (`pose_carving.carve_polytope` через
    `_halfspaces_from_observation` ожидает ТОТ же результат, что ring-модель; тест
    `test_ring_equivalent_pose_matches_ring_model_polytope` проверяет побитовое совпадение).
    Для G4 рекомендуется `contour_epsilon_px=1.5` — выпуклый контур прямоугольника даёт 4 угла
    вместо ~100 точек, ускоряя qhull в 5-20×, ценой субпиксельной симметрии контура
    (см. заметку bench_g4_endtoend.py / test_polytope_hull.py::
    test_mirrored_azimuths_give_mirrored_halfspaces — допуск там поднят с 1e-6 до 1e-2). Это
    осознанный выбор на уровне конкретного метода/вызывающего, а не глобальное изменение.

    `crop_to_object=True` — обрезать кадр до bbox маски + margin в каждом силуэте (см.
    `silhouette/camera.build_silhouette`): маска плотнее покрывает объект, ускоряя построение
    полупространств и решая проблему вырождения тонких объектов (где контур < 3 пикселей на
    полном кадре). Оптический центр при этом берётся из `Silhouette.cx`/`Silhouette.cy`.
    По умолчанию False — обратная совместимость со всеми существующими вызывающими."""
    all_normals, all_offsets = [], []
    for view_angle_deg, belt_position in views:
        result = _view_halfspaces(
            mesh, view_angle_deg, belt_position, fov_deg, camera_distances[view_angle_deg],
            resolution_px, contour_epsilon_px=contour_epsilon_px,
            crop_to_object=crop_to_object,
        )
        if result is None:
            return None
        normals, offsets = result
        all_normals.append(normals)
        all_offsets.append(offsets)

    A = np.concatenate(all_normals, axis=0)
    b = np.concatenate(all_offsets, axis=0)
    halfspaces = np.hstack([A, b[:, None]])

    interior_point = _find_interior_point(A, b, mesh.vertices.mean(axis=0))
    if interior_point is None:
        return None

    try:
        hs = HalfspaceIntersection(halfspaces, interior_point)
        vertices = hs.intersections
        if len(vertices) < 4:
            return None
        hull = ConvexHull(vertices)
    except QhullError:
        return None

    return PolytopeHull(vertices=vertices, faces=hull.simplices)
