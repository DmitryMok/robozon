"""Общая геометрия конуса обзора ОДНОЙ камеры — полупространства выпуклой оболочки контура
(`halfspaces_from_mask`, использует `polytope_hull.py`) и обобщённый (возможно невыпуклый) конус
как замкнутый меш (`cone_mesh_from_mask`, использует `exact_polyhedral_hull.py`).

Параметризовано общим "camera frame" — апекс + forward/right/down (единичные, взаимно
ортогональные, мировые координаты) + fx/fy/cx/cy — вместо того, чтобы жёстко знать про кольцевую
модель камеры (`silhouette/camera.py::camera_frame`, azimuth+distance, `up` = глобальный
`PRINCIPAL_AXIS` на все камеры). Это и есть генерализация, нужная для 3D-реконструкции по
реальным фото с произвольной позой камеры на кадр (`pose.camera_pose.CameraPose`) — см. заметку
задачи "3D-реконструкция объекта по реальным фото — от маски+позы к visual hull". Кольцевая
модель — частный случай: `forward`/`right` из `silhouette.analytical.view_axes`, `down =
-PRINCIPAL_AXIS`, `fx = fy = f_px` (квадратный пиксель).

Соглашение осей пикселя: `dx = (col - cx) / fx` растёт вправо по кадру, `dy = (row - cy) / fy`
растёт ВНИЗ по кадру (стандартная связка с `down`, не `up`, поэтому кольцевая модель подставляет
`down = -PRINCIPAL_AXIS`, а не `PRINCIPAL_AXIS` — см. её докстринг `silhouette/camera.py`).
Направление луча апекс->точка контура на глубине 1: `forward + dx*right + dy*down`.
"""

from __future__ import annotations

import cv2
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from shapely.geometry import Polygon

_MIN_NORMAL_LEN = 1e-9
_MIN_CONTOUR_POINTS = 3
_MATCH_TOL = 1e-6  # допуск совпадения координат крышки/бокового веера (см. cone_mesh_from_mask)


def halfspaces_from_mask(
    mask: np.ndarray,
    apex: np.ndarray,
    forward: np.ndarray,
    right: np.ndarray,
    down: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    contour_epsilon_px: float = 0.0,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Полупространства (нормали + смещения) конуса обзора по ВЫПУКЛОЙ оболочке контура маски:
    точка p внутри конуса <=> `normals @ p + offsets <= 0` для всех строк. None — вырожденная
    маска (пустая/контур < 3 уникальных направлений после схлопывания коллинеарных рёбер).

    `contour_epsilon_px` — допуск упрощения контура через `cv2.approxPolyDP` ПОСЛЕ выпуклой
    оболочки: выпуклый контур аппроксимируется ломаной с допуском `epsilon` пикселей. На
    маске 512² типичный контур коробки (~100 пикселей по периметру) сводится к 4 углам,
    цилиндра — к 10-20 вершинам вместо ~50-100. Это пропорционально уменьшает число
    полупространств (граней конуса), ускоряя `HalfspaceIntersection` (qhull) в polytope_hull
    и повышая численную устойчивость (меньше почти-коллинеарных рёбер). `epsilon=0` (ДЕФОЛТ)
    отключает упрощение — старое поведение, сохраняет обратную совместимость со всеми
    существующими вызывающими (`pose_carving._halfspaces_from_observation` и др.), которые
    не знают про новый параметр. Для G4 (k по выпуклой оболочке проекции) упрощение не
    теряет информацию — выпуклая оболочка упрощённого контура та же, что и у исходного, с
    точностью до `epsilon` пикселей. `polytope_hull.carve` по умолчанию передаёт 1.5 — это
    осознанный выбор на уровне конкретного метода, а не глобальное изменение поведения."""
    mask_u8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    all_pts = np.concatenate(contours, axis=0).reshape(-1, 2).astype(np.float32)
    hull_pts = cv2.convexHull(all_pts)[:, 0, :]  # (K, 2) [col, row], упорядочены по контуру
    if len(hull_pts) < 3:
        return None
    if contour_epsilon_px > 0:
        hull_pts = cv2.approxPolyDP(hull_pts, contour_epsilon_px, closed=True)[:, 0, :]
        if len(hull_pts) < 3:
            return None

    cols, rows = hull_pts[:, 0].astype(np.float64), hull_pts[:, 1].astype(np.float64)
    dx = (cols - cx) / fx
    dy = (rows - cy) / fy
    directions = (
        forward[None, :] + dx[:, None] * right[None, :] + dy[:, None] * down[None, :]
    )

    centroid_dir = directions.mean(axis=0)
    normals = np.cross(directions, np.roll(directions, -1, axis=0))  # (K,3), по одной на ребро
    # Ориентация обхода контура заранее неизвестна — разворачиваем нормаль так, чтобы
    # "внутреннее" направление (в среднем к центру контура) было по одну сторону от всех
    # плоскостей (centroid_dir @ n <= 0).
    flip = (normals @ centroid_dir) > 0
    normals[flip] *= -1

    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > _MIN_NORMAL_LEN  # вырожденные рёбра (совпадающие/коллинеарные соседние
    # направления) пропускаются
    if not valid.any():
        return None
    normals = normals[valid] / lengths[valid, None]

    offsets = -(normals @ apex)
    return normals, offsets


def fix_winding(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray | None:
    """Разворачивает грани так, чтобы соседние треугольники обходили общее ребро в
    противоположных направлениях, затем — если получившийся объём отрицательный — разворачивает
    всё целиком наружу. None — грань встречается не ровно у двух треугольников (не замкнутая
    2-многообразная поверхность)."""
    faces = faces.copy()
    n = len(faces)
    edge_faces: dict[tuple[int, int], list[int]] = {}
    for fi, (a, b, c) in enumerate(faces):
        for p, q in ((a, b), (b, c), (c, a)):
            edge_faces.setdefault((min(p, q), max(p, q)), []).append(fi)
    if any(len(v) != 2 for v in edge_faces.values()):
        return None

    adjacency: list[list[tuple[int, int, int]]] = [[] for _ in range(n)]  # (сосед, p, q)
    for (p, q), (f1, f2) in edge_faces.items():
        adjacency[f1].append((f2, p, q))
        adjacency[f2].append((f1, p, q))

    def directed_edge(face: np.ndarray, p: int, q: int) -> tuple[int, int]:
        a, b, c = face
        for x, y in ((a, b), (b, c), (c, a)):
            if {x, y} == {p, q}:
                return (x, y)
        raise ValueError("edge not found in face")  # не должно случаться при валидной топологии

    visited = np.zeros(n, dtype=bool)
    for start in range(n):
        if visited[start]:
            continue
        visited[start] = True
        stack = [start]
        while stack:
            fi = stack.pop()
            for fj, p, q in adjacency[fi]:
                if visited[fj]:
                    continue
                dir_i = directed_edge(faces[fi], p, q)
                dir_j = directed_edge(faces[fj], p, q)
                if dir_i == dir_j:  # общее ребро обходится ОДИНАКОВО -> сосед развёрнут неверно
                    faces[fj] = faces[fj][::-1]
                visited[fj] = True
                stack.append(fj)

    tri = vertices[faces]
    signed_volume = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0
    if signed_volume < 0:
        faces = faces[:, ::-1]
    return faces


def cone_mesh_from_mask(
    mask: np.ndarray,
    apex: np.ndarray,
    forward: np.ndarray,
    right: np.ndarray,
    down: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    far_distance: float,
    contour_epsilon_px: float = 1.0,
) -> trimesh.Trimesh | None:
    """Обобщённый (возможно невыпуклый) конус обзора одной камеры — замкнутый треугольный меш
    (апекс + невыпуклая дальняя крышка на глубине `far_distance`). None — вырожденная маска
    (пустая/контур < 3 точек) или ошибка триангуляции/сшивки."""
    mask_u8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)  # крупнейший контур — тело объекта
    if contour_epsilon_px > 0:
        contour = cv2.approxPolyDP(contour, contour_epsilon_px, closed=True)
    pts = contour.reshape(-1, 2).astype(np.float64)
    if len(pts) < _MIN_CONTOUR_POINTS:
        return None

    dx = (pts[:, 0] - cx) / fx
    dy = (pts[:, 1] - cy) / fy

    polygon2d = Polygon(np.stack([dx, dy], axis=-1))
    if not polygon2d.is_valid:
        polygon2d = polygon2d.buffer(0)  # пиксельный контур может самопересекаться на уступах
    if polygon2d.is_empty or polygon2d.geom_type != "Polygon" or len(polygon2d.exterior.coords) < 4:
        return None

    try:
        cap_uv, cap_faces = trimesh.creation.triangulate_polygon(polygon2d)
    except Exception:
        return None
    if len(cap_uv) < 3 or len(cap_faces) == 0:
        return None

    # Дедупликация вершин кольца ДО построения KDTree — см. докстринг
    # `exact_polyhedral_hull._view_cone_mesh` (историческая версия этой функции) для разбора
    # найденного не-2-многообразного бага без этого шага.
    rounded = np.round(cap_uv, 9)
    _, unique_idx, inverse = np.unique(rounded, axis=0, return_index=True, return_inverse=True)
    cap_uv = cap_uv[unique_idx]
    cap_faces = inverse[cap_faces].reshape(cap_faces.shape)
    if len(cap_uv) < 3 or len(cap_faces) == 0:
        return None

    def lift(uv: np.ndarray, depth: float) -> np.ndarray:
        return apex[None, :] + depth * (
            forward[None, :] + uv[:, 0:1] * right[None, :] + uv[:, 1:2] * down[None, :]
        )

    cap_verts = lift(cap_uv, far_distance)  # (M, 3) — дальняя крышка

    boundary_uv = np.asarray(polygon2d.exterior.coords[:-1])  # без дублирующей замыкающей точки
    tree = cKDTree(cap_uv)
    dist, boundary_idx = tree.query(boundary_uv)
    if np.any(dist > _MATCH_TOL):
        return None  # earcut не должен добавлять/терять граничные точки

    n_cap = len(cap_uv)
    apex_idx = n_cap
    ring = boundary_idx
    side_faces = np.stack(
        [np.full(len(ring), apex_idx), ring, np.roll(ring, -1)], axis=-1
    )

    vertices = np.vstack([cap_verts, apex[None, :]])
    faces = np.vstack([cap_faces, side_faces])
    faces = fix_winding(vertices, faces)
    if faces is None:
        return None

    cone = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if not cone.is_watertight:
        return None
    return cone
