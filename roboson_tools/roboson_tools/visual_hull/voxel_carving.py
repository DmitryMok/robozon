"""Voxel carving по перспективным силуэтам Camera Mode — визуализация "как выглядела бы
3D-реконструкция" по нескольким кадрам, каждый — своя пара (азимут камеры, положение на ленте).

**Один азимут не может ограничить форму по всем трём осям.** Силуэт с одного направления
взгляда ничего не говорит о протяжённости объекта ВДОЛЬ направления взгляда (глубина) — эта
ось остаётся ограниченной только произвольным запасом вокруг габарита объекта, а не реальной
геометрией. Несколько положений на ленте (start/center/end) ОДНОЙ и той же камеры этого не
чинят — они используют ту же ось взгляда, что и center. Поэтому реконструкция только по одной
камере выглядит как раздутый вдоль этой оси "блин" (было замечено пользователем на скриншоте —
один бок объекта визуально куда меньше другого). Чтобы ограничить объект по всем осям, нужно
пересекать силуэты с РАЗНЫХ азимутов — как минимум 2 взаимно поперечных (та же логика, что и
у Analytical Mode: 0°/45°/90°/135°, "2 камеры + 1 зеркало"). См. заметку задачи "Добавить
Camera Mode..." — там же обсуждение, почему start/end несут РЕАЛЬНУЮ дополнительную информацию
о форме сверх center: при смещении объекта к краю кадра лучи от камеры идут под углом к
продольной оси X, а не строго поперёк неё — в кадре проявляется часть торцевой грани, которую
орто-проекция (и центральный perspective-кадр) принципиально не видит.

В отличие от `visual_hull/reconstruct.py` (точное 2D-пересечение bands-призм для
ортографических силуэтов), перспективная "полоса" от одного силуэта — конус/усечённая
пирамида, не призма: пересечение полигонов сечения для неё не подходит (см. докстринг задачи).
Voxel carving обходит это естественно — вместо пересечения многоугольников каждый воксель
трёхмерной сетки проверяется на попадание в маску КАЖДОГО активного силуэта (space carving,
классический приём, работающий для любой модели проекции без явной геометрии пересечения
конусов).

ИСКЛЮЧИТЕЛЬНО для визуализации на Panel 1 (см. `gui/panels/model_panel.set_camera_hull_points`).
НЕ подключено к габаритам/круглой проекции — те по-прежнему считаются Analytical Mode
band-пересечением (см. `core/experiment.py`, `visual_hull/reconstruct.py`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry.mesh_io import Mesh, bounding_box
from ..silhouette.analytical import PRINCIPAL_AXIS
from ..silhouette.camera import (
    BeltPosition,
    belt_offset,
    build_silhouette,
    camera_frame,
    project_points,
)


@dataclass
class VoxelHull:
    points: np.ndarray  # (N, 3) центры уцелевших вокселей, мировые координаты (без сдвига belt)
    voxel_size: float


def carve(
    mesh: Mesh,
    views: list[tuple[float, BeltPosition]],
    fov_deg: float,
    camera_distances: dict[float, float],
    resolution_px: int = 512,
    grid_resolution: int = 40,
    margin_ratio: float = 0.08,
    crop_to_object: bool = False,
) -> VoxelHull | None:
    """Пересечение (по вокселям) объёмов, видимых силуэтами каждого кадра из `views` —
    список пар (view_angle_deg, belt_position). Для полноценного ограничения формы по всем
    трём осям нужно НЕСКОЛЬКО РАЗНЫХ азимутов (одного азимута с любым числом belt_position
    недостаточно — см. докстринг модуля). `camera_distances` — обязательная карта
    {азимут -> дистанция камеры до оси ленты}, ПО ОДНОЙ на каждый азимут, встречающийся в
    `views` (см. `core/camera_rig.py`) — риг "2 камеры + 1 зеркало" имеет разную дистанцию на
    прямых и зеркальных ракурсах, единого камера-расстояния на весь риг больше нет.
    `KeyError` — в `views` есть азимут, отсутствующий в `camera_distances` (намеренно без
    скрытого фоллбека — рассинхрон этих двух наборов даёт молча неверную реконструкцию).
    None — вырожденный случай (объект не виден хотя бы в одном кадре).

    `crop_to_object=True` — обрезать кадр до bbox маски + margin в каждом силуэте (см.
    `silhouette/camera.build_silhouette`): маска плотнее покрывает объект, ускоряя проверку
    попадания вокселей в маску (меньше пикселей в `silhouette.mask`) и уменьшая растровый
    шум на тонких объектах. Оптический центр берётся из `Silhouette.cx`/`Silhouette.cy`,
    если они не None, иначе из `camera_frame` (полный кадр)."""
    bbox_min, bbox_max = bounding_box(mesh.vertices)
    extent = bbox_max - bbox_min
    margin = float(extent.max()) * margin_ratio if extent.max() > 0 else 1.0
    lo = bbox_min - margin
    hi = bbox_max + margin

    axes = [np.linspace(lo[i], hi[i], grid_resolution) for i in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)

    survive = np.ones(len(grid), dtype=bool)
    for view_angle_deg, belt_position in views:
        camera_distance = camera_distances[view_angle_deg]
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
        shifted = grid + np.array([dx, 0.0, 0.0])
        # При ROI (silhouette.cx/cy не None) — проект в обрезанную маску (cx/cy из силуэта),
        # иначе — в полный кадр (camera_frame, как раньше). f_px от ROI не зависит (только от
        # fov и полного resolution_px). project_points всегда использует camera_frame (полный
        # кадр) — при ROI делаем проекцию вручную со cx/cy из силуэта.
        _cp, _fwd, _r, f_px, cx_full, cy_full = camera_frame(
            view_angle_deg, fov_deg, camera_distance, resolution_px
        )
        cx = silhouette.cx if silhouette.cx is not None else cx_full
        cy = silhouette.cy if silhouette.cy is not None else cy_full
        rel = shifted - _cp
        depth = rel @ _fwd
        px_col = f_px * (rel @ _r) / depth + cx
        px_row = cy - f_px * (rel @ PRINCIPAL_AXIS) / depth
        cols = np.round(px_col).astype(np.int64)
        rows = np.round(px_row).astype(np.int64)
        in_bounds = (
            (depth > 1e-6)
            & (rows >= 0)
            & (rows < silhouette.mask.shape[0])
            & (cols >= 0)
            & (cols < silhouette.mask.shape[1])
        )
        inside = np.zeros(len(grid), dtype=bool)
        idx = np.flatnonzero(in_bounds)
        inside[idx] = silhouette.mask[rows[idx], cols[idx]]
        survive &= inside

        if not survive.any():
            return None

    voxel_size = float(np.min((hi - lo) / max(grid_resolution - 1, 1)))
    return VoxelHull(points=grid[survive], voxel_size=voxel_size)
