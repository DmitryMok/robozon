"""Восстановление формы объекта по силуэтам (Visual Hull) пересечением bands.

Сечение (в плоскости Y-Z, перпендикулярной продольной оси X) в заданной позиции X получается
пересечением полос, по одной на каждый view_angle. Полная форма (`reconstruct_shape_from_
silhouettes`) — стопка таких сечений по всей длине объекта: из неё считаются габариты
"по силуэтам" и строится облако точек для проверки круглой проекции (потенциала к перекату).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce

import numpy as np
from shapely.geometry import Polygon

from ..geometry.mesh_io import Mesh
from ..silhouette.analytical import build_silhouette
from ..silhouette.base import Silhouette
from .bands import build_band


def band_half_length(bbox_min: np.ndarray, bbox_max: np.ndarray) -> float:
    """Полудлина прямоугольника band, достаточная, чтобы пересечение bands от всех углов
    дало корректно ограниченный полигон (см. build_band)."""
    diag = float(np.linalg.norm(np.asarray(bbox_max) - np.asarray(bbox_min)))
    return max(diag, 1.0) * 5.0


def reconstruct_section(
    mesh: Mesh,
    view_angles_deg: list[float],
    axis_pos: float,
    resolution_px: int,
) -> Polygon | None:
    """mesh должен быть уже в целевой ориентации (поворот применяется один раз в
    core/experiment.py, до построения силуэтов по всем углам). `axis_pos` — позиция вдоль
    продольной оси X, на которой восстанавливается сечение. Возвращает None для
    вырожденного/пустого пересечения."""
    half_length = band_half_length(mesh.vertices.min(axis=0), mesh.vertices.max(axis=0))
    silhouettes = {
        angle: build_silhouette(mesh, angle, resolution_px) for angle in view_angles_deg
    }
    return reconstruct_section_from_silhouettes(silhouettes, axis_pos, half_length)


def reconstruct_section_from_silhouettes(
    silhouettes: dict[float, Silhouette],
    axis_pos: float,
    half_length: float,
) -> Polygon | None:
    """Та же реконструкция, но по уже построенным силуэтам — используется там, где одни и те
    же силуэты пересекаются на многих позициях axis_pos (сканирование по всей длине объекта,
    см. reconstruct_shape_from_silhouettes ниже), чтобы не пересчитывать растр силуэта
    заново на каждую позицию."""
    bands = []
    for silhouette in silhouettes.values():
        band = build_band(silhouette, axis_pos, half_length)
        if band is None:
            return None
        bands.append(band)

    polygon = reduce(lambda a, b: a.intersection(b), bands)
    if polygon.is_empty or polygon.geom_type != "Polygon":
        return None
    return polygon


@dataclass
class ReconstructedShape:
    """Форма объекта, восстановленная только по силуэтам (стопка сечений вдоль X).

    slices — [(x, [(y, z), ...]), ...]: позиция среза и контур сечения (замкнутый, последняя
    точка равна первой). dims — габариты (X, Y, Z) реконструкции в текущей ориентации:
    X — пересечение занятых диапазонов строк всех силуэтов, Y/Z — объединение bbox сечений.
    """

    slices: list[tuple[float, list[tuple[float, float]]]]
    x_min: float
    x_max: float
    dims: tuple[float, float, float]


def silhouettes_x_extent(silhouettes: dict[float, Silhouette]) -> tuple[float, float] | None:
    """Диапазон X, занятый объектом, по строкам растров: Visual Hull лежит внутри каждой
    экструзии силуэта, поэтому берётся пересечение диапазонов всех силуэтов."""
    x_lo, x_hi = -np.inf, np.inf
    for s in silhouettes.values():
        rows = np.flatnonzero(s.mask.any(axis=1))
        if rows.size == 0:
            return None
        height = s.mask.shape[0]
        # строка row покрывает мировой X в [axis_min + (height-row-1)/ppu, axis_min + (height-row)/ppu]
        x_hi = min(x_hi, s.axis_min + (height - rows[0]) / s.px_per_unit)
        x_lo = max(x_lo, s.axis_min + (height - rows[-1] - 1) / s.px_per_unit)
    if not np.isfinite(x_lo) or x_hi <= x_lo:
        return None
    return float(x_lo), float(x_hi)


def reconstruct_shape_from_silhouettes(
    silhouettes: dict[float, Silhouette],
    num_slices: int,
) -> ReconstructedShape | None:
    """Стопка сечений по всей занятой длине X. Вырожденные позиции (пустая строка растра,
    пустое пересечение bands) пропускаются. None — если объект не виден ни в одном силуэте."""
    extent = silhouettes_x_extent(silhouettes)
    if extent is None:
        return None
    x_lo, x_hi = extent

    # half_length band'а — по габариту растра (полностью «по силуэтам», меш не нужен)
    raster_extent = max(s.mask.shape[0] / s.px_per_unit for s in silhouettes.values())
    half_length = max(raster_extent, 1.0) * 5.0

    # Срезы по центрам num_slices равных отрезков — крайние позиции не попадают ровно на
    # границу занятых строк (там строка уже может быть пустой из-за округления растра).
    step = (x_hi - x_lo) / max(num_slices, 1)
    positions = x_lo + step * (np.arange(max(num_slices, 1)) + 0.5)

    slices: list[tuple[float, list[tuple[float, float]]]] = []
    y_lo = z_lo = np.inf
    y_hi = z_hi = -np.inf
    for pos in positions:
        polygon = reconstruct_section_from_silhouettes(silhouettes, float(pos), half_length)
        if polygon is None:
            continue
        coords = list(polygon.exterior.coords)
        arr = np.asarray(coords)
        y_lo, y_hi = min(y_lo, arr[:, 0].min()), max(y_hi, arr[:, 0].max())
        z_lo, z_hi = min(z_lo, arr[:, 1].min()), max(z_hi, arr[:, 1].max())
        slices.append((float(pos), coords))

    if not slices:
        return None

    dims = (x_hi - x_lo, float(y_hi - y_lo), float(z_hi - z_lo))
    return ReconstructedShape(slices=slices, x_min=x_lo, x_max=x_hi, dims=dims)
