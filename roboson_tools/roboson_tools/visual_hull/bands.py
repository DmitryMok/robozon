"""Полоса (band) — реконструкция ширины сечения на позиции X из одного силуэта.

band(theta, X) — это пересечение двух полуплоскостей вдоль направления взгляда theta,
ограниченное по ширине крайними закрашенными пикселями силуэта в строке, соответствующей X.
Представляется как прямоугольный shapely.Polygon достаточной длины вдоль theta, чтобы после
пересечения bands от всех углов получить корректный ограниченный полигон сечения.
"""

from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon

from ..silhouette.base import Silhouette
from ..silhouette.raster import pixel_row_for_axis_pos, row_extent, world_u_from_pixel_x


def build_band(silhouette: Silhouette, axis_pos: float, half_length: float) -> Polygon | None:
    """Возвращает None, если силуэт в этой позиции пуст (вырожденный случай)."""
    height = silhouette.mask.shape[0]
    row = pixel_row_for_axis_pos(axis_pos, silhouette.axis_min, silhouette.px_per_unit, height)
    extent = row_extent(silhouette.mask[row, :])
    if extent is None:
        return None

    col_min, col_max = extent
    u_lo = world_u_from_pixel_x(col_min, silhouette.u_min, silhouette.px_per_unit)
    u_hi = world_u_from_pixel_x(col_max + 1, silhouette.u_min, silhouette.px_per_unit)

    theta = np.radians(silhouette.view_angle_deg)
    v = np.array([np.cos(theta), np.sin(theta)])  # направление взгляда в плоскости Y-Z
    r = np.array([-np.sin(theta), np.cos(theta)])  # поперечная ось силуэта в плоскости Y-Z

    corners = [
        u_lo * r - half_length * v,
        u_hi * r - half_length * v,
        u_hi * r + half_length * v,
        u_lo * r + half_length * v,
    ]
    return Polygon(corners)
