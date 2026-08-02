"""Analytical Mode — ортографическая проекция вдоль направления наблюдения theta.

Физический прообраз: объект движется по конвейеру вдоль оси X ("roll axis"); камеры/зеркало
образуют кольцо вокруг этой оси и смотрят на объект сбоку, поперёк направления движения —
azimuth theta отсчитывается вокруг X в плоскости Y-Z. Направление взгляда
v_theta = (0, cos theta, sin theta); поперечная (горизонтальная в изображении) ось силуэта
r_theta = (0, -sin theta, cos theta). Координата X не участвует в проекции и напрямую даёт
строку растра — отсюда прямое соответствие "строка растра <-> положение вдоль X" без искажений.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..geometry.mesh_io import Mesh, triangle_soup
from .base import Silhouette

PRINCIPAL_AXIS = np.array([1.0, 0.0, 0.0])  # ось X — "roll axis" / направление движения


def view_axes(view_angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Возвращает (v_theta, r_theta) — направление взгляда и поперечную ось силуэта, обе в
    плоскости Y-Z (перпендикулярной продольной оси X)."""
    theta = np.radians(view_angle_deg)
    v = np.array([0.0, np.cos(theta), np.sin(theta)])
    r = np.array([0.0, -np.sin(theta), np.cos(theta)])
    return v, r


def common_extent(mesh: Mesh, view_angles_deg: list[float]) -> float:
    """Максимальный габарит силуэта по всем углам наблюдения — общий extent, при котором
    все силуэты набора растеризуются с одинаковым px_per_unit (единый масштаб)."""
    verts = mesh.vertices
    axis_vals = verts @ PRINCIPAL_AXIS
    extent = float(axis_vals.max() - axis_vals.min())
    for angle in view_angles_deg:
        _, r = view_axes(angle)
        u = verts @ r
        extent = max(extent, float(u.max() - u.min()))
    return max(extent, 1e-9)


def build_silhouette(
    mesh: Mesh,
    view_angle_deg: float,
    resolution_px: int,
    margin_ratio: float = 0.05,
    extent_override: float | None = None,
) -> Silhouette:
    _, r = view_axes(view_angle_deg)

    verts = mesh.vertices  # (V,3)
    u = verts @ r
    axis_pos = verts @ PRINCIPAL_AXIS

    u_min, u_max = float(u.min()), float(u.max())
    axis_min, axis_max = float(axis_pos.min()), float(axis_pos.max())
    extent = max(u_max - u_min, axis_max - axis_min, 1e-9)
    if extent_override is not None:
        # Единый масштаб для всего набора углов (см. common_extent) — px_per_unit одинаков,
        # силуэты в GUI сравнимы между собой напрямую.
        extent = max(extent, extent_override)
    # Центрируем объект в растре (важно при extent_override: свой габарит силуэта может быть
    # меньше общего — без центрирования объект прижимался бы к углу картинки).
    u_min = (u_min + u_max) / 2.0 - extent / 2.0
    axis_min = (axis_min + axis_max) / 2.0 - extent / 2.0

    margin = extent * margin_ratio
    u_min -= margin
    axis_min -= margin
    extent_with_margin = extent + 2 * margin

    px_per_unit = resolution_px / extent_with_margin
    height = width = resolution_px

    tris = triangle_soup(mesh)  # (F,3,3)
    tri_u = tris @ r  # (F,3)
    tri_axis = tris @ PRINCIPAL_AXIS  # (F,3)

    px_x = (tri_u - u_min) * px_per_unit
    px_y = height - (tri_axis - axis_min) * px_per_unit  # верх изображения = большой X
    tri_px = np.stack([px_x, px_y], axis=-1).astype(np.int32)  # (F,3,2)

    mask = np.zeros((height, width), dtype=np.uint8)
    # Заливка по одному треугольнику: cv2.fillPoly со списком полигонов работает по правилу
    # чёт-нечет — перекрывающиеся передние/задние грани меша взаимно гасятся и внутри
    # силуэта остаются дыры. fillConvexPoly в цикле даёт честное объединение.
    for tri in tri_px:
        cv2.fillConvexPoly(mask, tri, 255)

    return Silhouette(
        mask=mask.astype(bool),
        view_angle_deg=view_angle_deg,
        u_min=u_min,
        axis_min=axis_min,
        px_per_unit=px_per_unit,
    )
