"""Анализ круглости восстановленного сечения: Rin, Rout, k=Rin/Rout, PASS/FAIL.

Rout — минимальная описанная окружность (cv2.minEnclosingCircle) по вершинам полигона, точно,
без растеризации. Rin — максимальная вписанная окружность через cv2.distanceTransform
растеризованного полигона (см. ТЗ, раздел "Анализ круглости").
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class RoundnessResult:
    r_in: float
    r_out: float
    k: float
    passed: bool
    center_in: tuple[float, float]  # мировые координаты (Y,Z) центра вписанной окружности
    center_out: tuple[float, float]  # мировые координаты (Y,Z) центра описанной окружности


def compute_projection_roundness(
    points_2d: np.ndarray,
    resolution_px: int = 512,
    threshold: float = 0.8,
) -> RoundnessResult | None:
    """Круглость ПРОЕКЦИИ облака точек: метрики считаются по выпуклой оболочке точек.

    Именно выпуклая оболочка проекции задаёт радиус контакта объекта при вращении вокруг
    оси проецирования, поэтому для оценки потенциала к перекату используется она, а не сам
    (возможно невыпуклый) контур. None — вырожденная проекция (< 3 несовпадающих точек)."""
    pts = np.asarray(points_2d, dtype=np.float32)
    if len(pts) < 3:
        return None
    hull = cv2.convexHull(pts)[:, 0, :]
    if len(hull) < 3:
        return None
    if cv2.contourArea(hull) <= 0:
        return None
    return compute_roundness(hull, resolution_px=resolution_px, threshold=threshold)


def compute_roundness(
    coords: np.ndarray,
    resolution_px: int = 1024,
    threshold: float = 0.8,
) -> RoundnessResult:
    """`coords` — (N,2) вершины полигона (контур сечения или выпуклая оболочка проекции),
    без Shapely-зависимости. Принимает как открытый контур (cv2.convexHull), так и замкнутый
    (последняя точка == первой, как отдают `reconstruct_section*`) — замыкающая точка, если
    есть, отбрасывается."""
    coords = np.asarray(coords, dtype=np.float64)
    if len(coords) > 1 and np.allclose(coords[0], coords[-1]):
        coords = coords[:-1]  # (N,2), без замыкающей

    (cx, cy), r_out = cv2.minEnclosingCircle(coords.astype(np.float32))

    bbox_min = coords.min(axis=0)
    bbox_max = coords.max(axis=0)
    extent = max(float((bbox_max - bbox_min).max()), 1e-9)
    margin = extent * 0.1
    origin = bbox_min - margin
    extent_with_margin = extent + 2 * margin
    px_per_unit = resolution_px / extent_with_margin

    # Sub-pixel растеризация (fillPoly shift) — без неё усечение координат до целых пикселей
    # систематически "срезает" наклонные грани полигона и занижает Rin на ~1%.
    shift_bits = 8
    px = np.round((coords - origin) * px_per_unit * (1 << shift_bits)).astype(np.int32)
    mask = np.zeros((resolution_px, resolution_px), dtype=np.uint8)
    cv2.fillPoly(mask, [px], 255, shift=shift_bits)

    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    row, col = np.unravel_index(np.argmax(dist), dist.shape)
    r_in = float(dist[row, col]) / px_per_unit
    center_in = (origin[0] + col / px_per_unit, origin[1] + row / px_per_unit)

    k = r_in / r_out if r_out > 0 else 0.0

    return RoundnessResult(
        r_in=r_in,
        r_out=float(r_out),
        k=k,
        passed=k >= threshold,
        center_in=center_in,
        center_out=(float(cx), float(cy)),
    )
