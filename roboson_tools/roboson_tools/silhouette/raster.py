"""Перевод между пиксельными координатами растра силуэта и мировыми координатами меша."""

from __future__ import annotations

import numpy as np


def pixel_row_for_axis_pos(axis_pos: float, axis_min: float, px_per_unit: float, height: int) -> int:
    row = int(round(height - (axis_pos - axis_min) * px_per_unit))
    return int(np.clip(row, 0, height - 1))


def world_u_from_pixel_x(px_x: float, u_min: float, px_per_unit: float) -> float:
    return u_min + px_x / px_per_unit


def row_extent(mask_row: np.ndarray) -> tuple[int, int] | None:
    """Индексы первого и последнего True-пикселя строки маски, либо None если строка пустая."""
    idx = np.flatnonzero(mask_row)
    if idx.size == 0:
        return None
    return int(idx[0]), int(idx[-1])
