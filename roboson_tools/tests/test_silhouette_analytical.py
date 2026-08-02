"""Юнит-тест силуэта Analytical Mode: ширина проекции куба под разными углами обзора."""

from __future__ import annotations

import numpy as np
import trimesh

from roboson_tools.geometry.mesh_io import from_trimesh
from roboson_tools.silhouette.analytical import build_silhouette


def _silhouette_width_world(mask: np.ndarray, px_per_unit: float) -> float:
    cols = np.any(mask, axis=0)
    idx = np.flatnonzero(cols)
    return (idx[-1] - idx[0] + 1) / px_per_unit


def test_cube_projection_width_axis_aligned():
    w = 100.0
    mesh = from_trimesh(trimesh.creation.box(extents=(w, w, w)))

    sil0 = build_silhouette(mesh, view_angle_deg=0, resolution_px=1024)
    width0 = _silhouette_width_world(sil0.mask, sil0.px_per_unit)
    assert abs(width0 - w) / w < 0.01

    sil45 = build_silhouette(mesh, view_angle_deg=45, resolution_px=1024)
    width45 = _silhouette_width_world(sil45.mask, sil45.px_per_unit)
    expected45 = w * np.sqrt(2)
    assert abs(width45 - expected45) / expected45 < 0.01
