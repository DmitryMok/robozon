"""Загрузка STL и базовые операции с мешем."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh


@dataclass
class Mesh:
    vertices: np.ndarray  # (V, 3) float64, центрировано по центру bounding box
    faces: np.ndarray  # (F, 3) int64, индексы в vertices


def from_trimesh(tm: "trimesh.Trimesh", center: bool = True) -> Mesh:
    """Конвертация trimesh.Trimesh -> Mesh, с центрированием по центру bounding box.

    Используется и для загрузки STL, и для синтетических мешей в тестах
    (`trimesh.creation.box(...)` и т.п.).
    """
    vertices = np.asarray(tm.vertices, dtype=np.float64)
    faces = np.asarray(tm.faces, dtype=np.int64)

    if center:
        bbox_min = vertices.min(axis=0)
        bbox_max = vertices.max(axis=0)
        vertices = vertices - (bbox_min + bbox_max) / 2.0

    return Mesh(vertices=vertices, faces=faces)


def load_stl(path: str | Path) -> Mesh:
    """Загружает STL через trimesh (ASCII/binary определяется автоматически) и центрирует
    меш по центру его bounding box."""
    tm = trimesh.load(str(path), force="mesh", process=False)
    return from_trimesh(tm)


def bounding_box(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return vertices.min(axis=0), vertices.max(axis=0)


def triangle_soup(mesh: Mesh) -> np.ndarray:
    """(F, 3, 3) — вершины треугольников по индексам faces, удобно для проекции/растеризации."""
    return mesh.vertices[mesh.faces]
