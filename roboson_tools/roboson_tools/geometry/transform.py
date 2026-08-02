"""Ориентация объекта: матрица поворота из roll/pitch/yaw и применение к мешу.

Конвенция (зафиксирована здесь и в README): R = Rz(yaw) @ Ry(pitch) @ Rx(roll), поворот
применяется к уже центрированным вершинам вокруг начала координат.
"""

from __future__ import annotations

import numpy as np

from .mesh_io import Mesh


def rotation_matrix(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    roll, pitch, yaw = np.radians([roll_deg, pitch_deg, yaw_deg])

    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])

    return rz @ ry @ rx


def apply_orientation(mesh: Mesh, roll_deg: float, pitch_deg: float, yaw_deg: float) -> Mesh:
    r = rotation_matrix(roll_deg, pitch_deg, yaw_deg)
    rotated = mesh.vertices @ r.T
    return Mesh(vertices=rotated, faces=mesh.faces)
