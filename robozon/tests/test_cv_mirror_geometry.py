"""Тесты геометрии CV-рига (без Webots): позиция/ориентация зеркальных камер.

Задача — [[Рассчитать геометрию 3-камерного рига и зеркала для CV
(CV-пайплайн Webots)]]. Две реальные находки по пути:
1. Попытка построить ориентацию зеркальной камеры через reflect_dir(forward
   реальной камеры) целилась мимо объекта на ~1.3м (отражение НАПРАВЛЕНИЯ
   реальной камеры — не то же самое, что направление "на объект" из
   отражённой ПОЗИЦИИ при офсете зеркала t≠0, тот же класс ошибки, что 2φ-θ
   для азимута). Решение — прицел строится напрямую как "объект минус
   позиция камеры", не отражением направления.
2. Дефолтная ориентация Camera в Webots (нулевой rotation) — локальная ось
   X ВПЕРЁД, Z ВВЕРХ (не "-Z вперёд, Y вверх", как ошибочно предполагалось
   изначально и как физически работает типичный OpenGL-конвеншен) — баг был
   не только в зеркальных, но и в ПРЯМЫХ камерах (top смотрела вдоль ленты,
   а не вниз), найден визуально пользователем на реальном захваченном кадре.
   См. tools/gen_world.py::_camera_frame.
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.config import load_layout   # noqa: E402
from tools.gen_world import (        # noqa: E402
    _camera_frame,
    _matrix_to_axis_angle,
    _mirror_camera_rotation,
    _rotation_from_frame,
)


def _camera_world_pos(cam: dict, rig: dict) -> tuple[float, float, float]:
    theta = math.radians(cam["angle"])
    d = rig["mirror_distance_by_angle"][float(cam["angle"])] if cam.get("mirror") else cam["distance"]
    return rig["x"], d * math.cos(theta), rig["belt_z"] + d * math.sin(theta)


def _camera_forward(cam: dict) -> tuple[float, float, float]:
    """Направление взгляда для rotation `1 0 0 (angle-90°)` из дефолта (0,0,-1)."""
    theta = math.radians(cam["angle"])
    return 0.0, -math.cos(theta), -math.sin(theta)


def test_mirror_cameras_point_at_the_belt_axis_object():
    """Луч из позиции каждой mirror-камеры вдоль её forward должен проходить
    точно через точку наблюдения (x=rig.x, y=0, z=belt_z) — ту же точку, на
    которую нацелены прямые камеры. Именно это не выполнялось для отвергнутого
    варианта с reflect_dir(forward) (промах ~1.3м по Y)."""
    layout = load_layout()
    rig = layout["cv_rig"]
    target = (rig["x"], 0.0, rig["belt_z"])

    for cam_name, cam in rig["cameras"].items():
        if not cam.get("mirror"):
            continue
        pos = _camera_world_pos(cam, rig)
        fwd = _camera_forward(cam)
        to_target = tuple(t - p for t, p in zip(target, pos))
        dist = math.sqrt(sum(c * c for c in to_target))
        to_target_n = tuple(c / dist for c in to_target)
        dot = sum(a * b for a, b in zip(fwd, to_target_n))
        assert dot > 0.9999, f"{cam_name}: forward не совпадает с направлением на объект (dot={dot})"


def test_mirror_distance_matches_true_azimuth_path_length():
    """mirror_distance_by_angle[angle] должно совпадать с фактическим
    расстоянием от вычисленной позиции камеры до точки наблюдения (иначе кадр
    будет либо слишком крупным, либо слишком мелким относительно расчёта)."""
    layout = load_layout()
    rig = layout["cv_rig"]
    target = (rig["x"], 0.0, rig["belt_z"])

    for cam_name, cam in rig["cameras"].items():
        if not cam.get("mirror"):
            continue
        pos = _camera_world_pos(cam, rig)
        dist = math.sqrt(sum((p - t) ** 2 for p, t in zip(pos, target)))
        expected = rig["mirror_distance_by_angle"][float(cam["angle"])]
        assert abs(dist - expected) < 1e-6, f"{cam_name}: {dist} != {expected}"


def _axis_angle_to_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    ax = np.asarray(axis, dtype=float)
    kx, ky, kz = ax
    k = np.array([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]])
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * (k @ k)


def test_direct_camera_rotation_looks_along_local_x():
    """Реальный найденный баг: rotation-поле для прямых камер строилось как
    поворот вокруг оси X из предположения "дефолт Webots-камеры = -Z вперёд" —
    неверно (эмпирически найдено: дефолт = +X вперёд, см. `_camera_frame`).
    Тест фиксирует находку: применение итогового VRML rotation к ЛОКАЛЬНОЙ
    оси X (та самая ось, что Webots действительно использует как forward)
    должно давать направление точно на объект — для всех азимутов рига."""
    layout = load_layout()
    rig = layout["cv_rig"]
    for cam_name, cam in rig["cameras"].items():
        if cam.get("mirror"):
            continue
        theta = math.radians(cam["angle"])
        forward, left, up = _camera_frame(theta)
        axis, angle = _rotation_from_frame(forward, left, up)
        rmat = _axis_angle_to_matrix(axis, angle)
        local_x = np.array([1.0, 0.0, 0.0])
        rendered_forward = rmat @ local_x
        assert np.allclose(rendered_forward, forward, atol=1e-9), (
            f"{cam_name}: локальная +X после поворота ({rendered_forward}) "
            f"не совпадает с расчётным forward ({forward})"
        )


def test_mirror_camera_rendered_forward_points_at_object():
    """То же самое для зеркальных камер — экстринсика строится отдельной
    функцией (`_mirror_camera_rotation`), проверяем её результат так же:
    локальная +X (реальный forward Webots) после применения посчитанного
    rotation должна указывать точно на точку наблюдения."""
    layout = load_layout()
    rig = layout["cv_rig"]
    target_offset = np.array([0.0, 0.0, 0.0])  # объект в системе, смещённой от точки наблюдения

    for cam_name, cam in rig["cameras"].items():
        if not cam.get("mirror"):
            continue
        source_angle = rig["cameras"][cam["reflects"]]["angle"]
        d = rig["mirror_distance_by_angle"][float(cam["angle"])]
        axis, angle = _mirror_camera_rotation(source_angle, cam["angle"], d, rig)
        rmat = _axis_angle_to_matrix(axis, angle)
        local_x = np.array([1.0, 0.0, 0.0])
        rendered_forward = rmat @ local_x

        theta_own = math.radians(cam["angle"])
        pos_offset = np.array([0.0, d * math.cos(theta_own), d * math.sin(theta_own)])
        expected_forward = target_offset - pos_offset
        expected_forward /= np.linalg.norm(expected_forward)

        assert np.allclose(rendered_forward, expected_forward, atol=1e-9), (
            f"{cam_name}: локальная +X после поворота ({rendered_forward}) "
            f"не совпадает с направлением на объект ({expected_forward})"
        )


if __name__ == "__main__":
    test_mirror_cameras_point_at_the_belt_axis_object()
    test_mirror_distance_matches_true_azimuth_path_length()
    test_direct_camera_rotation_looks_along_local_x()
    test_mirror_camera_rendered_forward_points_at_object()
    print("OK")
