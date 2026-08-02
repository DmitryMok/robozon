"""Адаптер torchhull: конвертирует наши маски + позы (camera_frame/ArUco CameraPose) в формат
torchhull.visual_hull(masks, transforms) и возвращает облако 3D-точек для G4.

Наши конвенции (silhouette/camera.py + analytical.py):
  PRINCIPAL_AXIS = (1, 0, 0) — продольная ось X (roll, направление движения по ленте).
  v_theta = (0, cos θ, sin θ) — направление от объекта к камере.
  forward = -v_theta — куда камера смотрит (OpenCV Z = forward).
  r_theta = (0, -sin θ, cos θ) — поперечная ось (OpenCV X = right, "право" в кадре).
  down   = -PRINCIPAL_AXIS = (-1, 0, 0) — "вниз" в кадре (т.к. px_row = cy - f*x/depth,
           рост X = движение ВВЕРХ по кадру => down = -X).

OpenCV convention (как требует torchhull):
  Камера: Z = forward, X = right, Y = down (стандартная OpenCV связка, Y вниз).
  Проекция: [u, v, 1]^T = K * (R | t) * [X, Y, Z, 1]^T, где (R | t) — world->camera,
  K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]].

Сборка transforms = K @ T (4x4, world->image до perspective division):
  R (3x3) = [right; down; forward] (строки = оси камеры в world frame),
  t (3,) = -R @ camera_pos,
  T (4x4) = [[R, t], [0, 0, 0, 1]],
  K (4x4) = [[fx, 0, cx, 0], [0, fy, cy, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
  transforms = K @ T.
"""
from __future__ import annotations

import numpy as np
import torch

from ..silhouette.analytical import PRINCIPAL_AXIS, view_axes


def build_transform_matrix(
    view_angle_deg: float,
    fov_deg: float,
    camera_distance: float,
    resolution_px: int | tuple[int, int],
    belt_offset_dx: float = 0.0,
) -> np.ndarray:
    """Собирает 4x4 матрицу transforms = K @ T в OpenCV convention для torchhull.visual_hull.

    `belt_offset_dx` — сдвиг объекта по X (PRINCIPAL_AXIS) для эмуляции позиции на ленте.
    Камера остаётся на месте, объект сдвигается на +dx: в world frame это эквивалентно
    камере, сдвинутой на -dx, поэтому t = -R @ (camera_pos - dx*PRINCIPAL_AXIS).
    """
    v_theta, r_theta = view_axes(view_angle_deg)
    forward = -v_theta
    camera_pos = camera_distance * v_theta
    # Оси камеры в world (строки R = [right, down, forward]):
    right = r_theta
    down = -PRINCIPAL_AXIS
    fwd = forward
    R = np.stack([right, down, fwd], axis=0)  # (3, 3)
    # Позиция камеры в world с учётом сдвига объекта: объект сдвинут на +dx*X,
    # эквивалентно камере в (camera_pos - dx*X) — это и берём для t.
    cam_pos_effective = camera_pos - belt_offset_dx * PRINCIPAL_AXIS
    t = -R @ cam_pos_effective  # (3,)
    # Intrinsics
    if isinstance(resolution_px, (tuple, list)):
        n_rows, n_cols = int(resolution_px[0]), int(resolution_px[1])
    else:
        n_rows = n_cols = int(resolution_px)
    fov = np.radians(fov_deg)
    f_px = (n_rows / 2.0) / np.tan(fov / 2.0)
    cx = n_cols / 2.0
    cy = n_rows / 2.0
    # K (4x4) — fx=fy (квадратный пиксель в нашей модели)
    K = np.array([
        [f_px, 0.0, cx, 0.0],
        [0.0, f_px, cy, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float32)
    # T (4x4) — world->camera
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R.astype(np.float32)
    T[:3, 3] = t.astype(np.float32)
    return K @ T  # (4, 4)


def build_transforms_batch(
    views: list[tuple[float, str, float]],  # (view_angle_deg, belt_position, dx)
    fov_deg: float,
    camera_distances: dict[float, float],
    resolution_px: int | tuple[int, int],
) -> torch.Tensor:
    """Собирает батч transforms (B, 4, 4) для всех views в OpenCV convention.

    `views` — список (view_angle_deg, belt_position_label, belt_offset_dx). belt_position_label
    не используется в расчёте (нужен только для документации), dx уже посчитан вызывающим
    через `silhouette.camera.belt_offset`.
    """
    transforms = np.stack([
        build_transform_matrix(ang, fov_deg, camera_distances[ang], resolution_px, dx)
        for ang, _label, dx in views
    ], axis=0)
    return torch.from_numpy(transforms).to(torch.float32)


def masks_to_torchhull(
    silhouettes: list[np.ndarray],  # каждая (H, W) bool
    device: torch.device,
) -> torch.Tensor:
    """Список 2D масок → тензор (B, H, W, 1) float32 {0,1} на device. Маски могут быть разных
    размеров (ROI) — torchhull требует одинаковый H, W, поэтому паддим до максимального."""
    h_max = max(m.shape[0] for m in silhouettes)
    w_max = max(m.shape[1] for m in silhouettes)
    out = np.zeros((len(silhouettes), h_max, w_max, 1), dtype=np.float32)
    for i, m in enumerate(silhouettes):
        h, w = m.shape
        # Паддинг справа/снизу нулями — маска объекта остаётся в верхнем-левом углу, что
        # СОЗДАЁТ проблему: torchhull считает, что оптический центр = (W/2, H/2) полного
        # кадра, а мы паддим — оптический центр смещается. Поэтому вызывающий должен передать
        # маски ОДИНАКОВОГО размера (без ROI, либо все с одним ROI-окном). Здесь мы только
        # паддим, чтобы не упасть на разном размере — НО вызывающий должен гарантировать
        # одинаковые маски или использовать `build_silhouette` без `crop_to_object`.
        out[i, :h, :w, 0] = m.astype(np.float32)
    return torch.from_numpy(out).to(device)


def visual_hull_points(
    masks: list[np.ndarray],
    transforms: torch.Tensor,
    cube_corner_bfl: list[float] | tuple[float, float, float],
    cube_length: float,
    level: int = 7,
    masks_partial: bool = False,
    device: torch.device | None = None,
) -> np.ndarray | None:
    """Вызывает torchhull.visual_hull и возвращает облако 3D-точек (N, 3) или None.

    `masks` — список 2D-масок (H, W) bool, будут преобразованы в (B, H, W, 1) float32.
    `transforms` — батч (B, 4, 4) float32 (можно torch или numpy) в OpenCV convention.
    `cube_corner_bfl`, `cube_length`, `level` — параметры области реконструкции torchhull.
    `masks_partial` — True если объекты могут быть частично обрезаны краем кадра (start/end).
    `device` — torch.device, по умолчанию cuda если доступна, иначе cpu (но torchhull CUDA-only).

    Возвращает:
      np.ndarray (N, 3) float32 — вершины восстановленного меша (мировые координаты),
      или None если реконструкция пустая/вырожденная.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import torchhull

    masks_t = masks_to_torchhull(masks, device)
    if isinstance(transforms, np.ndarray):
        transforms = torch.from_numpy(transforms)
    transforms = transforms.to(torch.float32).to(device)

    if masks_t.shape[0] != transforms.shape[0]:
        raise ValueError(f"masks B={masks_t.shape[0]} != transforms B={transforms.shape[0]}")
    if masks_t.shape[0] < 2:
        # torchhull визуально требует >=2 камер для meaningful reconstruction (1 камера
        # даёт неограниченный вдоль луча конус — как и в наших polytope/exact)
        return None

    try:
        verts, _faces = torchhull.visual_hull(
            masks=masks_t,
            transforms=transforms,
            level=level,
            cube_corner_bfl=list(cube_corner_bfl),
            cube_length=float(cube_length),
            masks_partial=masks_partial,
            transforms_convention="opencv",
            unique_verts=True,
        )
    except Exception:
        return None

    verts_np = verts.detach().cpu().numpy()
    if verts_np.size == 0 or not np.isfinite(verts_np).all():
        return None
    return verts_np.astype(np.float64)