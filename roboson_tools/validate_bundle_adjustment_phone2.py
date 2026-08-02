"""Проверка задачи "реализовать bundle adjustment для уточнения МНК-плоскости зеркала" на
РЕАЛЬНЫХ фото phone2: сравнивает регистрацию досок рига ДО (register_boards — попарное усреднение
переходов, как сейчас в проде, 2 прохода) и ПОСЛЕ (+ bundle_adjust_registration поверх того же
результата) — по (1) RMS репроекции меток на bridge-фото, (2) как изменилась МНК-плоскость
зеркала (`fit_mirror_plane`, точка+нормаль), (3) reprojection_error_pct КАЖДОГО фото при оценке
его собственной позы через merge_layout(reg2) vs merge_layout(reg3) (тот же путь, что реально
использует PhotoCaptureDialog/estimate_camera_pose).

Не трогает калибровку интринсиков (calibrate_camera) — переиспользует уже посчитанные интринсики
из существующего кэша `assets/photo/phone2/calibration_result.yaml` (не то, что уточняет эта
задача); сравнивает только эффект bundle adjustment на регистрацию досок/плоскость зеркала.

Запуск (Windows venv, из WSL):
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  /mnt/c/Projects/.venvs/venv-roboson-tools/Scripts/python.exe validate_bundle_adjustment_phone2.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from roboson_tools.pose.aruco_estimator import estimate_camera_pose
from roboson_tools.pose.board_registration import bundle_adjust_registration, merge_layout, register_boards
from roboson_tools.pose.calibration_cache import calibration_cache_path, load_calibration_cache
from roboson_tools.pose.marker_layout import (
    BELT_BOARD_PATH,
    MIRROR_BOARD_PATH,
    load_all_boards,
)
from roboson_tools.segmentation.region_filter import fit_mirror_plane

PHOTO_DIR = Path("assets/photo/phone2")
_REFERENCE_BOARD = "belt_left"


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> None:
    cache = load_calibration_cache(calibration_cache_path(PHOTO_DIR))
    if cache is None:
        log(f"Нет кэша калибровки в {PHOTO_DIR}")
        return
    intrinsics = cache.intrinsics
    log(f"Интринсики из кэша: fx={intrinsics.fx:.1f} fy={intrinsics.fy:.1f} "
        f"(RMS калибровки {cache.rms_reprojection_error_pct})")

    boards = load_all_boards(MIRROR_BOARD_PATH, BELT_BOARD_PATH)
    log(f"Доски рига: {sorted(boards)}")

    paths = sorted(PHOTO_DIR.glob("photo_*.jpg"))
    images = [cv2.imread(str(p)) for p in paths]
    names = [p.name for p in paths]
    log(f"Фото: {len(images)}")

    reg2 = register_boards(images, boards, intrinsics, reference_board=_REFERENCE_BOARD)
    log(f"\n[register_boards] {reg2.message}")
    for board_name, t in sorted(reg2.transforms.items()):
        log(f"  {board_name}: t={np.round(t.translation, 1)}")

    reg3, report = bundle_adjust_registration(images, boards, intrinsics, reg2)
    log(f"\n[bundle_adjust_registration] n_frames_used={report.n_frames_used} "
        f"boards_refined={report.boards_refined}")
    if report.rms_reprojection_px_before is not None:
        log(f"  RMS репроекции (bridge-кадры этой оптимизации): "
            f"{report.rms_reprojection_px_before:.3f}px -> {report.rms_reprojection_px_after:.3f}px")
    for board_name, t in sorted(reg3.transforms.items()):
        log(f"  {board_name}: t={np.round(t.translation, 1)}")

    def board_diff(name: str) -> None:
        t2, t3 = reg2.transforms.get(name), reg3.transforms.get(name)
        if t2 is None or t3 is None:
            return
        d_t = np.linalg.norm(t2.translation - t3.translation)
        cos_angle = np.clip((np.trace(t2.rotation @ t3.rotation.T) - 1.0) / 2.0, -1.0, 1.0)
        d_r = np.degrees(np.arccos(cos_angle))
        log(f"  доска {name}: сдвиг перевода {d_t:.2f}мм, поворот {d_r:.3f}°")

    log("\n=== Разница register_boards vs bundle_adjust_registration по доскам ===")
    for name in sorted(reg2.transforms):
        if name != _REFERENCE_BOARD:
            board_diff(name)

    merged2 = merge_layout(boards, reg2)
    merged3 = merge_layout(boards, reg3)

    plane2 = fit_mirror_plane(merged2)
    plane3 = fit_mirror_plane(merged3)
    if plane2 is not None and plane3 is not None:
        p2, n2 = plane2
        p3, n3 = plane3
        if np.dot(n2, n3) < 0:
            n3 = -n3
        angle = np.degrees(np.arccos(np.clip(np.dot(n2, n3), -1.0, 1.0)))
        log(f"\n=== МНК-плоскость зеркала: register_boards vs bundle adjustment ===")
        log(f"  точка на плоскости: {np.round(p2, 1)} -> {np.round(p3, 1)} "
            f"(сдвиг {np.linalg.norm(p2 - p3):.2f}мм)")
        log(f"  нормаль: {np.round(n2, 4)} -> {np.round(n3, 4)} (угол {angle:.3f}°)")

    log("\n=== Ошибка репроекции позы КАЖДОГО фото: merge(reg2) vs merge(reg3) ===")
    errs2, errs3 = [], []
    for name, image in zip(names, images):
        e2 = estimate_camera_pose(image, merged2, intrinsics)
        e3 = estimate_camera_pose(image, merged3, intrinsics)
        p2 = e2.reprojection_error_pct
        p3 = e3.reprojection_error_pct
        if p2 is not None:
            errs2.append(p2)
        if p3 is not None:
            errs3.append(p3)
        log(f"  {name}: {p2} -> {p3}")
    if errs2 and errs3:
        log(f"\nСредняя ошибка репроекции по фото: {np.mean(errs2):.3f}% -> {np.mean(errs3):.3f}%")


if __name__ == "__main__":
    main()
