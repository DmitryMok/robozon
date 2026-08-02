"""Сохраняет рядом с фото phone2 картинки с выделенными масками SAM3 (belt зелёным, mirror
голубым, полупрозрачная заливка + контур) — для визуального изучения качества границы маски.
Воспроизводит тот же пайплайн (кадрирование по региону, промпт "bottle", геометрический отбор
кандидатов), что и diagnose_photo2.py/PhotoCaptureDialog, но без реконструкции — только маски.

Запуск (Windows venv, из WSL):
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  /mnt/c/Projects/.venvs/venv-roboson-tools/Scripts/python.exe save_phone2_masks.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from roboson_tools.pose.aruco_estimator import estimate_camera_pose
from roboson_tools.pose.calibration_cache import calibration_cache_path, load_calibration_cache
from roboson_tools.segmentation.region_filter import (
    belt_and_mirror_regions,
    crop_to_regions,
    select_region_candidates,
    uncrop_mask,
)
from roboson_tools.segmentation.sam3_subprocess_backend import Sam3SubprocessBackend

PHOTO_DIR = Path("assets/photo/phone2")
OUT_DIR = PHOTO_DIR / "masks_review"
PROMPT = "bottle"

BELT_COLOR = (0, 200, 0)  # BGR — зелёный
MIRROR_COLOR = (255, 180, 0)  # BGR — голубой/бирюзовый


def log(msg: str) -> None:
    print(msg, flush=True)


def photo_number(path: Path) -> int:
    m = re.search(r"photo_(\d+)_", path.name)
    return int(m.group(1)) if m else 0


def draw_mask(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.35) -> None:
    overlay = image.copy()
    overlay[mask] = color
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, dst=image)
    contours, _ = cv2.findContours(mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, color, 2)


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    cache = load_calibration_cache(calibration_cache_path(PHOTO_DIR))
    if cache is None:
        log(f"Нет кэша калибровки в {PHOTO_DIR}")
        return
    layout, intrinsics = cache.layout, cache.intrinsics

    paths = sorted(PHOTO_DIR.glob("photo_*.jpg"), key=photo_number)
    images = {p: cv2.imread(str(p)) for p in paths}
    poses = {p: estimate_camera_pose(images[p], layout, intrinsics) for p in paths}
    regions = {
        p: belt_and_mirror_regions(layout, poses[p].pose) if poses[p].pose is not None else (None, None)
        for p in paths
    }

    jobs = []  # (path, region_name, offset)
    crops = []
    for p in paths:
        belt_region, mirror_region = regions[p]
        if belt_region is not None:
            cropped = crop_to_regions(images[p], (belt_region,))
            if cropped is not None:
                image, offset = cropped
                jobs.append((p, "belt", offset))
                crops.append(image)
        if mirror_region is not None:
            cropped = crop_to_regions(images[p], (mirror_region,))
            if cropped is not None:
                image, offset = cropped
                jobs.append((p, "mirror", offset))
                crops.append(image)

    log(f"Запускаю SAM3 на {len(crops)} кропах, промпт='{PROMPT}'...")
    backend = Sam3SubprocessBackend()
    results = backend.segment_batch(crops, prompt=PROMPT)
    backend.shutdown()

    masks: dict[Path, dict[str, np.ndarray | None]] = {p: {"belt": None, "mirror": None} for p in paths}
    for (p, region_name, offset), result in zip(jobs, results):
        if result is None:
            continue
        full_shape = images[p].shape[:2]
        all_masks_full = [uncrop_mask(m, offset, full_shape) for m in result.all_masks]
        belt_region, mirror_region = regions[p]
        region_poly = belt_region if region_name == "belt" else mirror_region
        candidates = select_region_candidates(all_masks_full, region_poly)
        if candidates:
            masks[p][region_name] = candidates[0]

    for p in paths:
        out = images[p].copy()
        if masks[p]["belt"] is not None:
            draw_mask(out, masks[p]["belt"], BELT_COLOR)
        if masks[p]["mirror"] is not None:
            draw_mask(out, masks[p]["mirror"], MIRROR_COLOR)
        out_path = OUT_DIR / f"{p.stem}_masks.png"
        cv2.imwrite(str(out_path), out)
        log(
            f"{p.name}: belt={'ok' if masks[p]['belt'] is not None else 'нет'}, "
            f"mirror={'ok' if masks[p]['mirror'] is not None else 'нет'} -> {out_path.name}"
        )

    log(f"\nСохранено в {OUT_DIR.resolve()} (зелёный=belt, голубой=mirror)")


if __name__ == "__main__":
    main()
