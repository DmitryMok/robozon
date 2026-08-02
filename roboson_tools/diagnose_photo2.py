"""Диагностика 3D-реконструкции по фото phone2: воспроизводит РЕАЛЬНЫЙ пайплайн диалога
(PhotoCaptureDialog._segmentation_jobs_for_item -> SAM3 -> select_region_candidates ->
carve_consensus), пошагово (2 наблюдения одного фото -> постепенно добавляем остальные фото,
затем belt-only без зеркала) и на каждом шаге сверяет проекцию полученного polytope-халла
(выпуклый -> проекция = convexHull проекций вершин, ТОЧНО, без аппроксимации) с реальной маской
сегментации через IoU. Плюс сохраняет overlay-картинки (зелёный контур = маска SAM3, красный =
проекция халла) для визуальной проверки.

Запуск (Windows venv, из WSL):
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  /mnt/c/Projects/.venvs/venv-roboson-tools/Scripts/python.exe diagnose_photo2.py
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
    fit_mirror_plane,
    select_region_candidates,
    uncrop_mask,
)
from roboson_tools.segmentation.sam3_subprocess_backend import Sam3SubprocessBackend
from roboson_tools.visual_hull import pose_carving

PHOTO_DIR = Path("assets/photo/phone2")
OUT_DIR = Path("diagnose_photo2_out")
PROMPT = "bottle"


def log(msg: str) -> None:
    print(msg, flush=True)


def project_world_points(points_world: np.ndarray, apex, forward, right, down, fx, fy, cx, cy) -> np.ndarray:
    d = points_world - apex[None, :]
    depth = d @ forward
    dx = (d @ right) / depth
    dy = (d @ down) / depth
    cols = cx + fx * dx
    rows = cy + fy * dy
    return np.column_stack([cols, rows])


def hull_silhouette_mask(vertices_world: np.ndarray, apex, forward, right, down, fx, fy, cx, cy, shape) -> np.ndarray:
    """Точная проекция ВЫПУКЛОГО полиэдра (`vertices_world` — все вершины polytope-халла) на
    плоскость кадра: проекция выпуклого тела = выпуклая оболочка проекций его вершин."""
    pts = project_world_points(vertices_world, apex, forward, right, down, fx, fy, cx, cy).astype(np.float32)
    mask = np.zeros(shape, dtype=np.uint8)
    if len(pts) < 3:
        return mask > 0
    hull = cv2.convexHull(pts)
    cv2.fillConvexPoly(mask, np.round(hull).astype(np.int32), 255)
    return mask > 0


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = (a & b).sum()
    union = (a | b).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def overlay(image_bgr: np.ndarray, real_mask: np.ndarray, proj_mask: np.ndarray) -> np.ndarray:
    out = image_bgr.copy()
    real_c, _ = cv2.findContours(real_mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    proj_c, _ = cv2.findContours(proj_mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, real_c, -1, (0, 255, 0), 3)  # зелёный = реальная маска SAM3
    cv2.drawContours(out, proj_c, -1, (0, 0, 255), 2)  # красный = проекция халла
    return out


class Item:
    def __init__(self, path: Path):
        self.path = path
        self.image_bgr = cv2.imread(str(path))
        self.pose = None
        self.belt_region = None
        self.mirror_region = None
        self.belt_candidates: list[np.ndarray] = []
        self.mirror_candidates: list[np.ndarray] = []


def photo_number(path: Path) -> int:
    m = re.search(r"photo_(\d+)_", path.name)
    return int(m.group(1)) if m else 0


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    cache = load_calibration_cache(calibration_cache_path(PHOTO_DIR))
    if cache is None:
        log(f"Нет кэша калибровки в {PHOTO_DIR}")
        return
    layout, intrinsics = cache.layout, cache.intrinsics
    log(f"Калибровка: RMS {cache.rms_reprojection_error_pct}, доска-референс {cache.reference_board}")

    paths = sorted(PHOTO_DIR.glob("photo_*.jpg"), key=photo_number)
    items = [Item(p) for p in paths]
    log(f"Фото: {[p.name for p in paths]}")

    # --- позы ---
    for it in items:
        est = estimate_camera_pose(it.image_bgr, layout, intrinsics)
        it.pose = est
        log(
            f"{it.path.name}: поза {est.status}, ошибка репроекции "
            f"{est.reprojection_error_pct if est.reprojection_error_pct is not None else '-'}"
        )

    # --- регионы belt/mirror по позе ---
    for it in items:
        if it.pose.pose is None:
            continue
        it.belt_region, it.mirror_region = belt_and_mirror_regions(layout, it.pose.pose)

    # --- сегментация SAM3 (кадрированные belt/mirror кропы, промпт "bottle") ---
    backend = Sam3SubprocessBackend()
    jobs = []  # (item, region_name, offset)
    crops = []
    for it in items:
        if it.belt_region is not None:
            cropped = crop_to_regions(it.image_bgr, (it.belt_region,))
            if cropped is not None:
                image, offset = cropped
                jobs.append((it, "belt", offset))
                crops.append(image)
        if it.mirror_region is not None:
            cropped = crop_to_regions(it.image_bgr, (it.mirror_region,))
            if cropped is not None:
                image, offset = cropped
                jobs.append((it, "mirror", offset))
                crops.append(image)

    log(f"Запускаю SAM3 на {len(crops)} кропах, промпт='{PROMPT}' (первый вызов ~15с — грузится модель)...")
    results = backend.segment_batch(crops, prompt=PROMPT)
    backend.shutdown()

    for (it, region_name, offset), result in zip(jobs, results):
        if result is None:
            log(f"{it.path.name}/{region_name}: 0 детекций")
            continue
        full_shape = it.image_bgr.shape[:2]
        all_masks_full = [uncrop_mask(m, offset, full_shape) for m in result.all_masks]
        region_poly = it.belt_region if region_name == "belt" else it.mirror_region
        candidates = select_region_candidates(all_masks_full, region_poly)
        log(
            f"{it.path.name}/{region_name}: {len(result.all_masks)} детекций всего, "
            f"{len(candidates)} прошли геометрический фильтр (conf лучшей={result.confidence:.2f})"
        )
        if region_name == "belt":
            it.belt_candidates = candidates
        else:
            it.mirror_candidates = candidates

    # --- сборка наблюдений ---
    mirror_plane = fit_mirror_plane(layout)

    def make_observation(it: Item, region_name: str) -> pose_carving.CandidateObservation | None:
        cands = it.belt_candidates if region_name == "belt" else it.mirror_candidates
        if not cands:
            return None
        return pose_carving.CandidateObservation(
            candidates=cands,
            pose=it.pose.pose,
            mirror_plane=mirror_plane if region_name == "mirror" else None,
            label=f"{it.path.name}/{region_name}",
        )

    def report_and_check(tag: str, obs_list, ready_meta):
        """ready_meta: list[(item, region_name)] parallel to obs_list, для проекционной проверки."""
        if len(obs_list) < 2:
            log(f"[{tag}] < 2 наблюдений, пропуск")
            return None
        consensus = pose_carving.carve_consensus(obs_list)
        if consensus.polytope is None:
            log(f"[{tag}] polytope: None (вырожденный случай); switched={consensus.switched_labels} dropped={consensus.dropped_labels}")
            return None
        dims = consensus.polytope.vertices.max(axis=0) - consensus.polytope.vertices.min(axis=0)
        log(
            f"[{tag}] polytope dims: {dims[0]:.0f}x{dims[1]:.0f}x{dims[2]:.0f} мм "
            f"(exact: {'ok' if consensus.exact is not None else 'None'}); "
            f"switched={consensus.switched_labels} dropped={consensus.dropped_labels}"
        )
        # Проекционная проверка на каждом наблюдении, реально вошедшем в халл
        ious = []
        for i, (it, region_name) in enumerate(ready_meta):
            chosen_idx = consensus.chosen[i]
            if chosen_idx is None:
                continue
            real_mask = obs_list[i].candidates[chosen_idx]
            if region_name == "belt":
                apex = it.pose.pose.camera_pos_world
                forward, right, down = it.pose.pose.forward_world, it.pose.pose.right_world, it.pose.pose.down_world
            else:
                apex, forward, right, down = pose_carving.reflect_camera_frame(it.pose.pose, mirror_plane)
            intr = it.pose.pose.intrinsics
            proj_mask = hull_silhouette_mask(
                consensus.polytope.vertices, apex, forward, right, down,
                intr.fx, intr.fy, intr.cx, intr.cy, real_mask.shape,
            )
            score = iou(real_mask, proj_mask)
            ious.append(score)
            log(f"    {it.path.name}/{region_name}: IoU(проекция халла, маска SAM3) = {score:.3f}")
            out_path = OUT_DIR / f"{tag}__{it.path.name}__{region_name}.png"
            cv2.imwrite(str(out_path), overlay(it.image_bgr, real_mask, proj_mask))
        if ious:
            log(f"[{tag}] средний IoU = {np.mean(ious):.3f}")
        return dims

    # ================= Шаг 1: 2 наблюдения ОДНОГО фото (belt + mirror) =================
    log("\n=== Шаг 1: belt+mirror одного фото ===")
    first_both = next(
        (it for it in items if it.belt_candidates and it.mirror_candidates), None
    )
    if first_both is not None:
        obs = [make_observation(first_both, "belt"), make_observation(first_both, "mirror")]
        meta = [(first_both, "belt"), (first_both, "mirror")]
        report_and_check(f"step1_{first_both.path.stem}", obs, meta)
    else:
        log("Ни одно фото не дало belt+mirror кандидатов одновременно")

    # ================= Шаг 2: постепенно добавляем фото (belt+mirror) =================
    log("\n=== Шаг 2: постепенное добавление фото (belt+mirror) ===")
    accum_obs, accum_meta = [], []
    for it in items:
        for region_name in ("belt", "mirror"):
            o = make_observation(it, region_name)
            if o is not None:
                accum_obs.append(o)
                accum_meta.append((it, region_name))
        report_and_check(f"step2_upto_{it.path.stem}", accum_obs, accum_meta)

    # ================= Шаг 3: belt-only (без отражений) =================
    log("\n=== Шаг 3: belt-only, все фото, без зеркала ===")
    belt_obs, belt_meta = [], []
    for it in items:
        o = make_observation(it, "belt")
        if o is not None:
            belt_obs.append(o)
            belt_meta.append((it, "belt"))
    report_and_check("step3_belt_only", belt_obs, belt_meta)

    log(f"\nOverlay-картинки сохранены в {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
