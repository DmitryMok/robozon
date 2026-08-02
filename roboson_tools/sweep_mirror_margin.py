"""Ищет практическое решение проблемы, найденной в diagnose_photo2.py: mirror-наблюдения
систематически "подрезают" истинный объём (виртуальная камера дальше от объекта -> та же
угловая ошибка позы/плоскости зеркала даёт большую позиционную ошибку в мм на объекте, см.
заметку задачи). Маски САМИ по себе точные (подтверждено пользователем и overlay в
masks_review/) — проблема не в контуре, а в том, что `carve_polytope`/`carve_exact` берут ТОЧНОЕ
пересечение полупространств без допуска: один чуть смещённый конус обзора режет реальный объём.

Решение: дать mirror-наблюдениям (и только им — belt уже хорошо согласованы) РАВНОМЕРНЫЙ допуск
в виде дилатации маски на K пикселей ПЕРЕД построением полупространств/конуса. Почему именно
пиксели, а не мм: угловая (не метрическая) ошибка позы/плоскости зеркала при проекции на объект
на расстоянии D даёт позиционную ошибку ~D*θ — эта же угловая ошибка в пиксельном пространстве
контура постоянна независимо от D (см. геометрию `_cone_geometry.py`: `dx=(col-cx)/fx`), а при
обратной проекции конуса на глубину D автоматически масштабируется в ту же D*θ. Т.е. константный
пиксельный допуск = константный угловой допуск = физически верная модель источника шума (в
отличие от константного допуска в мм, который бы одинаково недо/пере-жимал разные дистанции).

Валидация — той же метрикой, что и в diagnose_photo2.py: IoU проекции халла (построенного
С допуском) против РЕАЛЬНОЙ (без дилатации) маски SAM3.
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
PROMPT = "bottle"
MARGINS_PX = [0, 5, 10, 15, 20, 25, 30, 40, 55, 70]


def log(msg: str) -> None:
    print(msg, flush=True)


def photo_number(path: Path) -> int:
    m = re.search(r"photo_(\d+)_", path.name)
    return int(m.group(1)) if m else 0


def project_world_points(points_world, apex, forward, right, down, fx, fy, cx, cy):
    d = points_world - apex[None, :]
    depth = d @ forward
    dx = (d @ right) / depth
    dy = (d @ down) / depth
    return np.column_stack([cx + fx * dx, cy + fy * dy])


def hull_silhouette_mask(vertices_world, apex, forward, right, down, fx, fy, cx, cy, shape):
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


def dilate_mask(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(mask.astype(np.uint8), kernel) > 0


class Item:
    def __init__(self, path: Path):
        self.path = path
        self.image_bgr = cv2.imread(str(path))
        self.pose = None
        self.belt_region = None
        self.mirror_region = None
        self.belt_candidates: list[np.ndarray] = []
        self.mirror_candidates: list[np.ndarray] = []


def main() -> None:
    cache = load_calibration_cache(calibration_cache_path(PHOTO_DIR))
    layout, intrinsics = cache.layout, cache.intrinsics
    paths = sorted(PHOTO_DIR.glob("photo_*.jpg"), key=photo_number)
    items = [Item(p) for p in paths]

    for it in items:
        it.pose = estimate_camera_pose(it.image_bgr, layout, intrinsics)
    for it in items:
        if it.pose.pose is not None:
            it.belt_region, it.mirror_region = belt_and_mirror_regions(layout, it.pose.pose)

    jobs, crops = [], []
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

    log(f"SAM3 на {len(crops)} кропах (один раз, дальше только пересчёт геометрии)...")
    backend = Sam3SubprocessBackend()
    results = backend.segment_batch(crops, prompt=PROMPT)
    backend.shutdown()

    for (it, region_name, offset), result in zip(jobs, results):
        if result is None:
            continue
        full_shape = it.image_bgr.shape[:2]
        all_masks_full = [uncrop_mask(m, offset, full_shape) for m in result.all_masks]
        region_poly = it.belt_region if region_name == "belt" else it.mirror_region
        candidates = select_region_candidates(all_masks_full, region_poly)
        if region_name == "belt":
            it.belt_candidates = candidates
        else:
            it.mirror_candidates = candidates

    mirror_plane = fit_mirror_plane(layout)
    items_both = [it for it in items if it.belt_candidates and it.mirror_candidates]
    log(f"Фото с belt+mirror кандидатами: {len(items_both)}/{len(items)}")

    mean_reproj = np.mean([it.pose.reprojection_error_pct for it in items_both])

    def make_meta(mirror_margin_px, per_photo_scaled: bool = False):
        obs_list, meta = [], []
        for it in items_both:
            obs_list.append(
                pose_carving.CandidateObservation(
                    candidates=it.belt_candidates, pose=it.pose.pose, mirror_plane=None,
                    label=f"{it.path.name}/belt",
                )
            )
            meta.append((it, "belt", it.belt_candidates))
            if per_photo_scaled:
                k = mirror_margin_px * (it.pose.reprojection_error_pct / mean_reproj)
            else:
                k = mirror_margin_px
            dilated = [dilate_mask(m, round(k)) for m in it.mirror_candidates]
            obs_list.append(
                pose_carving.CandidateObservation(
                    candidates=dilated, pose=it.pose.pose, mirror_plane=mirror_plane,
                    label=f"{it.path.name}/mirror",
                )
            )
            meta.append((it, "mirror", it.mirror_candidates))  # ORIGINAL (undilated) для валидации IoU
        return obs_list, meta

    def evaluate(obs_list, meta, label):
        consensus = pose_carving.carve_consensus(obs_list)
        if consensus.polytope is None:
            log(f"{label:>9} | None (вырожденный случай)")
            return
        dims = consensus.polytope.vertices.max(axis=0) - consensus.polytope.vertices.min(axis=0)
        belt_ious, mirror_ious = [], []
        for i, (it, region_name, orig_candidates) in enumerate(meta):
            chosen_idx = consensus.chosen[i]
            if chosen_idx is None:
                continue
            real_mask = orig_candidates[min(chosen_idx, len(orig_candidates) - 1)]
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
            (belt_ious if region_name == "belt" else mirror_ious).append(score)
        all_ious = belt_ious + mirror_ious
        log(
            f"{label:>9} | {dims[0]:5.0f}x{dims[1]:5.0f}x{dims[2]:5.0f} | "
            f"{np.mean(all_ious):.3f}   | {min(all_ious):.3f}   | "
            f"{np.mean(belt_ious):.3f}    | {np.mean(mirror_ious):.3f}"
        )

    log("\n--- Единый допуск для ВСЕХ mirror-наблюдений ---")
    log("  margin | dims (мм)            | avg IoU | min IoU | belt avg | mirror avg")
    for margin_px in MARGINS_PX:
        evaluate(*make_meta(margin_px), label=str(margin_px))

    log("\n--- Допуск, пропорциональный ошибке репроекции ПОЗЫ ЭТОГО фото (относительно среднего) ---")
    log("  scale   | dims (мм)            | avg IoU | min IoU | belt avg | mirror avg")
    for scale in [10, 15, 20, 25, 30, 40, 50]:
        evaluate(*make_meta(scale, per_photo_scaled=True), label=str(scale))


if __name__ == "__main__":
    main()


def debug_per_observation(margin_px: int = 15) -> None:
    cache = load_calibration_cache(calibration_cache_path(PHOTO_DIR))
    layout, intrinsics = cache.layout, cache.intrinsics
    paths = sorted(PHOTO_DIR.glob("photo_*.jpg"), key=photo_number)
    items = [Item(p) for p in paths]
    for it in items:
        it.pose = estimate_camera_pose(it.image_bgr, layout, intrinsics)
    for it in items:
        if it.pose.pose is not None:
            it.belt_region, it.mirror_region = belt_and_mirror_regions(layout, it.pose.pose)
    jobs, crops = [], []
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
    backend = Sam3SubprocessBackend()
    results = backend.segment_batch(crops, prompt=PROMPT)
    backend.shutdown()
    for (it, region_name, offset), result in zip(jobs, results):
        if result is None:
            continue
        full_shape = it.image_bgr.shape[:2]
        all_masks_full = [uncrop_mask(m, offset, full_shape) for m in result.all_masks]
        region_poly = it.belt_region if region_name == "belt" else it.mirror_region
        candidates = select_region_candidates(all_masks_full, region_poly)
        if region_name == "belt":
            it.belt_candidates = candidates
        else:
            it.mirror_candidates = candidates
    mirror_plane = fit_mirror_plane(layout)
    items_both = [it for it in items if it.belt_candidates and it.mirror_candidates]

    obs_list, meta = [], []
    for it in items_both:
        obs_list.append(pose_carving.CandidateObservation(
            candidates=it.belt_candidates, pose=it.pose.pose, mirror_plane=None, label=f"{it.path.name}/belt"))
        meta.append((it, "belt", it.belt_candidates))
        dilated = [dilate_mask(m, margin_px) for m in it.mirror_candidates]
        obs_list.append(pose_carving.CandidateObservation(
            candidates=dilated, pose=it.pose.pose, mirror_plane=mirror_plane, label=f"{it.path.name}/mirror"))
        meta.append((it, "mirror", it.mirror_candidates))

    consensus = pose_carving.carve_consensus(obs_list)
    dims = consensus.polytope.vertices.max(axis=0) - consensus.polytope.vertices.min(axis=0)
    log(f"margin={margin_px}px dims={dims[0]:.0f}x{dims[1]:.0f}x{dims[2]:.0f}mm")
    for i, (it, region_name, orig_candidates) in enumerate(meta):
        chosen_idx = consensus.chosen[i]
        if chosen_idx is None:
            log(f"  {it.path.name}/{region_name}: ИСКЛЮЧЕНО консенсусом")
            continue
        real_mask = orig_candidates[min(chosen_idx, len(orig_candidates) - 1)]
        if region_name == "belt":
            apex = it.pose.pose.camera_pos_world
            forward, right, down = it.pose.pose.forward_world, it.pose.pose.right_world, it.pose.pose.down_world
        else:
            apex, forward, right, down = pose_carving.reflect_camera_frame(it.pose.pose, mirror_plane)
        intr = it.pose.pose.intrinsics
        proj_mask = hull_silhouette_mask(
            consensus.polytope.vertices, apex, forward, right, down,
            intr.fx, intr.fy, intr.cx, intr.cy, real_mask.shape)
        score = iou(real_mask, proj_mask)
        log(f"  {it.path.name}/{region_name}: IoU={score:.3f} reproj_err={it.pose.reprojection_error_pct:.3f}%")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "debug":
    debug_per_observation()


def targeted_exclusion(margin_px: int = 10, exclude=("photo_6", "photo_7")) -> None:
    cache = load_calibration_cache(calibration_cache_path(PHOTO_DIR))
    layout, intrinsics = cache.layout, cache.intrinsics
    paths = sorted(PHOTO_DIR.glob("photo_*.jpg"), key=photo_number)
    items = [Item(p) for p in paths]
    for it in items:
        it.pose = estimate_camera_pose(it.image_bgr, layout, intrinsics)
    for it in items:
        if it.pose.pose is not None:
            it.belt_region, it.mirror_region = belt_and_mirror_regions(layout, it.pose.pose)
    jobs, crops = [], []
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
    backend = Sam3SubprocessBackend()
    results = backend.segment_batch(crops, prompt=PROMPT)
    backend.shutdown()
    for (it, region_name, offset), result in zip(jobs, results):
        if result is None:
            continue
        full_shape = it.image_bgr.shape[:2]
        all_masks_full = [uncrop_mask(m, offset, full_shape) for m in result.all_masks]
        region_poly = it.belt_region if region_name == "belt" else it.mirror_region
        candidates = select_region_candidates(all_masks_full, region_poly)
        if region_name == "belt":
            it.belt_candidates = candidates
        else:
            it.mirror_candidates = candidates
    mirror_plane = fit_mirror_plane(layout)
    items_both = [it for it in items if it.belt_candidates and it.mirror_candidates]

    obs_list, meta = [], []
    for it in items_both:
        obs_list.append(pose_carving.CandidateObservation(
            candidates=it.belt_candidates, pose=it.pose.pose, mirror_plane=None, label=f"{it.path.name}/belt"))
        meta.append((it, "belt", it.belt_candidates))
        if any(ex in it.path.name for ex in exclude):
            continue  # эту mirror-наблюдение не добавляем вовсе
        dilated = [dilate_mask(m, margin_px) for m in it.mirror_candidates]
        obs_list.append(pose_carving.CandidateObservation(
            candidates=dilated, pose=it.pose.pose, mirror_plane=mirror_plane, label=f"{it.path.name}/mirror"))
        meta.append((it, "mirror", it.mirror_candidates))

    log(f"\n--- Точечное исключение {exclude} + margin={margin_px}px на оставшихся mirror ---")
    evaluate(obs_list, meta, mirror_plane, f"excl+{margin_px}")


def evaluate(obs_list, meta, mirror_plane, label):
    consensus = pose_carving.carve_consensus(obs_list)
    if consensus.polytope is None:
        log(f"{label}: None (вырожденный случай)")
        return
    dims = consensus.polytope.vertices.max(axis=0) - consensus.polytope.vertices.min(axis=0)
    belt_ious, mirror_ious = [], []
    for i, (it, region_name, orig_candidates) in enumerate(meta):
        chosen_idx = consensus.chosen[i]
        if chosen_idx is None:
            continue
        real_mask = orig_candidates[min(chosen_idx, len(orig_candidates) - 1)]
        if region_name == "belt":
            apex = it.pose.pose.camera_pos_world
            forward, right, down = it.pose.pose.forward_world, it.pose.pose.right_world, it.pose.pose.down_world
        else:
            apex, forward, right, down = pose_carving.reflect_camera_frame(it.pose.pose, mirror_plane)
        intr = it.pose.pose.intrinsics
        proj_mask = hull_silhouette_mask(
            consensus.polytope.vertices, apex, forward, right, down,
            intr.fx, intr.fy, intr.cx, intr.cy, real_mask.shape)
        score = iou(real_mask, proj_mask)
        (belt_ious if region_name == "belt" else mirror_ious).append(score)
        log(f"    {it.path.name}/{region_name}: IoU={score:.3f}")
    all_ious = belt_ious + mirror_ious
    log(
        f"{label}: dims={dims[0]:.0f}x{dims[1]:.0f}x{dims[2]:.0f}mm avg IoU={np.mean(all_ious):.3f} "
        f"min IoU={min(all_ious):.3f} belt avg={np.mean(belt_ious):.3f} mirror avg={np.mean(mirror_ious):.3f}"
    )


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "targeted":
    targeted_exclusion()


def full_photo_exclusion(margin_px: int = 10, exclude=("photo_6", "photo_7")) -> None:
    cache = load_calibration_cache(calibration_cache_path(PHOTO_DIR))
    layout, intrinsics = cache.layout, cache.intrinsics
    paths = sorted(PHOTO_DIR.glob("photo_*.jpg"), key=photo_number)
    items = [Item(p) for p in paths]
    for it in items:
        it.pose = estimate_camera_pose(it.image_bgr, layout, intrinsics)
    for it in items:
        if it.pose.pose is not None:
            it.belt_region, it.mirror_region = belt_and_mirror_regions(layout, it.pose.pose)
    jobs, crops = [], []
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
    backend = Sam3SubprocessBackend()
    results = backend.segment_batch(crops, prompt=PROMPT)
    backend.shutdown()
    for (it, region_name, offset), result in zip(jobs, results):
        if result is None:
            continue
        full_shape = it.image_bgr.shape[:2]
        all_masks_full = [uncrop_mask(m, offset, full_shape) for m in result.all_masks]
        region_poly = it.belt_region if region_name == "belt" else it.mirror_region
        candidates = select_region_candidates(all_masks_full, region_poly)
        if region_name == "belt":
            it.belt_candidates = candidates
        else:
            it.mirror_candidates = candidates
    mirror_plane = fit_mirror_plane(layout)
    items_both = [
        it for it in items
        if it.belt_candidates and it.mirror_candidates and not any(ex in it.path.name for ex in exclude)
    ]
    log(f"Фото после ПОЛНОГО исключения {exclude}: {len(items_both)}/9")

    obs_list, meta = [], []
    for it in items_both:
        obs_list.append(pose_carving.CandidateObservation(
            candidates=it.belt_candidates, pose=it.pose.pose, mirror_plane=None, label=f"{it.path.name}/belt"))
        meta.append((it, "belt", it.belt_candidates))
        dilated = [dilate_mask(m, margin_px) for m in it.mirror_candidates]
        obs_list.append(pose_carving.CandidateObservation(
            candidates=dilated, pose=it.pose.pose, mirror_plane=mirror_plane, label=f"{it.path.name}/mirror"))
        meta.append((it, "mirror", it.mirror_candidates))

    log(f"\n--- ПОЛНОЕ исключение фото {exclude} (belt+mirror) + margin={margin_px}px на mirror ---")
    evaluate(obs_list, meta, mirror_plane, f"full_excl+{margin_px}")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "fullexcl":
    full_photo_exclusion()
