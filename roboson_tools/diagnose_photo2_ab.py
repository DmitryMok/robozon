"""A/B: тот же пайплайн diagnose_photo2.py, но SAM3 гоняется ОДИН раз, а carve_consensus -
дважды на одних и тех же масках: с реальными dist_coeffs (после фикса) и с dist_coeffs=None
(как было ДО фикса дисторсии) - изолирует именно эффект коррекции, без шума от повторного
вызова SAM3 (не полностью детерминирован call-to-call, см. заметку задачи)."""
from __future__ import annotations

import dataclasses
import re
import sys
from pathlib import Path

import cv2
import numpy as np



from roboson_tools.pose.aruco_estimator import estimate_camera_pose
from roboson_tools.pose.calibration_cache import calibration_cache_path, load_calibration_cache
from roboson_tools.pose.camera_pose import CameraPose
from roboson_tools.segmentation.region_filter import (
    belt_and_mirror_regions, crop_to_regions, fit_mirror_plane, select_region_candidates, uncrop_mask,
)
from roboson_tools.segmentation.sam3_subprocess_backend import Sam3SubprocessBackend
from roboson_tools.visual_hull import pose_carving

PHOTO_DIR = Path("assets/photo/phone2")
PROMPT = "bottle"


def log(msg):
    print(msg, flush=True)


class Item:
    def __init__(self, path):
        self.path = path
        self.image_bgr = cv2.imread(str(path))
        self.pose = None
        self.belt_region = None
        self.mirror_region = None
        self.belt_candidates = []
        self.mirror_candidates = []


def photo_number(path):
    m = re.search(r"photo_(\d+)_", path.name)
    return int(m.group(1)) if m else 0


def strip_dist(pose):
    intr = pose.intrinsics
    intr_no_dist = dataclasses.replace(intr, dist_coeffs=None)
    return CameraPose(rotation=pose.rotation, translation=pose.translation, intrinsics=intr_no_dist)


def main():
    cache = load_calibration_cache(calibration_cache_path(PHOTO_DIR))
    layout, intrinsics = cache.layout, cache.intrinsics
    log(f"dist_coeffs исходной калибровки: {intrinsics.dist_coeffs}")

    paths = sorted(PHOTO_DIR.glob("photo_*.jpg"), key=photo_number)
    items = [Item(p) for p in paths]

    for it in items:
        it.pose = estimate_camera_pose(it.image_bgr, layout, intrinsics)

    for it in items:
        if it.pose.pose is None:
            continue
        it.belt_region, it.mirror_region = belt_and_mirror_regions(layout, it.pose.pose)

    backend = Sam3SubprocessBackend()
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

    log(f"SAM3 на {len(crops)} кропах...")
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

    def make_obs(it, region_name, use_dist):
        cands = it.belt_candidates if region_name == "belt" else it.mirror_candidates
        if not cands:
            return None
        pose = it.pose.pose if use_dist else strip_dist(it.pose.pose)
        return pose_carving.CandidateObservation(
            candidates=cands, pose=pose,
            mirror_plane=mirror_plane if region_name == "mirror" else None,
            label=f"{it.path.name}/{region_name}",
        )

    def run(label, region_names, use_dist):
        obs = []
        for it in items:
            for region_name in region_names:
                o = make_obs(it, region_name, use_dist)
                if o is not None:
                    obs.append(o)
        if len(obs) < 2:
            log(f"[{label}] < 2 наблюдений")
            return
        consensus = pose_carving.carve_consensus(obs)
        if consensus.polytope is None:
            log(f"[{label}] polytope: None")
            return
        dims_p = consensus.polytope.vertices.max(axis=0) - consensus.polytope.vertices.min(axis=0)
        dims_e = None
        if consensus.exact is not None:
            dims_e = consensus.exact.vertices.max(axis=0) - consensus.exact.vertices.min(axis=0)
        log(
            f"[{label:<28}] polytope={dims_p[0]:6.1f}x{dims_p[1]:6.1f}x{dims_p[2]:6.1f}мм  "
            f"exact={'—' if dims_e is None else f'{dims_e[0]:6.1f}x{dims_e[1]:6.1f}x{dims_e[2]:6.1f}мм'}  "
            f"dropped={consensus.dropped_labels}"
        )

    log("\n=== ПОЛНЫЙ набор (belt+mirror, 18 наблюдений) ===")
    run("full  WITH dist_coeffs (после фикса)", ("belt", "mirror"), True)
    run("full  WITHOUT dist_coeffs (до фикса)", ("belt", "mirror"), False)

    log("\n=== belt-only (9 наблюдений, без зеркала) ===")
    run("belt  WITH dist_coeffs (после фикса)", ("belt",), True)
    run("belt  WITHOUT dist_coeffs (до фикса)", ("belt",), False)

    log("\n=== mirror-only (9 наблюдений, только зеркало) ===")
    run("mirror WITH dist_coeffs (после фикса)", ("mirror",), True)
    run("mirror WITHOUT dist_coeffs (до фикса)", ("mirror",), False)


if __name__ == "__main__":
    main()
