"""Экспорт эксперимента: STL, ориентация, углы, силуэты, Visual Hull, метрики — в директорию."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from ..core.experiment import EvaluationResult, Orientation, roundness_to_dict


def _save_silhouette_png(mask: np.ndarray, path: Path) -> None:
    cv2.imwrite(str(path), mask.astype(np.uint8) * 255)


def _render_section_image(result: EvaluationResult, resolution_px: int = 1024) -> np.ndarray | None:
    """Полигон сечения + вписанная/описанная окружность, для архивного PNG."""
    if result.polygon_coords is None:
        return None

    coords = np.array(result.polygon_coords[:-1], dtype=np.float64)
    bbox_min = coords.min(axis=0)
    bbox_max = coords.max(axis=0)
    extent = max(float((bbox_max - bbox_min).max()), 1e-9)
    margin = extent * 0.15
    origin = bbox_min - margin
    extent_with_margin = extent + 2 * margin
    px_per_unit = resolution_px / extent_with_margin

    img = np.full((resolution_px, resolution_px, 3), 255, dtype=np.uint8)
    px = ((coords - origin) * px_per_unit).astype(np.int32)
    cv2.polylines(img, [px], isClosed=True, color=(40, 40, 40), thickness=2)

    if result.roundness is not None:
        r = result.roundness
        c_out = tuple(((np.array(r.center_out) - origin) * px_per_unit).astype(int))
        cv2.circle(img, c_out, int(r.r_out * px_per_unit), (80, 80, 200), 2)
        c_in = tuple(((np.array(r.center_in) - origin) * px_per_unit).astype(int))
        cv2.circle(img, c_in, int(r.r_in * px_per_unit), (80, 150, 80), 2)

    return img


def export_experiment(
    stl_path: Path,
    orientation: Orientation,
    result: EvaluationResult,
    out_dir: Path,
) -> Path:
    """Сохраняет в `out_dir`: копию STL, силуэты (silhouettes/*.png), картинку сечения
    (visual_hull.png) и manifest.json (ориентация, углы, полигон, метрики)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    silhouettes_dir = out_dir / "silhouettes"
    silhouettes_dir.mkdir(exist_ok=True)

    shutil.copy2(stl_path, out_dir / stl_path.name)

    for angle, silhouette in result.silhouettes.items():
        _save_silhouette_png(silhouette.mask, silhouettes_dir / f"angle_{angle:g}.png")

    section_img = _render_section_image(result)
    if section_img is not None:
        cv2.imwrite(str(out_dir / "visual_hull.png"), section_img)

    manifest = {
        "stl": stl_path.name,
        "orientation": {
            "roll_deg": orientation.roll_deg,
            "pitch_deg": orientation.pitch_deg,
            "yaw_deg": orientation.yaw_deg,
        },
        "view_angles_deg": result.view_angles_deg,
        "axis_pos": result.axis_pos,
        "bbox_min": list(result.bbox_min),
        "bbox_max": list(result.bbox_max),
        "polygon_coords": result.polygon_coords,
        "roundness": roundness_to_dict(result.roundness),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return out_dir
