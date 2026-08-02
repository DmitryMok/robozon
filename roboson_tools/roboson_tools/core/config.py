"""Загрузка config/objects.yaml и config/app_settings.yaml."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"


@dataclass
class ObjectEntry:
    key: str
    label: str
    path: Path
    is_round_reference: bool


@dataclass
class AppSettings:
    stl_dir: Path
    raster_resolution_px: int
    roundness_threshold: float
    model_check_samples: int
    axis_step_deg: float
    camera_fov_deg: float
    # Разрешение сенсора реальной камеры: (строки = вдоль ленты, столбцы = поперёк); int в
    # yaml тоже принимается (квадратный кадр). Отдельно от raster_resolution_px (Analytical
    # Mode) — это параметр ЖЕЛЕЗА, а не растеризации.
    camera_resolution_px: tuple[int, int]
    # Супсемплинг силуэтов Camera Mode (мягкие субпиксельные маски, 1 = бинарные).
    camera_supersample: int


def load_app_settings(config_dir: Path = CONFIG_DIR) -> AppSettings:
    data = yaml.safe_load((config_dir / "app_settings.yaml").read_text(encoding="utf-8"))
    stl_dir = config_dir.parent / data["stl_dir"]
    cam_res_raw = data.get("camera_resolution_px", data["raster_resolution_px"])
    if isinstance(cam_res_raw, (list, tuple)):
        camera_resolution = (int(cam_res_raw[0]), int(cam_res_raw[1]))
    else:
        camera_resolution = (int(cam_res_raw), int(cam_res_raw))
    return AppSettings(
        stl_dir=stl_dir,
        raster_resolution_px=int(data["raster_resolution_px"]),
        roundness_threshold=float(data["roundness_threshold"]),
        model_check_samples=int(data["model_check_samples"]),
        axis_step_deg=float(data.get("axis_step_deg", 5.0)),
        camera_fov_deg=float(data.get("camera_fov_deg", 80.0)),
        camera_resolution_px=camera_resolution,
        camera_supersample=int(data.get("camera_supersample", 1)),
    )


def load_objects(config_dir: Path = CONFIG_DIR) -> dict[str, ObjectEntry]:
    data = yaml.safe_load((config_dir / "objects.yaml").read_text(encoding="utf-8"))
    settings = load_app_settings(config_dir)
    objects: dict[str, ObjectEntry] = {}
    for key, entry in data["objects"].items():
        objects[key] = ObjectEntry(
            key=key,
            label=entry["label"],
            path=settings.stl_dir / entry["path"],
            is_round_reference=bool(entry["is_round_reference"]),
        )
    return objects
