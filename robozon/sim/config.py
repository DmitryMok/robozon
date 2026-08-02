"""Загрузка конфигурации симуляции (config/objects.yaml, config/layout.yaml)."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_objects(path: Path | None = None) -> dict:
    path = path or ROOT / "config" / "objects.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)["objects"]


def load_layout(path: Path | None = None) -> dict:
    path = path or ROOT / "config" / "layout.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
