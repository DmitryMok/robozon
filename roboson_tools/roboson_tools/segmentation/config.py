"""Конфигурация backend'а сегментации — `config/segmentation.yaml`. Отдельный файл (не
`app_settings.yaml`), по аналогии с `camera_intrinsics.yaml`/`aruco_markers.yaml`: описывает
внешнюю систему (путь к соседнему проекту UavVisionLab и его Windows venv), а не геометрию стенда.

Классы (текстовый промпт SAM3), которые пользователь правит из диалога (поле «Классы» рядом с
кнопкой «Сегментация...»), сохраняются НЕ в `segmentation.yaml` (там ручные комментарии,
затирать их при каждом сохранении не хочется), а в отдельный маленький state-файл — тот же
приём, что уже применён для классов SAM3 в `UavVisionLab/nodes/cv/sam3/node.py`
(`_CLASSES_STATE_FILE`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from ..core.config import CONFIG_DIR

SEGMENTATION_CONFIG_PATH = CONFIG_DIR / "segmentation.yaml"
SEGMENTATION_CLASSES_STATE_PATH = CONFIG_DIR / "state" / "segmentation_classes.json"


@dataclass(frozen=True)
class SegmentationConfig:
    prompt: str
    conf_threshold: float
    imgsz: int
    uavvisionlab_root: Path
    python_exe: Path
    timeout_s: float


def load_segmentation_config(path: Path | str = SEGMENTATION_CONFIG_PATH) -> SegmentationConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return SegmentationConfig(
        prompt=str(data["prompt"]),
        conf_threshold=float(data.get("conf_threshold", 0.30)),
        imgsz=int(data.get("imgsz", 1008)),
        uavvisionlab_root=Path(data["uavvisionlab_root"]),
        python_exe=Path(data["python_exe"]),
        timeout_s=float(data.get("timeout_s", 180)),
    )


def load_classes_text(path: Path | str = SEGMENTATION_CLASSES_STATE_PATH) -> str | None:
    """Текст поля «Классы» (через запятую), сохранённый в предыдущем запуске. `None`, если
    состояния ещё нет — вызывающая сторона подставляет `SegmentationConfig.prompt`."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    text = data.get("classes_text")
    return str(text) if text else None


def save_classes_text(text: str, path: Path | str = SEGMENTATION_CLASSES_STATE_PATH) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"classes_text": text}, ensure_ascii=False), encoding="utf-8")
