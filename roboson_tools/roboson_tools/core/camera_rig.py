"""Риг камер — именованные пресеты списков (угол, дистанция) для Camera Mode.

Заменяет прежнюю пару "один общий `camera_distance` на весь набор ракурсов" +
"`angle_set_presets` — список углов без дистанций": реальный риг "2 камеры + 1 зеркало" имеет
РАЗНУЮ дистанцию на разных азимутах (прямые ракурсы короче, зеркальные — длиннее за счёт
оптического пути через зеркало, см. `docs/mirror_geometry_calc.md`), поэтому дистанция должна
быть парой с углом, а не отдельным скаляром на весь риг.

fov/разрешение/супсемплинг НЕ входят в `CameraSpec` — по решению пользователя они остаются
общими на весь риг (физически все камеры одной модели), меняются только угол и дистанция
каждой камеры.

Хранилище — `config/camera_rig_presets.yaml`, версionируется в git (не QSettings): пресеты
рига — часть конфигурации эксперимента, а не персональная настройка окна одного пользователя.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .config import CONFIG_DIR

CAMERA_RIG_PRESETS_PATH = CONFIG_DIR / "camera_rig_presets.yaml"


@dataclass(frozen=True)
class CameraSpec:
    angle_deg: float
    distance_mm: float


@dataclass
class CameraRigPreset:
    name: str
    cameras: list[CameraSpec] = field(default_factory=list)
    # Встроенные (поставляемые с проектом) пресеты нельзя перезаписать/удалить под тем же
    # именем — иначе первый же "Сохранить как" с тем же именем молча стирает эталонный риг
    # ("2 камеры + зеркало"), на который ссылается вся остальная документация проекта.
    readonly: bool = False


def _validate_cameras(cameras: list[CameraSpec]) -> None:
    angles = [c.angle_deg for c in cameras]
    if len(angles) != len(set(angles)):
        raise ValueError(f"дублирующиеся углы в риге камер: {angles}")


def rig_angles(cameras: list[CameraSpec]) -> list[float]:
    return [c.angle_deg for c in cameras]


def rig_to_distances(cameras: list[CameraSpec]) -> dict[float, float]:
    return {c.angle_deg: c.distance_mm for c in cameras}


def apply_distance_to_all(cameras: list[CameraSpec], source_angle_deg: float) -> list[CameraSpec]:
    """Копирует `distance_mm` камеры `source_angle_deg` на ВСЕ камеры рига — угол каждой камеры
    не меняется (кнопка "Применить ко всем" в диалоге рига камер, см. заметку задачи)."""
    source = next((c for c in cameras if c.angle_deg == source_angle_deg), None)
    if source is None:
        raise ValueError(f"камера с углом {source_angle_deg}° не найдена в риге")
    return [CameraSpec(angle_deg=c.angle_deg, distance_mm=source.distance_mm) for c in cameras]


def _preset_from_dict(data: dict) -> CameraRigPreset:
    cameras = [
        CameraSpec(angle_deg=float(c["angle_deg"]), distance_mm=float(c["distance_mm"]))
        for c in data["cameras"]
    ]
    _validate_cameras(cameras)
    return CameraRigPreset(
        name=str(data["name"]), cameras=cameras, readonly=bool(data.get("readonly", False))
    )


def _preset_to_dict(preset: CameraRigPreset) -> dict:
    return {
        "name": preset.name,
        "readonly": preset.readonly,
        "cameras": [
            {"angle_deg": c.angle_deg, "distance_mm": c.distance_mm} for c in preset.cameras
        ],
    }


def load_camera_rig_presets(path: Path = CAMERA_RIG_PRESETS_PATH) -> list[CameraRigPreset]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [_preset_from_dict(p) for p in data["presets"]]


def save_camera_rig_presets(
    presets: list[CameraRigPreset], path: Path = CAMERA_RIG_PRESETS_PATH
) -> None:
    """Перезаписывает файл пресетов целиком — вызывающая сторона (диалог рига камер) сама
    решает порядок (см. `upsert_preset`/`delete_preset`, оставляющие относительный порядок)."""
    data = {"presets": [_preset_to_dict(p) for p in presets]}
    path.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )


def upsert_preset(
    presets: list[CameraRigPreset], preset: CameraRigPreset
) -> list[CameraRigPreset]:
    """Добавляет новый пресет или заменяет существующий с тем же именем. Перезапись
    `readonly=True` пресета запрещена — заставляет пользователя сохранять свои изменения под
    новым именем, а не тихо стирать эталонный риг."""
    existing = next((p for p in presets if p.name == preset.name), None)
    if existing is not None and existing.readonly:
        raise ValueError(f"пресет {preset.name!r} встроенный (readonly) — нельзя перезаписать")
    _validate_cameras(preset.cameras)
    return [p for p in presets if p.name != preset.name] + [preset]


def delete_preset(presets: list[CameraRigPreset], name: str) -> list[CameraRigPreset]:
    existing = next((p for p in presets if p.name == name), None)
    if existing is None:
        raise ValueError(f"пресет {name!r} не найден")
    if existing.readonly:
        raise ValueError(f"пресет {name!r} встроенный (readonly) — нельзя удалить")
    return [p for p in presets if p.name != name]
