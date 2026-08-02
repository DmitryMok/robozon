"""Слой данных рига камер (core/camera_rig.py) — пресеты угол+дистанция, без Qt.

GUI-диалог настройки рига (см. заметку задачи "Добавить настройку рига камер...") сам не
тестируется (в проекте нет pytest-qt, GUI никогда не покрывается тестами — см. остальные
test_*.py), но вся его логика (загрузка/сохранение/apply-to-all/защита readonly-пресетов)
вынесена в чистые функции этого модуля и тестируется здесь.
"""

from __future__ import annotations

import pytest

from roboson_tools.core.camera_rig import (
    CAMERA_RIG_PRESETS_PATH,
    CameraRigPreset,
    CameraSpec,
    apply_distance_to_all,
    delete_preset,
    load_camera_rig_presets,
    rig_angles,
    rig_to_distances,
    save_camera_rig_presets,
    upsert_preset,
)


def test_load_default_preset_matches_shipped_yaml():
    """Дефолтный риг "2 камеры + зеркало" — те же дистанции, что документированы в
    docs/mirror_geometry_calc.md (45°->2900мм, 135°->4908мм, прямые ракурсы->2000мм) — тот
    самый расчёт, который раньше был мёртвым полем camera_distance_mm_by_angle в
    app_settings.yaml и нигде не участвовал в 3D-реконструкции."""
    presets = load_camera_rig_presets()
    default = next(p for p in presets if p.name == "2 камеры + зеркало (дефолт)")
    assert default.readonly is True
    assert rig_to_distances(default.cameras) == {0.0: 2000.0, 45.0: 2900.0, 90.0: 2000.0, 135.0: 4908.0}
    assert rig_angles(default.cameras) == [0.0, 45.0, 90.0, 135.0]


def test_upsert_readonly_preset_raises():
    presets = load_camera_rig_presets()
    default = next(p for p in presets if p.readonly)
    tampered = CameraRigPreset(name=default.name, cameras=[CameraSpec(0.0, 1.0)])
    with pytest.raises(ValueError, match="readonly"):
        upsert_preset(presets, tampered)


def test_delete_readonly_preset_raises():
    presets = load_camera_rig_presets()
    default = next(p for p in presets if p.readonly)
    with pytest.raises(ValueError, match="readonly"):
        delete_preset(presets, default.name)


def test_delete_missing_preset_raises():
    with pytest.raises(ValueError, match="не найден"):
        delete_preset(load_camera_rig_presets(), "нет такого пресета")


def test_apply_distance_to_all_copies_distance_not_angle():
    cameras = [CameraSpec(0.0, 2000.0), CameraSpec(45.0, 2900.0), CameraSpec(90.0, 2000.0)]
    result = apply_distance_to_all(cameras, source_angle_deg=45.0)
    assert rig_angles(result) == [0.0, 45.0, 90.0]  # углы не тронуты
    assert rig_to_distances(result) == {0.0: 2900.0, 45.0: 2900.0, 90.0: 2900.0}  # все = источник


def test_apply_distance_to_all_missing_source_raises():
    cameras = [CameraSpec(0.0, 2000.0)]
    with pytest.raises(ValueError, match="не найдена"):
        apply_distance_to_all(cameras, source_angle_deg=999.0)


def test_duplicate_angle_in_rig_raises_on_upsert():
    presets = load_camera_rig_presets()
    bad = CameraRigPreset(
        name="дубль углов", cameras=[CameraSpec(0.0, 2000.0), CameraSpec(0.0, 3000.0)]
    )
    with pytest.raises(ValueError, match="дублирующ"):
        upsert_preset(presets, bad)


def test_save_then_load_roundtrip(tmp_path):
    path = tmp_path / "camera_rig_presets.yaml"
    presets = load_camera_rig_presets(CAMERA_RIG_PRESETS_PATH)
    custom = CameraRigPreset(
        name="Мой риг",
        cameras=[CameraSpec(0.0, 2100.0), CameraSpec(120.0, 3300.0)],
        readonly=False,
    )
    updated = upsert_preset(presets, custom)
    save_camera_rig_presets(updated, path)

    reloaded = load_camera_rig_presets(path)
    mine = next(p for p in reloaded if p.name == "Мой риг")
    assert mine.readonly is False
    assert rig_to_distances(mine.cameras) == {0.0: 2100.0, 120.0: 3300.0}

    reloaded_without_mine = delete_preset(reloaded, "Мой риг")
    assert all(p.name != "Мой риг" for p in reloaded_without_mine)
