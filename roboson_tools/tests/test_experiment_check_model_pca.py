"""Регрессионные тесты для `check_model_roll_pca` и `check_model_roll_f1` — PCA-критерий
круглой проекции.

Параллель к `test_experiment_check_model.py` (там тестируется `check_model_roll` —
перебор полусферы). Здесь — те же инварианты для PCA- и F1-методов, плюс специфичные:
  - гантель с круглым стержнем и квадратными торцами (мотивировавший задачу кейс) —
    PCA/F1 должны отбросить;
  - почти-куб 400×400×300 — PCA/F1 должны отбросить (перебор полусферы ошибается);
  - «Цилиндр» из assets — PCA/F1 должны отбросить (гранёный профиль, ложное
    срабатывание Visual Hull у перебора полусферы);
  - шлем — F1 должен «катится» (ослабленный критерий + диагонали покрывают косую ось
    качения; чистый PCA пропускает).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from roboson_tools.core.experiment import (
    Orientation,
    check_model_roll,
    check_model_roll_f1,
    check_model_roll_g4,
    check_model_roll_pca,
    ROLL_VERDICT_NOT_ROUND,
    ROLL_VERDICT_ROUND,
    ROLL_VERDICT_UNCERTAIN,
)
from roboson_tools.geometry.mesh_io import from_trimesh, load_stl

ASSETS_STL_DIR = Path(__file__).resolve().parents[1] / "assets" / "stl"
DUMBBELL_STL = Path("/tmp/dumbbell.stl")
VIEW_ANGLES = [0, 45, 90, 135]


def _cylinder():
    return from_trimesh(trimesh.creation.cylinder(radius=50.0, height=300.0))  # ось = Z


def _box():
    return from_trimesh(trimesh.creation.box(extents=(300.0, 100.0, 100.0)))


def _azim_dist(azim_deg: float, target_deg: float) -> float:
    d = abs((azim_deg - target_deg) % 180.0)
    return min(d, 180.0 - d)


def test_cylinder_along_x_rolls_pca():
    """PCA находит ось цилиндра (X) как главную ось инерции, k ≥ 0.8."""
    result = check_model_roll_pca(
        _cylinder(),
        Orientation(roll_deg=0, pitch_deg=90, yaw_deg=0),
        VIEW_ANGLES,
        resolution_px=512,
        num_slices=30,
    )
    assert result.found_round
    assert result.best is not None and result.best.k > 0.9
    assert _azim_dist(result.best.axis_azim_deg, 0.0) <= 10.0
    assert abs(result.best.axis_elev_deg) <= 10.0


def test_upright_cylinder_rolls_pca():
    """Вертикальный цилиндр: PCA-ось (вертикаль) даёт круглую проекцию → катится."""
    result = check_model_roll_pca(
        _cylinder(),
        Orientation(roll_deg=0, pitch_deg=0, yaw_deg=0),
        VIEW_ANGLES,
        resolution_px=512,
        num_slices=30,
    )
    assert result.found_round
    assert abs(result.best.axis_elev_deg - 90.0) <= 10.0


def test_box_does_not_roll_pca():
    """Коробка: ни одна из 3 PCA-осей не даёт k ≥ порога → не катится. В отличие от
    `test_box_roll_22_5_is_known_hull_artifact` для перебора полусферы, PCA-метод
    НЕ выдаёт ложное срабатывание на коробе, повёрнутом на 22.5° — потому что cos 22.5°
    достигается только на косой оси, которой у PCA нет (только 3 главные оси)."""
    for yaw in (0.0, 22.5, 45.0):
        result = check_model_roll_pca(
            _box(),
            Orientation(roll_deg=0, pitch_deg=0, yaw_deg=yaw),
            VIEW_ANGLES,
            resolution_px=512,
            num_slices=30,
        )
        assert not result.found_round, f"yaw={yaw}: PCA ошибочно признал коробку круглой"
        assert result.best is not None and result.best.k < 0.8


def test_bottle_rolls_pca():
    """Бутылка: PCA находит ось Y (азимут 90°), k > 0.9 — как и перебор полусферы."""
    mesh = load_stl(ASSETS_STL_DIR / "Бутылка.stl")
    result = check_model_roll_pca(
        mesh,
        Orientation(roll_deg=0, pitch_deg=0, yaw_deg=0),
        VIEW_ANGLES,
        resolution_px=512,
        num_slices=40,
    )
    assert result.found_round
    assert result.best.k > 0.9
    assert _azim_dist(result.best.axis_azim_deg, 90.0) <= 10.0
    assert abs(result.best.axis_elev_deg) <= 10.0


def test_pca_rejects_near_cube_400():
    """Почти-куб 400×400×300: перебор полусферы ошибается (k=0.826 на косой оси,
    см. docs/method.md), PCA корректно отбрасывает — оси инерции почти-куба близки к
    осям бокса, проекции вдоль них — прямоугольники, не круги."""
    mesh = load_stl(ASSETS_STL_DIR / "Короб 400х400х300.stl")
    result = check_model_roll_pca(
        mesh,
        Orientation(roll_deg=0, pitch_deg=0, yaw_deg=0),
        VIEW_ANGLES,
        resolution_px=512,
        num_slices=40,
    )
    assert not result.found_round, "PCA ошибочно признал почти-куб круглым"


def test_pca_rejects_granular_cylinder():
    """«Цилиндр» из assets — НЕ тело вращения (гранёный профиль, квадратные фланцы),
    перебор полусферы даёт ложное k=0.807 от инфляции Visual Hull 4 ракурсов; PCA
    корректно отбрасывает (k<0.8 на всех 3 осях)."""
    mesh = load_stl(ASSETS_STL_DIR / "Цилиндр.stl")
    result = check_model_roll_pca(
        mesh,
        Orientation(roll_deg=0, pitch_deg=0, yaw_deg=0),
        VIEW_ANGLES,
        resolution_px=512,
        num_slices=40,
    )
    assert not result.found_round, "PCA ошибочно признал «Цилиндр» круглым"


def test_pca_rejects_dumbbell_with_square_caps():
    """Синтетическая гантель (круглый стержень Ø40 + квадратные торцы 80×80) — НЕ
    катится физически (опирается на квадратные торцы). Мотивирующий кейс задачи."""
    assert DUMBBELL_STL.exists(), (
        f"Синтетическая гантель не сгенерирована: {DUMBBELL_STL}. "
        "Сгенерируйте её перед запуском тестов (см. bench_roll_methods.py)."
    )
    mesh = load_stl(DUMBBELL_STL)
    result = check_model_roll_pca(
        mesh,
        Orientation(roll_deg=0, pitch_deg=0, yaw_deg=0),
        VIEW_ANGLES,
        resolution_px=512,
        num_slices=40,
    )
    assert not result.found_round, "PCA ошибочно признал гантель с квадратными торцами круглой"


def test_pca_returns_three_axis_hits():
    """Структура PcaRollResult: axis_hits содержит ровно 3 записи (по одной на каждую
    главную ось инерции), у каждой есть k и passed. Регрессия на структуру результата."""
    result = check_model_roll_pca(
        _box(),
        Orientation(roll_deg=0, pitch_deg=0, yaw_deg=0),
        VIEW_ANGLES,
        resolution_px=512,
        num_slices=30,
    )
    assert len(result.axis_hits) == 3
    for h in result.axis_hits:
        assert 0.0 <= h.axis_azim_deg < 180.0
        assert -90.0 <= h.axis_elev_deg <= 90.0
        assert 0.0 <= h.k <= 1.0


def test_pca_faster_than_sphere_scan():
    """PCA должен быть быстрее перебора полусферы как минимум на порядок — это одна из
    целей метода (реальное время на конвейере). Замер по elapsed_seconds (от
    реконструкции формы до конца расчёта k). Допускаем margin 5×, чтобы тест не
    флапал от шума на CI, — реальное ускорение 80–130× по бенчмарку."""
    result_pca = check_model_roll_pca(
        _box(),
        Orientation(roll_deg=0, pitch_deg=0, yaw_deg=0),
        VIEW_ANGLES,
        resolution_px=512,
        num_slices=30,
    )
    result_scan = check_model_roll(
        _box(),
        Orientation(roll_deg=0, pitch_deg=0, yaw_deg=0),
        VIEW_ANGLES,
        resolution_px=512,
        num_slices=30,
        axis_step_deg=5.0,
    )
    assert result_scan.elapsed_seconds > result_pca.elapsed_seconds * 5.0, (
        f"PCA ({result_pca.elapsed_seconds*1000:.1f}мс) не быстрее перебора "
        f"({result_scan.elapsed_seconds*1000:.1f}мс) хотя бы в 5×"
    )


# ---------------------------------------------------------------------------
#   F1 — ослабленный radial_symmetry + диагонали PCA-осей + финальный k
#   (см. check_model_roll_f1). Ожидание: 11/11 совпадений с эталоном, включая шлем
#   (косая ось качения, который PCA пропускает) и без ложных срабатываний на
#   гантели с квадратными торцами / почти-кубе / гранёном цилиндре.
# ---------------------------------------------------------------------------


def test_f1_cylinder_along_x_rolls():
    result = check_model_roll_f1(
        _cylinder(),
        Orientation(roll_deg=0, pitch_deg=90, yaw_deg=0),
        VIEW_ANGLES,
        resolution_px=512,
        num_slices=30,
    )
    assert result.found_round
    assert result.best is not None and result.best.k > 0.9
    assert _azim_dist(result.best.axis_azim_deg, 0.0) <= 10.0
    assert abs(result.best.axis_elev_deg) <= 10.0


def test_f1_upright_cylinder_rolls():
    result = check_model_roll_f1(
        _cylinder(),
        Orientation(roll_deg=0, pitch_deg=0, yaw_deg=0),
        VIEW_ANGLES,
        resolution_px=512,
        num_slices=30,
    )
    assert result.found_round


def test_f1_box_does_not_roll():
    for yaw in (0.0, 22.5, 45.0):
        result = check_model_roll_f1(
            _box(),
            Orientation(roll_deg=0, pitch_deg=0, yaw_deg=yaw),
            VIEW_ANGLES,
            resolution_px=512,
            num_slices=30,
        )
        assert not result.found_round, f"yaw={yaw}: F1 ошибочно признал коробку круглой"


def test_f1_bottle_rolls():
    mesh = load_stl(ASSETS_STL_DIR / "Бутылка.stl")
    result = check_model_roll_f1(
        mesh,
        Orientation(roll_deg=0, pitch_deg=0, yaw_deg=0),
        VIEW_ANGLES,
        resolution_px=512,
        num_slices=40,
    )
    assert result.found_round
    assert result.best.k > 0.9
    assert _azim_dist(result.best.axis_azim_deg, 90.0) <= 10.0


def test_f1_helmet_rolls():
    """Шлем — ключевой кейс F1 (улучшение над чистым PCA): ось качения косая, ни одна
    PCA-ось её не даёт (k<0.8), но диагональ PCA0+PCA2 попадает с k≈0.858. Если F1
    пропускает шлем — метод потерял смысл."""
    mesh = load_stl(ASSETS_STL_DIR / "Шлем.stl")
    result = check_model_roll_f1(
        mesh,
        Orientation(roll_deg=0, pitch_deg=0, yaw_deg=0),
        VIEW_ANGLES,
        resolution_px=512,
        num_slices=40,
    )
    assert result.found_round, "F1 пропустил шлем — метод не сработал"
    assert result.best.k >= 0.8


def test_f1_rejects_near_cube_400():
    mesh = load_stl(ASSETS_STL_DIR / "Короб 400х400х300.stl")
    result = check_model_roll_f1(
        mesh,
        Orientation(roll_deg=0, pitch_deg=0, yaw_deg=0),
        VIEW_ANGLES,
        resolution_px=512,
        num_slices=40,
    )
    assert not result.found_round


def test_f1_rejects_granular_cylinder():
    mesh = load_stl(ASSETS_STL_DIR / "Цилиндр.stl")
    result = check_model_roll_f1(
        mesh,
        Orientation(roll_deg=0, pitch_deg=0, yaw_deg=0),
        VIEW_ANGLES,
        resolution_px=512,
        num_slices=40,
    )
    assert not result.found_round


def test_f1_rejects_dumbbell_with_square_caps():
    """Тот же кейс, что и для PCA — гантель с круглым стержнем и квадратными торцами.
    F1 с диагоналями не должен ошибаться: ослабленный критерий пропустит стержень
    (ratio≈1.0), но финальный k по проекции вдоль стержня = квадрат → отбросится."""
    assert DUMBBELL_STL.exists(), f"Синтетическая гантель не сгенерирована: {DUMBBELL_STL}"
    mesh = load_stl(DUMBBELL_STL)
    result = check_model_roll_f1(
        mesh,
        Orientation(roll_deg=0, pitch_deg=0, yaw_deg=0),
        VIEW_ANGLES,
        resolution_px=512,
        num_slices=40,
    )
    assert not result.found_round


def test_f1_returns_method_label():
    """`PcaRollResult.method` для F1 — "f1" (не "pca"), чтобы различать результаты."""
    result = check_model_roll_f1(
        _box(),
        Orientation(roll_deg=0, pitch_deg=0, yaw_deg=0),
        VIEW_ANGLES,
        resolution_px=512,
        num_slices=30,
    )
    assert result.method == "f1"


def test_f1_faster_than_half_second_budget():
    """Бюджет конвейера (см. постановка задачи): анализ <0.5 с, иначе товары поступают
    быстрее, чем обрабатываются. F1 на любом объекте из тестового набора должен
    укладываться. Допускаем 0.2 с с запасом на медленном CI. Реальное время 14–25 мс."""
    for stl in ["Бутылка.stl", "Шлем.stl", "Короб 400х400х300.stl"]:
        mesh = load_stl(ASSETS_STL_DIR / stl)
        result = check_model_roll_f1(
            mesh,
            Orientation(roll_deg=0, pitch_deg=0, yaw_deg=0),
            VIEW_ANGLES,
            resolution_px=512,
            num_slices=40,
        )
        assert result.elapsed_seconds < 0.2, (
            f"{stl}: F1 превысил 0.2 с ({result.elapsed_seconds*1000:.0f}мс)"
        )


# ---------------------------------------------------------------------------
#   G4 — F1 + локальный поиск для пограничных + трёхкатегорный вердикт
#   (см. check_model_roll_g4). Ключевое свойство: НИ ОДНОГО пропуска катающегося
#   объекта в основной сортировщик (вердикт "round" или "uncertain" для всех
#   круглых; "not_round" только для явно некруглых с k<low_threshold).
# ---------------------------------------------------------------------------


def test_g4_cylinder_rolls():
    result = check_model_roll_g4(
        _cylinder(),
        Orientation(roll_deg=0, pitch_deg=90, yaw_deg=0),
        VIEW_ANGLES,
        resolution_px=512,
        num_slices=30,
    )
    assert result.verdict == ROLL_VERDICT_ROUND
    assert result.found_round


def test_g4_box_not_round():
    """Коробка — явно не катится. Вердикт должен быть not_round ИЛИ uncertain (если
    k6 попал в зону неуверенности [0.65, 0.8)), но НИКОГДА не round. Это реализация
    логики «лучше отправить на перепроверку, чем пропустить катающийся» — коробка
    с k6 на границе зоны уходит оператору, что безопасно."""
    result = check_model_roll_g4(
        _box(),
        Orientation(roll_deg=0, pitch_deg=0, yaw_deg=0),
        VIEW_ANGLES,
        resolution_px=512,
        num_slices=30,
    )
    assert result.verdict in (ROLL_VERDICT_NOT_ROUND, ROLL_VERDICT_UNCERTAIN), (
        f"Коробка: недопустимый verdict={result.verdict}"
    )
    assert result.verdict != ROLL_VERDICT_ROUND


def test_g4_helmet_round():
    """Шлем — F1 находит k=0.858 на диагонали PCA0+PCA2 → вердикт "round" без fallback."""
    mesh = load_stl(ASSETS_STL_DIR / "Шлем.stl")
    result = check_model_roll_g4(
        mesh,
        Orientation(roll_deg=0, pitch_deg=0, yaw_deg=0),
        VIEW_ANGLES,
        resolution_px=512,
        num_slices=40,
    )
    assert result.verdict == ROLL_VERDICT_ROUND
    assert result.best is not None and result.best.k >= 0.8


def test_g4_cyl_skewed_false_positive_via_fallback():
    """cyl_skewed — ИЗВЕСТНЫЙ ложноположительный результат (переименован 2026-07-31):
    пользователь визуально осмотрел реальную форму и подтвердил, что объект физически
    НЕ катится (`robozon/config/objects.yaml` уже давно правильно относит его к
    `category: ok`, не round — только этот тестовый набор был не обновлён). F1 даёт
    k=0.691 (зона неуверенности), локальный поиск вокруг лучшего из 6 базовых
    направлений цепляется за диагональный клин и находит k=0.8 → "round" — это
    артефакт локального поиска (тот же класс ошибки, что и куб по пространственной
    диагонали), НЕ настоящая ось переката. Тест зафиксирован как регрессия ТЕКУЩЕГО
    поведения (не как желаемое) — если оно изменится, значит алгоритм стал точнее и
    тест нужно обновить/удалить, см. docs/method.md."""
    assert (ASSETS_STL_DIR / "cyl_skewed.stl").exists(), "cyl_skewed.stl отсутствует"
    mesh = load_stl(ASSETS_STL_DIR / "cyl_skewed.stl")
    result = check_model_roll_g4(
        mesh,
        Orientation(roll_deg=0, pitch_deg=0, yaw_deg=0),
        VIEW_ANGLES,
        resolution_px=512,
        num_slices=40,
    )
    assert result.fallback_triggered, "G4 не запустил fallback для cyl_skewed (k6 в зоне [0.65, 0.8))"
    assert result.verdict == ROLL_VERDICT_ROUND, (
        f"известное текущее (ложноположительное) поведение изменилось: verdict={result.verdict}, "
        f"k={result.best.k} — если это исправление, обновите docs/method.md и переименуйте тест"
    )


def test_g4_asym_barrel_uncertain_not_missed():
    """asym_barrel — пограничный кейс: F1 даёт k=0.769, локальный поиск не находит
    k≥0.8 (максимум 0.794). Вердикт должен быть "uncertain" — НЕ "not_round", чтобы
    объект не попал тихо в основной сортировщик. Это реализация логики пользователя
    «лучше ошибиться в сторону круглого»."""
    assert (ASSETS_STL_DIR / "asym_barrel.stl").exists(), "asym_barrel.stl отсутствует"
    mesh = load_stl(ASSETS_STL_DIR / "asym_barrel.stl")
    result = check_model_roll_g4(
        mesh,
        Orientation(roll_deg=0, pitch_deg=0, yaw_deg=0),
        VIEW_ANGLES,
        resolution_px=512,
        num_slices=40,
    )
    assert result.fallback_triggered
    assert result.verdict == ROLL_VERDICT_UNCERTAIN, (
        f"asym_barrel должен быть uncertain (не not_round!), verdict={result.verdict}"
    )


def test_g4_dumbbell_not_round_or_uncertain():
    """Гантель с квадратными торцами — НЕ катится. Но k=0.701 (в зоне неуверенности),
    поэтому G4 даёт "uncertain" → отправляется оператору, а не в основной сортировщик.
    Вердикт "round" здесь был бы критической ошибкой (ложное «катится»)."""
    assert DUMBBELL_STL.exists(), f"Синтетическая гантель не сгенерирована: {DUMBBELL_STL}"
    mesh = load_stl(DUMBBELL_STL)
    result = check_model_roll_g4(
        mesh,
        Orientation(roll_deg=0, pitch_deg=0, yaw_deg=0),
        VIEW_ANGLES,
        resolution_px=512,
        num_slices=40,
    )
    assert result.verdict in (ROLL_VERDICT_UNCERTAIN, ROLL_VERDICT_NOT_ROUND), (
        f"Гантель с квадратными торцами не должна получить verdict=round, "
        f"получено: {result.verdict}"
    )
    assert result.verdict != ROLL_VERDICT_ROUND


def test_g4_no_round_for_non_round_assets():
    """Ни один заведомо некруглый объект из assets не должен получить verdict="round"
    — это бы значило, что в основной сортировщик попадёт катающийся объект (наоборот,
    но всё равно плохо: ложное срабатывание нагружает перепроверку). Все некруглые
    должны быть либо not_round, либо uncertain."""
    non_round = ["Короб 300х200х200.stl", "Короб 400х400х300.stl", "Цилиндр.stl", "Ручка.stl"]
    for stl in non_round:
        mesh = load_stl(ASSETS_STL_DIR / stl)
        result = check_model_roll_g4(
            mesh,
            Orientation(roll_deg=0, pitch_deg=0, yaw_deg=0),
            VIEW_ANGLES,
            resolution_px=512,
            num_slices=40,
        )
        assert result.verdict != ROLL_VERDICT_ROUND, (
            f"{stl}: ложное verdict=round для некруглого объекта (k={result.best.k})"
        )


def test_g4_all_roll_assets_get_round_or_uncertain():
    """Все заведомо катящиеся объекты из assets должны получить verdict "round" или
    "uncertain" — НИКОГДА не "not_round". Это ключевое свойство G4 для логики
    «пропуск катающегося в основной сортировщик недопустим»."""
    roll_objects = ["Бутылка.stl", "Тарелка.stl", "Пуфик.stl", "Шлем.stl", "Мешок.stl"]
    for stl in roll_objects:
        mesh = load_stl(ASSETS_STL_DIR / stl)
        result = check_model_roll_g4(
            mesh,
            Orientation(roll_deg=0, pitch_deg=0, yaw_deg=0),
            VIEW_ANGLES,
            resolution_px=512,
            num_slices=40,
        )
        assert result.verdict in (ROLL_VERDICT_ROUND, ROLL_VERDICT_UNCERTAIN), (
            f"{stl}: пропуск катающегося объекта (verdict={result.verdict}, "
            f"k={result.best.k if result.best else None})"
        )


def test_g4_returns_method_label():
    result = check_model_roll_g4(
        _box(),
        Orientation(roll_deg=0, pitch_deg=0, yaw_deg=0),
        VIEW_ANGLES,
        resolution_px=512,
        num_slices=30,
    )
    assert result.method == "g4"


def test_g4_faster_than_half_second_budget():
    """Бюджет конвейера <0.5 с. G4 с локальным fallback должен укладываться даже на
    пограничных объектах (где запускается поиск по ~80 направлениям). Допускаем 0.4 с
    с запасом на медленном CI. Реальное время 17–150 мс."""
    for stl in ["Бутылка.stl", "Шлем.stl", "Короб 400х400х300.stl", "asym_barrel.stl", "cyl_skewed.stl"]:
        mesh = load_stl(ASSETS_STL_DIR / stl)
        result = check_model_roll_g4(
            mesh,
            Orientation(roll_deg=0, pitch_deg=0, yaw_deg=0),
            VIEW_ANGLES,
            resolution_px=512,
            num_slices=40,
        )
        assert result.elapsed_seconds < 0.4, (
            f"{stl}: G4 превысил 0.4 с ({result.elapsed_seconds*1000:.0f}мс)"
        )