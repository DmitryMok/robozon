"""Headless CLI — интерфейс для LLM/скриптов: параметры на входе, JSON на выходе, без Qt."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..core.camera_rig import load_camera_rig_presets, rig_angles
from ..core.config import load_app_settings, load_objects
from ..core.experiment import (
    Orientation,
    check_model_roll,
    compare_configurations,
    evaluate,
    evaluation_to_dict,
    roll_check_to_dict,
    roundness_to_dict,
)
from ..export.exporter import export_experiment
from ..geometry.mesh_io import load_stl
from ..search.counterexample import search_counterexamples

# Наборы углов (без дистанций) для `compare` — сравнение конфигураций Analytical Mode, не
# завязанное на риг камер (см. gui/main_window.py::_COMPARE_ANGLE_PRESETS, тот же список).
_COMPARE_ANGLE_PRESETS: list[list[float]] = [
    [0.0, 90.0],
    [0.0, 45.0, 90.0, 135.0],
    [0.0, 30.0, 60.0, 90.0, 120.0, 150.0],
    [0.0, 22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5],
]


def _default_view_angles() -> list[float]:
    """Углы дефолтного (встроенного, первого) пресета рига камер — заменяет прежний
    settings.default_view_angles (см. config/camera_rig_presets.yaml)."""
    return rig_angles(load_camera_rig_presets()[0].cameras)


def _parse_angles(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def _parse_angle_sets(value: str) -> list[list[float]]:
    """'0,90;0,45,90,135' -> [[0,90],[0,45,90,135]]."""
    return [_parse_angles(part) for part in value.split(";") if part.strip()]


def _parse_range(value: str) -> tuple[float, float]:
    lo, hi = value.split(",")
    return float(lo), float(hi)


def resolve_stl(stl_arg: str) -> tuple[Path, bool | None]:
    """Аргумент --stl — либо ключ из config/objects.yaml, либо путь к произвольному STL.
    Возвращает (путь, is_round_reference); is_round_reference — None для произвольного пути
    (объект не описан в конфиге, признак неизвестен)."""
    objects = load_objects()
    if stl_arg in objects:
        entry = objects[stl_arg]
        return entry.path, entry.is_round_reference
    path = Path(stl_arg)
    if not path.exists():
        raise FileNotFoundError(
            f"'{stl_arg}' — не найден ни ключ в config/objects.yaml, ни файл по такому пути"
        )
    return path, None


def cmd_evaluate(args: argparse.Namespace) -> dict:
    settings = load_app_settings()
    stl_path, _ = resolve_stl(args.stl)
    mesh = load_stl(stl_path)

    orientation = Orientation(roll_deg=args.roll, pitch_deg=args.pitch, yaw_deg=args.yaw)
    angles = _parse_angles(args.angles) if args.angles else _default_view_angles()
    resolution = args.resolution or settings.raster_resolution_px
    threshold = args.threshold if args.threshold is not None else settings.roundness_threshold

    result = evaluate(
        mesh,
        orientation,
        angles,
        axis_pos=args.axis_pos,
        resolution_px=resolution,
        roundness_threshold=threshold,
    )
    return evaluation_to_dict(result)


def cmd_check(args: argparse.Namespace) -> dict:
    """"Проверить модель" headless: габариты по силуэтам + потенциал к перекату
    (круглая проекция, см. core/experiment.check_model_roll и docs/method.md)."""
    settings = load_app_settings()
    stl_path, is_round_reference = resolve_stl(args.stl)
    mesh = load_stl(stl_path)

    orientation = Orientation(roll_deg=args.roll, pitch_deg=args.pitch, yaw_deg=args.yaw)
    angles = _parse_angles(args.angles) if args.angles else _default_view_angles()
    resolution = args.resolution or settings.raster_resolution_px
    threshold = args.threshold if args.threshold is not None else settings.roundness_threshold

    result = check_model_roll(
        mesh,
        orientation,
        angles,
        resolution_px=resolution,
        roundness_threshold=threshold,
        num_slices=args.slices or settings.model_check_samples,
        axis_step_deg=args.axis_step or settings.axis_step_deg,
    )
    return {
        "stl": args.stl,
        "is_round_reference": is_round_reference,
        "orientation": {"roll_deg": args.roll, "pitch_deg": args.pitch, "yaw_deg": args.yaw},
        "view_angles_deg": angles,
        **roll_check_to_dict(result),
    }


def cmd_search(args: argparse.Namespace) -> dict:
    settings = load_app_settings()
    stl_path, is_round_reference = resolve_stl(args.stl)
    mesh = load_stl(stl_path)

    angles = _parse_angles(args.angles) if args.angles else _default_view_angles()
    resolution = args.resolution or 512
    threshold = args.threshold if args.threshold is not None else settings.roundness_threshold

    hits = search_counterexamples(
        mesh,
        angles,
        step_deg=args.step,
        roundness_threshold=threshold,
        resolution_px=resolution,
        roll_range=_parse_range(args.roll_range),
        pitch_range=_parse_range(args.pitch_range),
        yaw_range=_parse_range(args.yaw_range),
        max_workers=args.workers,
    )

    result = {
        "stl": args.stl,
        "is_round_reference": is_round_reference,
        "view_angles_deg": angles,
        "step_deg": args.step,
        "threshold": threshold,
        "hits": [
            {"roll_deg": h.roll_deg, "pitch_deg": h.pitch_deg, "yaw_deg": h.yaw_deg, "k": h.k}
            for h in hits
        ],
    }
    if is_round_reference:
        result["warning"] = (
            "is_round_reference=true для этого объекта — найденные 'hits' не являются "
            "настоящими контрпримерами (объект действительно круглый)."
        )
    return result


def cmd_compare(args: argparse.Namespace) -> dict:
    settings = load_app_settings()
    stl_path, _ = resolve_stl(args.stl)
    mesh = load_stl(stl_path)

    orientation = Orientation(roll_deg=args.roll, pitch_deg=args.pitch, yaw_deg=args.yaw)
    angle_sets = _parse_angle_sets(args.angle_sets) if args.angle_sets else _COMPARE_ANGLE_PRESETS
    resolution = args.resolution or settings.raster_resolution_px
    threshold = args.threshold if args.threshold is not None else settings.roundness_threshold

    entries = compare_configurations(
        mesh,
        orientation,
        angle_sets,
        axis_pos=args.axis_pos,
        resolution_px=resolution,
        roundness_threshold=threshold,
    )

    return {
        "orientation": {"roll_deg": args.roll, "pitch_deg": args.pitch, "yaw_deg": args.yaw},
        "results": [
            {"view_angles_deg": e.view_angles_deg, "roundness": roundness_to_dict(e.roundness)}
            for e in entries
        ],
    }


def cmd_export(args: argparse.Namespace) -> dict:
    settings = load_app_settings()
    stl_path, _ = resolve_stl(args.stl)
    mesh = load_stl(stl_path)

    orientation = Orientation(roll_deg=args.roll, pitch_deg=args.pitch, yaw_deg=args.yaw)
    angles = _parse_angles(args.angles) if args.angles else _default_view_angles()
    resolution = args.resolution or settings.raster_resolution_px
    threshold = args.threshold if args.threshold is not None else settings.roundness_threshold

    result = evaluate(
        mesh,
        orientation,
        angles,
        axis_pos=args.axis_pos,
        resolution_px=resolution,
        roundness_threshold=threshold,
    )
    out_dir = export_experiment(stl_path, orientation, result, Path(args.out_dir))
    return {"exported_to": str(out_dir), **evaluation_to_dict(result)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="roboson_tools.cli")
    sub = parser.add_subparsers(dest="command", required=True)

    p_eval = sub.add_parser("evaluate", help="Рассчитать метрики для одной ориентации")
    p_eval.add_argument("--stl", required=True, help="Ключ из config/objects.yaml или путь к STL")
    p_eval.add_argument("--roll", type=float, default=0.0)
    p_eval.add_argument("--pitch", type=float, default=0.0)
    p_eval.add_argument("--yaw", type=float, default=0.0)
    p_eval.add_argument("--angles", type=str, default=None, help="Например: 0,45,90,135")
    p_eval.add_argument("--axis-pos", type=float, default=None)
    p_eval.add_argument("--resolution", type=int, default=None)
    p_eval.add_argument("--threshold", type=float, default=None)
    p_eval.add_argument(
        "--out", type=str, default=None, help="Путь для сохранения JSON (иначе — stdout)"
    )
    p_eval.set_defaults(func=cmd_evaluate)

    p_check = sub.add_parser(
        "check", help="Проверить модель: габариты по силуэтам + круглая проекция (перекат)"
    )
    p_check.add_argument("--stl", required=True, help="Ключ из config/objects.yaml или путь к STL")
    p_check.add_argument("--roll", type=float, default=0.0)
    p_check.add_argument("--pitch", type=float, default=0.0)
    p_check.add_argument("--yaw", type=float, default=0.0)
    p_check.add_argument("--angles", type=str, default=None, help="Например: 0,45,90,135")
    p_check.add_argument("--resolution", type=int, default=None)
    p_check.add_argument("--threshold", type=float, default=None)
    p_check.add_argument("--slices", type=int, default=None, help="Число сечений реконструкции")
    p_check.add_argument(
        "--axis-step", type=float, default=None, help="Шаг перебора направлений оси, градусы"
    )
    p_check.add_argument("--out", type=str, default=None)
    p_check.set_defaults(func=cmd_check)

    p_search = sub.add_parser("search", help="Поиск контрпримеров: брутфорс Roll/Pitch/Yaw")
    p_search.add_argument("--stl", required=True, help="Ключ из config/objects.yaml или путь к STL")
    p_search.add_argument("--angles", type=str, default=None, help="Например: 0,45,90,135")
    p_search.add_argument("--step", type=float, required=True, help="Шаг перебора, градусы")
    p_search.add_argument("--roll-range", type=str, default="0,360")
    p_search.add_argument("--pitch-range", type=str, default="0,360")
    p_search.add_argument("--yaw-range", type=str, default="0,360")
    p_search.add_argument("--resolution", type=int, default=None)
    p_search.add_argument("--threshold", type=float, default=None)
    p_search.add_argument("--workers", type=int, default=None, help="Число процессов (по умолчанию — все ядра)")
    p_search.add_argument("--out", type=str, default=None)
    p_search.set_defaults(func=cmd_search)

    p_compare = sub.add_parser("compare", help="Сравнение наборов view_angles для одной ориентации")
    p_compare.add_argument("--stl", required=True, help="Ключ из config/objects.yaml или путь к STL")
    p_compare.add_argument("--roll", type=float, default=0.0)
    p_compare.add_argument("--pitch", type=float, default=0.0)
    p_compare.add_argument("--yaw", type=float, default=0.0)
    p_compare.add_argument(
        "--angle-sets", type=str, default=None, help="Например: '0,90;0,45,90,135'"
    )
    p_compare.add_argument("--axis-pos", type=float, default=None)
    p_compare.add_argument("--resolution", type=int, default=None)
    p_compare.add_argument("--threshold", type=float, default=None)
    p_compare.add_argument("--out", type=str, default=None)
    p_compare.set_defaults(func=cmd_compare)

    p_export = sub.add_parser("export", help="Экспорт эксперимента в директорию")
    p_export.add_argument("--stl", required=True, help="Ключ из config/objects.yaml или путь к STL")
    p_export.add_argument("--roll", type=float, default=0.0)
    p_export.add_argument("--pitch", type=float, default=0.0)
    p_export.add_argument("--yaw", type=float, default=0.0)
    p_export.add_argument("--angles", type=str, default=None)
    p_export.add_argument("--axis-pos", type=float, default=None)
    p_export.add_argument("--resolution", type=int, default=None)
    p_export.add_argument("--threshold", type=float, default=None)
    p_export.add_argument("--out-dir", type=str, required=True)
    p_export.add_argument("--out", type=str, default=None, help="JSON-сводка (иначе — stdout)")
    p_export.set_defaults(func=cmd_export)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result_dict = args.func(args)

    output = json.dumps(result_dict, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(output, encoding="utf-8")
    else:
        print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
