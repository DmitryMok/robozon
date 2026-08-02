"""Расширенное исследование ошибки габаритов для зеркальных ракурсов (компактная версия).

Проверяет ключевые вопросы задачи:
1. Учёт разной длины оптического пути 45/135: точная пара vs заглушка 2600/2600 vs без зеркала.
2. Влияние camera_distance (масштаб рига) при рекомендованной паре t=-1500.
3. Влияние supersample.

Условия: FOV 68°, 2592x1944, belt start/center/end.
Сокращено: 2 объекта, 4 yaw (0/45/90/135), чтобы уложиться в ~5 мин.
"""
from __future__ import annotations
from pathlib import Path
from roboson_tools.core.experiment import Orientation, check_simple_camera_dims_v3, mesh_true_dims
from roboson_tools.geometry.mesh_io import load_stl

FOV_DEG = 68.0
RESOLUTION_PX = (2592, 1944)
TOP_ANGLE = 90.0
BELT_POSITIONS = ["start", "center", "end"]
YAWS = [0.0, 45.0, 90.0, 135.0]

ASSETS = Path(__file__).parent / "assets" / "stl"
OBJECTS = [
    ("Korob300", ASSETS / "Короб 300х200х200.stl"),
    ("Korob400", ASSETS / "Короб 400х400х300.stl"),
]

MIRROR_CONFIGS = [
    ("stub_2600_2600", {45.0: 2600.0, 135.0: 2600.0}),
    ("t=-800_exact",   {45.0: 2028.0, 135.0: 3532.0}),
    ("t=-1500_exact",  {45.0: 2900.0, 135.0: 4908.0}),
    ("t=-2500_exact",  {45.0: 4620.0, 135.0: 6890.0}),
    ("no_mirror",      None),
]


def max_abs_err_pct(dims, true_extents):
    d = sorted(dims, reverse=True)
    t = sorted(true_extents, reverse=True)
    return max(abs(100.0 * (a - b) / b) for a, b in zip(d, t))


def run_one(mesh, true_dims, distances, camera_distance, supersample):
    errs = []
    for yaw in YAWS:
        ori = Orientation(roll_deg=0.0, pitch_deg=0.0, yaw_deg=yaw)
        side_angles = [0.0] if distances is None else [0.0, 45.0, 135.0]
        r = check_simple_camera_dims_v3(
            mesh, ori, FOV_DEG, camera_distance, RESOLUTION_PX,
            top_angle_deg=TOP_ANGLE, side_angles_deg=side_angles,
            belt_positions=BELT_POSITIONS, camera_distances=distances,
            supersample=supersample,
        )
        errs.append(float("nan") if r is None
                    else max_abs_err_pct([r.length, r.width, r.height], true_dims))
    valid = [e for e in errs if e == e]
    return (max(valid) if valid else float("nan"),
            sum(valid) / len(valid) if valid else float("nan"),
            errs)


def section(label):
    print("\n" + "=" * 100)
    print(label)
    print("=" * 100)


def main():
    section("1. Влияние учёта разной длины оптического пути (cam_dist=2000, ss=3)")
    print(f"{'Объект':<10} {'Конфиг':<18} {'max%':>6} {'mean%':>7}  err% по yaw(0/45/90/135)")
    print("-" * 100)
    for name, path in OBJECTS:
        mesh = load_stl(path)
        td = mesh_true_dims(mesh, axis_step_deg=5.0)
        for label, distances in MIRROR_CONFIGS:
            mx, mn, errs = run_one(mesh, td, distances, 2000.0, 3)
            by_yaw = " ".join(f"{e:5.1f}" if e == e else "  NaN" for e in errs)
            print(f"{name:<10} {label:<18} {mx:6.1f} {mn:7.1f}  {by_yaw}")

    section("2. Влияние camera_distance (t=-1500, ss=3)")
    print(f"{'Объект':<10} {'cam_dist':>9} {'max%':>6} {'mean%':>7}  err% по yaw(0/45/90/135)")
    print("-" * 100)
    distances_t1500 = {45.0: 2900.0, 135.0: 4908.0}
    for name, path in OBJECTS:
        mesh = load_stl(path)
        td = mesh_true_dims(mesh, axis_step_deg=5.0)
        for cam_d in [1500.0, 2000.0, 3000.0]:
            scale = cam_d / 2000.0
            distances = {a: d * scale for a, d in distances_t1500.items()}
            mx, mn, errs = run_one(mesh, td, distances, cam_d, 3)
            by_yaw = " ".join(f"{e:5.1f}" if e == e else "  NaN" for e in errs)
            print(f"{name:<10} {cam_d:>9.0f} {mx:6.1f} {mn:7.1f}  {by_yaw}")

    section("3. Влияние supersample (cam_dist=2000, t=-1500)")
    print(f"{'Объект':<10} {'ss':>3} {'max%':>6} {'mean%':>7}  err% по yaw(0/45/90/135)")
    print("-" * 100)
    for name, path in OBJECTS:
        mesh = load_stl(path)
        td = mesh_true_dims(mesh, axis_step_deg=5.0)
        for ss in [1, 3, 5]:
            mx, mn, errs = run_one(mesh, td, distances_t1500, 2000.0, ss)
            by_yaw = " ".join(f"{e:5.1f}" if e == e else "  NaN" for e in errs)
            print(f"{name:<10} {ss:>3} {mx:6.1f} {mn:7.1f}  {by_yaw}")


if __name__ == "__main__":
    main()