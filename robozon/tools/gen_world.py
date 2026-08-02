#!/usr/bin/env python3
"""Генерация мира Webots из config/layout.yaml.

Единственный источник геометрии участка — layout.yaml: и мир, и логика
supervisor-контроллера читают одни и те же числа. После правки конфига
перезапустить генерацию (scripts/run.sh делает это автоматически):

    python3 tools/gen_world.py
"""
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sim.config import load_layout, load_objects  # noqa: E402

WORLD_PATH = ROOT / "webots" / "worlds" / "sorting_line.wbt"


def paddle_node(name: str, dv: dict, pos: dict) -> str:
    """Поворотный шибер: стойка + щит на вертикальном шарнире.

    Ось — за краем ленты (anchor_y * side), щит в парковке лежит вдоль ленты
    (по +x от оси), в рабочем положении диагонально перекрывает ленту.
    Мотор принадлежит Robot-супервизору, в детей которого узел вставляется.
    """
    ax = pos["x"]
    ay = dv["anchor_y"] * pos["side"]
    length = dv["arm_length"]
    th = dv["arm_thickness"]
    h = dv["arm_height"]
    zb = dv["z_bottom"]
    tilt = dv["arm_tilt"] * pos["side"]   # верх щита нависает над объектом
    zc = zb + h / 2
    post_h = zb + h
    return f"""    Pose {{
      translation {ax} {ay} {post_h / 2}
      children [
        Shape {{
          appearance DEF POST_APP PBRAppearance {{ baseColor 0.30 0.32 0.36 roughness 0.7 metalness 0.5 }}
          geometry Cylinder {{ radius 0.05 height {post_h} }}
        }}
      ]
    }}
    HingeJoint {{
      jointParameters HingeJointParameters {{
        anchor {ax} {ay} 0
        axis 0 0 1
      }}
      device [
        RotationalMotor {{
          name "paddle_{name}"
          maxVelocity {dv["motor_velocity"]}
          maxTorque {dv.get("motor_torque", 120)}
          controlPID {" ".join(str(v) for v in dv.get("motor_control_pid", [10, 0, 0]))}
          sound ""
        }}
        PositionSensor {{
          name "paddle_{name}_sensor"
        }}
      ]
      endPoint Solid {{
        name "paddle {name}"
        contactMaterial "paddle"
        children [
          Pose {{
            translation {ax + length / 2} {ay} {zc}
            rotation 1 0 0 {tilt}
            children [
              Shape {{
                appearance PBRAppearance {{ baseColor 0.85 0.55 0.10 roughness 0.4 metalness 0.6 }}
                geometry Box {{ size {length} {th} {h} }}
              }}
            ]
          }}
        ]
        boundingObject Pose {{
          translation {ax + length / 2} {ay} {zc}
          rotation 1 0 0 {tilt}
          children [
            Box {{ size {length} {th} {h} }}
          ]
        }}
        physics Physics {{
          density -1
          mass 3
          centerOfMass [
            {ax + length / 2} {ay} {zc}
          ]
        }}
      }}
    }}"""


def mesh_templates_node(objects_cfg: dict, http_port: int) -> str:
    """Заготовки геометрии STL — по одной DEF-меш-ноде на тип товара.

    Спавн объекта (supervisor_main.py) далее делает `geometry USE MESH_<type>`
    вместо собственного `Mesh { url }`: STL скачивается и парсится (в т.ч.
    браузерным вьюером через assimpjs) один раз при загрузке мира, а не на
    каждый спавн — иначе видимая пауза/окно загрузки повторялись бы каждый
    раз. Держим ноды глубоко под полом — они нужны только как источник DEF.
    """
    shapes = "\n    ".join(
        f'Shape {{ castShadows FALSE geometry DEF MESH_{name} Mesh {{ url [ "http://127.0.0.1:{http_port}/meshes/{name}.stl" ] }} }}'
        for name in objects_cfg
    )
    return f"""DEF MESH_TEMPLATES Solid {{
  translation 0 0 -50
  name "mesh_templates"
  children [
    {shapes}
  ]
}}"""


def flat_conveyor_node(name: str, x_start: float, length: float, width: float,
                        height: float, speed: float, border_height: float = 0.0,
                        border_drop: float = 0.0) -> str:
    """Плоский горизонтальный сегмент ленты (SimpleConveyor вдоль X).

    border_height=0 на рабочем (высоком) сегменте — шиберы сталкивают товар
    за край в кейджи, бортик бы этому мешал. На сегменте появления бортик,
    наоборот, нужен: случайный наклон при спавне (см. supervisor_main.py)
    иногда даёт товару крен вбок, и без бортика он укатывается с ленты, не
    доехав до шибера (см. обсуждение — крупные товары вроде пуфика почти
    вровень с шириной ленты).

    border_drop — опускает сам бортик (не меняя border_height) так, чтобы он
    меньше выступал над лентой — нужно на зоне CV, где side-камера смотрит
    почти вдоль ленты и бортик перекрывал низ объекта в кадре (см. заметку
    задачи "Рассчитать геометрию 3-камерного рига и зеркала для CV").
    """
    xc = x_start + length / 2
    return f"""SimpleConveyor {{
  translation {xc} 0 0
  name "{name}"
  size {length} {width} {height}
  speed {speed}
  borderDrop {border_drop}
  borderHeight {border_height}
}}"""


def ramp_conveyor_node(name: str, x_start: float, length: float, width: float,
                        height_start: float, height_end: float, speed: float,
                        thickness: float = 0.15) -> str:
    """Наклонный сегмент ленты: Track, повёрнутый вокруг оси Y так, чтобы
    верх ленты шёл прямой линией от (x_start, height_start) до
    (x_start+length, height_end). Рама/бортики не рисуются (только сам
    приводной щит) — упрощение ради наклонного участка.
    """
    x_lo, x_hi = x_start, x_start + length
    z_lo, z_hi = height_start, height_end
    dx, dz = x_hi - x_lo, z_hi - z_lo
    true_length = math.hypot(dx, dz)
    theta = -math.atan2(dz, dx)   # знак подобран так, чтобы дальний конец (x_hi) поднимался
    mid_x, mid_z = (x_lo + x_hi) / 2, (z_lo + z_hi) / 2
    tx = mid_x - (thickness / 2) * math.sin(theta)
    tz = mid_z - (thickness / 2) * math.cos(theta)
    return f"""Robot {{
  translation {tx} 0 {tz}
  rotation 0 1 0 {theta}
  name "{name}"
  controller "belt_drive"
  controllerArgs [ "{speed}" ]
  children [
    Track {{
      translation 0 0 0
      children [
        Shape {{
          appearance PBRAppearance {{ baseColor 0.16 0.16 0.18 roughness 0.9 metalness 0 }}
          geometry DEF RAMP_BO Box {{ size {true_length} {width} {thickness} }}
        }}
      ]
      boundingObject USE RAMP_BO
      physics Physics {{ density -1 mass 1 }}
      device [
        LinearMotor {{
          name "belt_motor"
          maxVelocity {abs(speed) + 0.01}
          sound ""
        }}
      ]
    }}
  ]
}}"""


def belt_a_nodes(a: dict) -> str:
    """Три физических сегмента ленты A: плоский -> пандус -> плоский."""
    width, speed = a["width"], a["speed"]
    parts = []
    for i, seg in enumerate(a["segments"]):
        name = f"belt_a_{seg['name']}"
        if "height" in seg:
            parts.append(flat_conveyor_node(name, seg["x_start"], seg["length"],
                                             width, seg["height"], speed,
                                             seg.get("border_height", 0.0),
                                             seg.get("border_drop", 0.0)))
        else:
            parts.append(ramp_conveyor_node(name, seg["x_start"], seg["length"], width,
                                             seg["height_start"], seg["height_end"], speed))
    return "\n".join(parts)


def angled_wall_node(name: str, x0: float, y0: float, x1: float, y1: float,
                      thickness: float, z_lo: float, z_hi: float) -> str:
    """Прямая статическая стенка-направляющая между двумя точками (план: X,Y)."""
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)
    angle = math.atan2(dy, dx)
    mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2
    zc, h = (z_lo + z_hi) / 2, z_hi - z_lo
    return f"""Solid {{
  translation {mid_x} {mid_y} {zc}
  rotation 0 0 1 {angle}
  name "{name}"
  children [
    Shape {{
      appearance PBRAppearance {{ baseColor 0.42 0.46 0.52 roughness 0.6 metalness 0.4 }}
      geometry DEF {name.upper()}_BO Box {{ size {length} {thickness} {h} }}
    }}
  ]
  boundingObject USE {name.upper()}_BO
}}"""


def zone_b_chute_node(chute: dict, lane_y_end: float) -> str:
    """Статический наклонный раструб-лоток (без привода): пол опускается и
    сужается от высокого сегмента ленты A до входа в ленту B; боковые
    направляющие заданы отдельными прямыми стенками (angled_wall_node).
    Пол шире итоговой ширины ленты (без сужения по факту) — направляют
    именно боковые стенки, пол лишь исключает провал. lane_y_end — конец
    ленты B (правая стенка идёт прямой линией до этой точки).
    """
    y_lo, y_hi = chute["y_start"], chute["y_end"]
    z_lo, z_hi = chute["height_start"], chute["height_end"]
    x_right = chute["x_right"]
    w_start, w_end = chute["width_start"], chute["width_end"]
    thickness = 0.05

    dy, dz = y_hi - y_lo, z_hi - z_lo
    true_length = math.hypot(dy, dz)
    phi = math.atan2(dz, dy)
    mid_y, mid_z = (y_lo + y_hi) / 2, (z_lo + z_hi) / 2
    ty = mid_y + (thickness / 2) * math.sin(phi)
    tz = mid_z - (thickness / 2) * math.cos(phi)
    floor_width = w_start
    tx = x_right - floor_width / 2

    floor = f"""Solid {{
  translation {tx} {ty} {tz}
  rotation 1 0 0 {phi}
  name "zone_b_chute_floor"
  contactMaterial "chute"
  children [
    Shape {{
      appearance PBRAppearance {{ baseColor 0.16 0.16 0.18 roughness 0.9 metalness 0 }}
      geometry DEF CHUTE_FLOOR_BO Box {{ size {floor_width} {true_length} {thickness} }}
    }}
  ]
  boundingObject USE CHUTE_FLOOR_BO
}}"""

    z_wall_lo, z_wall_hi = 0.55, 0.90
    right_wall = angled_wall_node("wall_b_right", x_right, y_lo, x_right, lane_y_end,
                                   0.02, z_wall_lo, z_wall_hi)
    left_taper = angled_wall_node("wall_b_left_taper", x_right - w_start, y_lo,
                                   x_right - w_end, y_hi, 0.02, z_wall_lo, z_wall_hi)
    left_straight = angled_wall_node("wall_b_left_straight", x_right - w_end, y_hi,
                                      x_right - w_end, lane_y_end, 0.02, z_wall_lo, z_wall_hi)
    return "\n".join([floor, right_wall, left_taper, left_straight])


def belt_b_node(b: dict) -> str:
    """Приёмная лента зоны B — перпендикулярно ленте A (вдоль +Y)."""
    length = b["y_end"] - b["y_start"]
    yc = (b["y_start"] + b["y_end"]) / 2
    return f"""SimpleConveyor {{
  translation {b["x_center"]} {yc} 0
  rotation 0 0 1 1.5708
  name "belt_b"
  size {length} {b["width"]} {b["height"]}
  speed {b["speed"]}
  borderHeight 0
}}
DEF END_WALL_B Solid {{
  translation {b["x_center"]} {b["y_end"] + 0.07} {b["height"] + 0.25}
  name "end wall b"
  children [
    Shape {{
      appearance PBRAppearance {{ baseColor 0.42 0.46 0.52 roughness 0.6 metalness 0.4 }}
      geometry DEF WALL_B_BO Box {{ size {b["width"] + 0.2} 0.04 0.5 }}
    }}
  ]
  boundingObject USE WALL_B_BO
}}"""


def tilted_plate_node(name: str, x0: float, x1: float, y0: float, y1: float,
                       z0: float, z1: float, thickness: float) -> str:
    """Статическая пластина (плоская или наклонная только в плоскости Y-Z,
    X — ширина без наклона) без привода, скольжение — как пол zone_b_chute.
    """
    dy, dz = y1 - y0, z1 - z0
    true_length = math.hypot(dy, dz)
    phi = math.atan2(dz, dy)
    xc, yc, zc = (x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2
    ty = yc + (thickness / 2) * math.sin(phi)
    tz = zc - (thickness / 2) * math.cos(phi)
    return f"""Solid {{
  translation {xc} {ty} {tz}
  rotation 1 0 0 {phi}
  name "{name}"
  contactMaterial "chute"
  children [
    Shape {{
      appearance PBRAppearance {{ baseColor 0.16 0.16 0.18 roughness 0.9 metalness 0 }}
      geometry DEF {name.upper()}_BO Box {{ size {abs(x1 - x0)} {true_length} {thickness} }}
    }}
  ]
  boundingObject USE {name.upper()}_BO
}}"""


def d_entry_ramp_node(v: dict, cage: dict, pos: dict) -> str:
    """Козырёк-скат над 4-й (входной) стенкой кейджа D: один наклонный
    участок только над самой стенкой (не заходит вглубь бокса — иначе
    мешал бы вывозить кейдж). x-диапазон (по всей длине бокса, за вычетом
    боковых стенок) выводится из roll_cage/positions.D; наклон (y_end/z_end)
    задаётся в layout.yaml.
    """
    t = cage["wall_thickness"]
    x0 = pos["x"] - cage["length"] / 2 + t
    x1 = pos["x"] + cage["length"] / 2 - t

    return tilted_plate_node("d_entry_ramp", x0, x1,
                              v["y_start"], v["y_end"],
                              v["z_start"], v["z_end"], v["thickness"])


def roll_cage_node(name: str, cage: dict, pos: dict) -> str:
    closed = "TRUE" if pos.get("closed") else "FALSE"
    return f"""RollCage {{
  translation {pos["x"]} {pos["y"]} 0
  rotation 0 0 1 {pos["yaw"]}
  name "roll cage {name}"
  size {cage["width"]} {cage["length"]} {cage["height"]}
  wallThickness {cage["wall_thickness"]}
  closed {closed}
}}"""


def _camera_frame(theta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(forward, left, up) прямой камеры на азимуте θ (рад).

    ВАЖНО (найдено эмпирически 2026-07-19, реальным захватом кадра — camera
    "top" со старой формулой смотрела вдоль ленты, а не вниз): дефолтная
    ориентация Camera в Webots (нулевой rotation) — локальная ОСЬ X ВПЕРЁД,
    Z ВВЕРХ, Y ВЛЕВО (не "-Z вперёд, Y вверх, X вправо", как ошибочно
    предполагалось раньше и как физически работает типичный OpenGL-конвеншен).
    Проверено разбором уже работающей камеры "overview" (её рукописной
    axis-angle ориентации) — локальная +X даёт направление, совпадающее с
    направлением на сцену, остальные варианты — нет.

    `up` для ВСЕХ камер рига — константа (1,0,0) (мировая X, вдоль ленты) —
    так делает "вдоль ленты" вертикальной осью КАЖДОГО кадра одинаково,
    независимо от азимута (согласовано с конвенцией resolution: [строки
    (вдоль ленты), столбцы (поперёк)] в config/layout.yaml). `forward` —
    направление от камеры на точку наблюдения (объект): v_θ=(0,cosθ,sinθ) —
    позиция камеры, значит forward=-v_θ=(0,-cosθ,-sinθ). `left` достраивается
    из условия forward×left=up (стандартная право-ориентированная тройка)."""
    up = np.array([1.0, 0.0, 0.0])
    forward = np.array([0.0, -math.cos(theta), -math.sin(theta)])
    left = np.cross(up, forward)
    return forward, left, up


def _rotation_from_frame(forward: np.ndarray, left: np.ndarray, up: np.ndarray) -> tuple[np.ndarray, float]:
    """VRML ось+угол для тройки (forward,left,up), интерпретируемых как образы
    локальных осей (X=forward,Y=left,Z=up) — см. `_camera_frame`."""
    rmat = np.column_stack([forward, left, up])
    return _matrix_to_axis_angle(rmat)


def _mirror_normal_3d(phi_deg: float) -> np.ndarray:
    phi = math.radians(phi_deg)
    return np.array([0.0, math.sin(phi), -math.cos(phi)])


def _reflect_dir(v: np.ndarray, n: np.ndarray) -> np.ndarray:
    return v - 2.0 * np.dot(v, n) * n


def _matrix_to_axis_angle(rmat: np.ndarray) -> tuple[np.ndarray, float]:
    """Робастная (Shepperd) конвертация 3x3 матрицы поворота (det=+1) в
    ось+угол через кватернион — упрощённая формула (axis из антисимметричной
    части, angle=acos((tr-1)/2)) вырождается при angle≈180°, а для зеркальных
    камер этого рига это ВСЕГДА так (нормаль зеркала лежит в плоскости Y-Z без
    X-компоненты, right после реконструкции всегда точно противоположен
    исходному (1,0,0))."""
    m00, m01, m02 = rmat[0]
    m10, m11, m12 = rmat[1]
    m20, m21, m22 = rmat[2]
    tr = m00 + m11 + m22
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        qw, qx, qy, qz = s / 4, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2
        qw, qx, qy, qz = (m21 - m12) / s, s / 4, (m01 + m10) / s, (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2
        qw, qx, qy, qz = (m02 - m20) / s, (m01 + m10) / s, s / 4, (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2
        qw, qx, qy, qz = (m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, s / 4
    qw = max(-1.0, min(1.0, qw))
    angle = 2.0 * math.acos(qw)
    sin_half = math.sqrt(max(0.0, 1.0 - qw * qw))
    if sin_half < 1e-8:
        return np.array([1.0, 0.0, 0.0]), 0.0
    return np.array([qx, qy, qz]) / sin_half, angle


def _mirror_camera_rotation(source_angle_deg: float, own_angle_deg: float,
                             own_distance: float, rig: dict) -> tuple[np.ndarray, float]:
    """Ось+угол (VRML rotation) для зеркальной камеры, отражающей прямую камеру
    с азимутом `source_angle_deg`.

    Два НЕЗАВИСИМЫХ требования, раньше ошибочно слитых в одно:
    1. Прицел — камера должна смотреть точно на объект (тот же object-centered
       принцип, что у всех остальных камер рига). `forward` строится НАПРЯМУЮ
       как направление "объект минус позиция камеры" (позиция — d·v_θ при её
       СОБСТВЕННОМ true-azimuth `own_angle_deg`), а НЕ отражением forward
       реальной камеры — отражение НАПРАВЛЕНИЯ ≠ направление "на объект" из
       отражённой ПОЗИЦИИ при t≠0 (тот же класс ошибки, что 2φ-θ для азимута).
    2. Хиральность — отражение (несобственное преобразование) не сводится к
       повороту+переносу: картинка через зеркало физически зеркально
       перевёрнута относительно того, что увидела бы обычная камера в той же
       точке. `left` реальной камеры (см. `_camera_frame`) отражается
       (`reflect_dir`) и ортогонализуется к скорректированному forward
       (Грам-Шмидт) — это "лево" в отражённом мире; `up` (не `left`!)
       ВОССТАНАВЛИВАЕТСЯ как forward×left (гарантированно собственная, det=+1,
       тройка — rotation-поле Webots не может закодировать несобственное
       преобразование). Проверено численно: именно up получается
       противоположным истинному (1,0,0) — не left — поэтому компенсирующий
       flip кадра в `supervisor_main.py` ВЕРТИКАЛЬНЫЙ (не горизонтальный, как
       было бы в старом, неверном соглашении осей Webots)."""
    theta_source = math.radians(source_angle_deg)
    _, real_left, _ = _camera_frame(theta_source)
    n = _mirror_normal_3d(rig["mirror_normal_deg"])

    # Позиция камеры и объекта — в одной системе координат, СМЕЩЁННОЙ от точки
    # наблюдения (та же конвенция, что в cv_camera_node: translation = observation
    # + (0, d·cosθ, d·sinθ)); объект в этой системе — начало координат (0,0,0).
    theta_own = math.radians(own_angle_deg)
    virtual_pos_offset = np.array([0.0, own_distance * math.cos(theta_own),
                                    own_distance * math.sin(theta_own)])
    forward = -virtual_pos_offset
    forward /= np.linalg.norm(forward)

    left_raw = _reflect_dir(real_left, n)
    left = left_raw - np.dot(left_raw, forward) * forward
    left /= np.linalg.norm(left)

    up = np.cross(forward, left)
    up /= np.linalg.norm(up)

    return _rotation_from_frame(forward, left, up)


def cv_camera_node(name: str, cam: dict, rig: dict) -> str:
    """Узел Camera для CV-рига.

    Прямая камера: азимут θ вокруг оси X в плоскости Y-Z, v_θ=(0,cosθ,sinθ),
    камера в точке (x_cv, belt_z+d·cosθ, belt_z+d·sinθ), смотрит вдоль -v_θ —
    ориентация строится через `_camera_frame`/`_rotation_from_frame` (НЕ через
    простое "поворот на (θ-90°) вокруг X от дефолта" — та формула молчаливо
    предполагала дефолт Webots-камеры "-Z вперёд", что оказалось НЕВЕРНО, см.
    `_camera_frame`).

    Зеркальная камера (`cam["mirror"]`): та же ПОЗИЦИЯ (d·v_θ при true azimuth,
    см. config/layout.yaml: cv_rig и
    roboson_tools/docs/mirror_geometry_calc_3cam.md §2), но ОРИЕНТАЦИЯ строится
    отдельно — см. `_mirror_camera_rotation` (прицел на объект + правильная
    хиральность через reflect_dir, две независимые вещи). Захваченный кадр
    требует вертикальный flip в `supervisor_main.py`.
    """
    theta = math.radians(cam["angle"])
    if cam.get("mirror"):
        d = rig["mirror_distance_by_angle"][float(cam["angle"])]
    else:
        d = cam["distance"]
    x = rig["x"]
    y = d * math.cos(theta)
    z = rig["belt_z"] + d * math.sin(theta)
    if cam.get("mirror"):
        source_angle = rig["cameras"][cam["reflects"]]["angle"]
        axis, angle = _mirror_camera_rotation(source_angle, cam["angle"], d, rig)
    else:
        axis, angle = _rotation_from_frame(*_camera_frame(theta))
    rot_line = f"{axis[0]} {axis[1]} {axis[2]} {angle}"
    # per-camera "resolution"/"fov_deg" (опционально) переопределяют общее —
    # см. cv_rig.cameras.top_track в layout.yaml (выделенная низкоразрешающая
    # камера для непрерывного CV-триггера, заметка задачи "CV-триггер момента
    # по top-локализации..."). ВАЖНО: fov_deg ОБЯЗАН быть переопределён вместе
    # с n_cols — Webots привязывает fieldOfView к оси width(=n_cols), высота
    # выводится из отношения n_rows/n_cols (см. комментарий в layout.yaml) —
    # без согласованного fov_deg узкий n_cols не сужает реальный физический
    # обзор, только прореживает пиксели того же широкого кадра.
    n_rows, n_cols = cam.get("resolution", rig["resolution"])
    fov_deg_cam = cam.get("fov_deg", rig["fov_deg"])
    return f"""    Camera {{
      name "cv_{name}"
      translation {x} {y} {z}
      rotation {rot_line}
      width {n_cols}
      height {n_rows}
      fieldOfView {math.radians(fov_deg_cam)}
    }}"""


def cv_mirror_node(rig: dict) -> str:
    """Объект-зеркало для визуала — ПОЛАЯ РАМКА (4 тонких бруска по периметру),
    не сплошная пластина.

    Решение пользователя (2026-07-19, по итогам визуальной проверки реальных
    кадров): сплошная пластина физически стоит на луче между зеркальными
    камерами (они дальше от ленты, чем зеркало — см. mirror_distance_by_angle)
    и объектом, и перекрывает им обзор. Рамка (полая середина) даёт визуальный
    ориентир размера/положения зеркала, не блокируя ни камеры, ни будущую
    сегментацию. `mirror_size` — [длина вдоль X, толщина бруска, высота вдоль
    направления наклона] — теперь 1.6×0.02×0.8м (горизонтальная ориентация,
    было 0.6×0.02×1.4 — портретная, тоже находка пользователя).

    Позиция/ориентация — из ФИЗИЧЕСКОГО макета (не из mirror_normal_deg/
    mirror_offset — формула "ближайшая к точке наблюдения точка плоскости"
    работала для старой симметричной 2-камерной схемы, но для этой геометрии
    даёт точку ВНЕ видимого полотна, см. roboson_tools/docs/
    mirror_geometry_calc_3cam.md §3): нижний край `mirror_bottom_edge` (Y,Z) +
    направление вверх, заданное наклоном `mirror_tilt_from_vertical_deg` от
    вертикали со знаком `mirror_lean_sign` (в сторону от ленты).
    """
    tilt = math.radians(rig["mirror_tilt_from_vertical_deg"])
    lean = rig["mirror_lean_sign"]
    d_y = lean * math.sin(tilt)
    d_z = math.cos(tilt)
    bottom_y, bottom_z = rig["mirror_bottom_edge"]
    sx, st, sz = rig["mirror_size"]  # sz — высота вдоль направления наклона
    mid_y = bottom_y + 0.5 * sz * d_y
    mid_z = bottom_z + 0.5 * sz * d_z
    x = rig["x"]
    y = mid_y
    z = rig["belt_z"] + mid_z
    # Поворот вокруг X: локальная ось Z (0,0,1) должна совпасть с (d_y,d_z).
    rot_angle = math.atan2(-d_y, d_z)

    bar = rig.get("mirror_frame_bar_width", 0.03)  # ширина бруска рамки, м
    appearance = """PBRAppearance {
        baseColor 0.9 0.9 0.95
        roughness 0.05
        metalness 0.9
      }"""

    def bar_node(cx: float, cz: float, bx: float, bz: float) -> str:
        return f"""    Pose {{
      translation {cx} 0 {cz}
      children [
        Shape {{
          appearance {appearance}
          geometry Box {{ size {bx} {st} {bz} }}
        }}
      ]
    }}"""

    bars = "\n".join([
        bar_node(0.0, sz / 2 - bar / 2, sx, bar),           # верхний брусок
        bar_node(0.0, -sz / 2 + bar / 2, sx, bar),          # нижний брусок
        bar_node(-sx / 2 + bar / 2, 0.0, bar, sz - 2 * bar),  # левый брусок
        bar_node(sx / 2 - bar / 2, 0.0, bar, sz - 2 * bar),   # правый брусок
    ])
    return f"""Solid {{
  translation {x} {y} {z}
  rotation 1 0 0 {rot_angle}
  name "cv_mirror"
  children [
{bars}
  ]
}}"""


def main() -> None:
    lt = load_layout()
    objects_cfg = load_objects()
    a = lt["belt_a"]
    b = lt["belt_b"]
    chute = lt["zone_b_chute"]
    d_ramp = lt["d_entry_ramp"]
    dv = lt["diverters"]
    cage = lt["roll_cage"]
    wz = lt["work_zone"]
    phys = lt["physics"]
    cv = lt["cv_rig"]

    paddles = "\n".join(
        paddle_node(name, dv, pos) for name, pos in dv["positions"].items()
    )
    cages = "\n".join(
        roll_cage_node(name, cage, pos) for name, pos in cage["positions"].items()
    )
    cv_cameras = "\n".join(
        cv_camera_node(name, cam, cv) for name, cam in cv["cameras"].items()
    )
    cv_mirrors = cv_mirror_node(cv)

    w = f"""#VRML_SIM R2025a utf8
# Сгенерировано tools/gen_world.py из config/layout.yaml — руками не править.

EXTERNPROTO "../protos/SimpleConveyor.proto"
EXTERNPROTO "../protos/RollCage.proto"

WorldInfo {{
  title "Robozon: участок предварительной сортировки"
  basicTimeStep {phys["basic_time_step_ms"]}
  optimalThreadCount {min(os.cpu_count() or 1, 4)}
  contactProperties [
    ContactProperties {{
      coulombFriction [
        1.2
      ]
      rollingFriction 0.05 0.05 0.05
      bounce 0
      softCFM 0.0001
    }}
    ContactProperties {{
      material2 "paddle"
      coulombFriction [
        {phys["paddle_friction"]}
      ]
      rollingFriction 0.02 0.02 0.02
      bounce 0
      softCFM 0.0001
    }}
    ContactProperties {{
      material2 "chute"
      coulombFriction [
        0.02
      ]
      rollingFriction 0.0 0.0 0.0
      bounce 0
      softCFM 0.0001
    }}
    ContactProperties {{
      material2 "round"
      coulombFriction [
        1.2
      ]
      rollingFriction 0.02 0.02 0.02
      bounce 0
      softCFM 0.0001
    }}
    ContactProperties {{
      material2 "slippery"
      coulombFriction [
        1.2
      ]
      rollingFriction 0.15 0.15 0.15
      bounce 0
      softCFM 0.0001
    }}
  ]
}}
DEF VIEW Viewpoint {{
  orientation -0.310 0.388 0.868 1.491
  position 5 -4.5 4.5
}}
Background {{
  skyColor [
    0.65 0.72 0.80
  ]
  luminosity 0.6
}}
DirectionalLight {{
  direction 0.3 0.4 -1
  intensity 2.5
  castShadows TRUE
}}
DirectionalLight {{
  direction -0.5 -0.3 -1
  intensity 1.0
}}
DEF FLOOR Solid {{
  translation {wz["length_x"] / 2} 0 -0.05
  name "floor"
  children [
    Shape {{
      appearance PBRAppearance {{
        baseColor 0.55 0.57 0.60
        roughness 0.95
        metalness 0
      }}
      geometry DEF FLOOR_BO Box {{
        size {wz["length_x"] + 1.0} {wz["width_y"] + 1.0} 0.1
      }}
    }}
  ]
  boundingObject USE FLOOR_BO
}}
{mesh_templates_node(objects_cfg, lt["http"]["port"])}
{belt_a_nodes(a)}
{zone_b_chute_node(chute, b["y_end"])}
{belt_b_node(b)}
{cages}
{d_entry_ramp_node(d_ramp, cage, cage["positions"]["D"])}
{cv_mirrors}
DEF SORTER Robot {{
  translation 0 0 0
  name "sorter"
  controller "supervisor_main"
  supervisor TRUE
  children [
    Camera {{
      # Широкий план всей линии (спавн -> лента A -> кейдж C, вид на лоток B),
      # пересчитан 2026-07-20 под новую геометрию ([[Расчёт перекладки CV-зоны
      # и ramp]]) той же конвенцией forward/left/up -> axis-angle, что и cv_*
      # камеры (см. _camera_frame/_rotation_from_frame): look-at из (4.30,
      # -11.00, 9.00) в (4.30, -0.10, 0.80), up_ref = мировая Z.
      name "overview"
      translation 4.30 -11.00 9.00
      rotation -0.3021 0.3021 0.9041 1.6714
      width 1280
      height 720
    }}
{cv_cameras}
{paddles}
  ]
}}
"""
    WORLD_PATH.write_text(w, encoding="utf-8")
    print(f"OK: {WORLD_PATH}")


if __name__ == "__main__":
    main()
