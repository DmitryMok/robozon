#!/usr/bin/env python3
"""Автономная демо-сцена: модифицированный шибер с вращающимся отталкивающим
валом (нижняя часть щита) — проверка идеи "зазор 1см над лентой + вал с
лопастями не даёт мелкому/клиновидному объекту застрять или проскочить".

Мир/контроллеры отдельные (не sorting_line.wbt, ничего в основной сцене не
меняется), НО для проверки реальных товаров (не только клина/конуса) читает
ТЕ ЖЕ config/objects.yaml + assets/meshes + assets/collision, что и основная
линия (sim.config/sim.collision, только на чтение) — чтобы не пересчитывать
заново convex decomposition/плотность для тех же 11 STL.
Правьте константы ниже и перезапускайте:

    python3 tools/gen_demo_roller_shiber.py

Затем открыть webots/worlds/demo_roller_shiber.wbt напрямую в Webots (без
run.ps1/run.sh — контроллеры на Python Webots подхватывает сам).
"""
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORLD_PATH = ROOT / "webots" / "worlds" / "demo_roller_shiber.wbt"

sys.path.insert(0, str(ROOT))
from sim.collision import collision_bounding_node  # noqa: E402
from sim.config import load_layout, load_objects  # noqa: E402

REAL_OBJECTS = load_objects()
OBJECT_DENSITY = load_layout()["physics"]["object_density_kg_m3"]

# --- Лента -------------------------------------------------------------
BELT_LENGTH = 5.0
BELT_WIDTH = 0.5
BELT_HEIGHT = 0.82     # как в основной линии (belt_a, высокий сегмент)
BELT_SPEED = 1.0       # рабочая скорость

# --- Модифицированный шибер (щит + вал под ним) -------------------------
SHIBER_X = 2.6
ANCHOR_Y = 0.30        # |y| оси — за краем ленты, как anchor_y в layout.yaml
SIDE = 1.0             # сторона, с которой навешен шибер (+Y)
ARM_LENGTH = 1.15       # длина щита/вала, как в основной линии
ARM_TILT = 0.15         # наклон щита верхом на объект, как в основной линии
DEPLOY_ANGLE = 0.61     # 35° — рабочий (развёрнутый) угол, как в основной линии

ROLLER_GAP = 0.01           # ключевой параметр задачи: зазор лента-вал = 1см
ROLLER_SHAFT_RADIUS = 0.025  # радиус собственно вала (без лопастей)
BLADE_TIP_RADIUS = 0.045     # радиус до конца лопасти (вылет = 0.02м)
BLADE_COUNT = 4
BLADE_THICKNESS = 0.006
ROLLER_VELOCITY = -10.0      # рад/с — ЗНАК КРИТИЧЕН (найдено пользователем на
                             # живом прогоне на Windows): с +10 нижняя лопасть
                             # в точке контакта движется НАВСТРЕЧУ ленте —
                             # вместо отбрасывания объекта вбок-вперёд лопасть
                             # его ЗАЩЕМЛЯЕТ (затягивает в зазор роллер/панель).
                             # Отрицательная скорость — нижняя лопасть движется
                             # ПОПУТНО ленте (в направлении хода), с обгоном —
                             # отбрасывает объект вперёд-вбок, а не затягивает.
                             # ПРОВЕРЕНО эмпирически через панель (см. чат) —
                             # не выводилось из геометрии вручную (там легко
                             # ошибиться, как уже было с знаком DEPLOY_ANGLE).
                             #
                             # Главный риск задачи (см. чат): тонкая быстрая
                             # лопасть может "протуннелировать" сквозь тонкий
                             # объект за один шаг физики (особенно конус — он
                             # выступает в зону лопастей всего на 1мм). Если
                             # что-то ведёт себя необъяснимо — сначала крутить
                             # |ROLLER_VELOCITY| и/или basicTimeStep, а не
                             # считать идею проверенной.
ROLLER_MASS = 0.6
ROLLER_TORQUE = 40.0    # maxTorque мотора вала, Н·м. Webots-default (10) —
                        # вал клинило (велосервоприводу не хватало усилия
                        # провернуть лопасть сквозь тяжёлый/заклиненный объект)

PANEL_HEIGHT = 0.30     # часть щита НАД валом (визуально тот же шибер)
PANEL_THICKNESS = 0.03

# --- Тестовые объекты ----------------------------------------------------
# Клин (форма дверного стоппера): треугольная призма, вытянута по Y.
WEDGE_LENGTH = 0.080    # вдоль хода по ленте (X до поворота объекта)
WEDGE_WIDTH = 0.035
WEDGE_HEIGHT = 0.025    # высокий конец сзади, спереди сходит к нулю
WEDGE_MASS = 0.05

# Плоский конус — отдельная проверка (подозрение пользователя: низкий плоский
# профиль почти вровень с зазором может не зацепиться лопастями).
CONE_RADIUS = 0.020     # Ø40мм
CONE_HEIGHT = 0.011     # 11мм — на 1мм выше зазора 1см
CONE_MASS = 0.006

# Пресеты для веб-панели (demo_panel) — тот же набор случаев, что был в
# первом батч-прогоне (см. чат: wedge_b/wedge_c прошли, остальные 4 — нет).
# Оставлены как быстрые кнопки, а не автоспавн при загрузке мира — спавн
# теперь по команде с панели (см. webots/controllers/demo_panel).
SPAWN_X = 0.30           # x спавна по умолчанию (общий для всех живых спавнов)
PRESETS = [
    ("wedge", "клин, носом вперёд", 0.05, 0),
    ("wedge", "клин, боком", -0.05, 90),
    ("wedge", "клин, наискось", 0.10, 45),
    ("cone", "конус, центр", 0.00, 0),
    ("wedge", "клин, тупым концом", -0.10, 180),
    ("cone", "конус, край", 0.12, 0),
]

# SHIBER_X + ARM_LENGTH*cos(DEPLOY_ANGLE) ~= 3.54 — самая дальняя точка
# диагональной полосы щита/вала (см. roller_shiber_node). X_FINISH ДОЛЖЕН
# быть заметно дальше: живой прогон через веб-панель поймал баг — при
# X_FINISH=SHIBER_X+0.55=3.15 (внутри полосы!) исход иногда замораживался
# ДО того, как объект долетал до финального бокового смещения (клин ещё
# продолжал скользить вдоль щита за отметкой x=3.15) — PASS ошибочно
# читался как FAIL. 3.9 — с запасом дальше 3.54.
X_FINISH = 3.9
Y_SIDE_THRESHOLD = 0.12          # |y| смещения больше — считаем "вытолкнут вбок"


def wedge_ifs(length: float, width: float, height: float, color: str) -> str:
    """Клин = треугольная призма (выпуклая — годится и для boundingObject
    напрямую, без аппроксимации). Треугольное сечение в X-Z: высокий зад
    (x=0, z=height), низкий заострённый перед (x=length, z=0)."""
    hw = width / 2
    coord = (
        f"0 {-hw} 0, 0 {-hw} {height}, {length} {-hw} 0, "
        f"0 {hw} 0, 0 {hw} {height}, {length} {hw} 0"
    )
    # 0,1,2 = левая грань (back-bottom, back-top, tip); 3,4,5 = правая грань.
    # Порядок вершин проверен численно (cross(v1-v0,v2-v0) должен совпадать
    # с ожидаемой внешней нормалью) — исходный порядок был зеркальным на
    # всех 5 гранях (solid TRUE = backface culling, при неверном порядке
    # грани были бы не видны снаружи).
    coord_index = (
        "0 2 1 -1, "      # левая (нормаль -Y)
        "3 4 5 -1, "      # правая (нормаль +Y)
        "0 3 5 2 -1, "    # низ (z=0, нормаль -Z)
        "0 1 4 3 -1, "    # зад (x=0, нормаль -X)
        "1 2 5 4 -1"      # скошенная верхняя грань
    )
    return f"""IndexedFaceSet {{
          coord Coordinate {{ point [ {coord} ] }}
          coordIndex [ {coord_index} ]
          creaseAngle 0
        }}"""


def wedge_node(name: str, x: float, y: float, yaw_deg: float, side_sign: float) -> str:
    yaw = math.radians(yaw_deg)
    geom = wedge_ifs(WEDGE_LENGTH, WEDGE_WIDTH, WEDGE_HEIGHT, "")
    return f"""DEF {name.upper()} Solid {{
  translation {x} {y} {BELT_HEIGHT + 0.002}
  rotation 0 0 1 {yaw}
  name "{name}"
  children [
    Shape {{
      appearance PBRAppearance {{ baseColor 0.85 0.20 0.20 roughness 0.6 metalness 0.1 }}
      geometry {geom}
    }}
  ]
  boundingObject {geom}
  physics Physics {{ density -1 mass {WEDGE_MASS} }}
}}"""


def cone_ifs(radius: float, height: float, n_sides: int = 16) -> str:
    """Низкополигональный конус (выпуклый) — ТОЛЬКО для boundingObject: нативный
    Webots-примитив `Cone` не поддерживается как boundingObject (ODE не умеет
    конус, см. предупреждение при первом прогоне: "Cone geometry node cannot
    be used in bounding object" — визуально используем гладкий Cone, для
    физики отдельный выпуклый меш, чтобы сохранить именно скошенный борт
    конуса (не заменять его цилиндром — как раз скошенный борт и есть
    предмет проверки: соскользнёт ли конус под лопасть или зацепится)."""
    base = [
        f"{radius * math.cos(2 * math.pi * k / n_sides)} "
        f"{radius * math.sin(2 * math.pi * k / n_sides)} 0"
        for k in range(n_sides)
    ]
    coord = ", ".join(base) + f", 0 0 {height}"
    apex_idx = n_sides
    sides = ", ".join(
        f"{k} {(k + 1) % n_sides} {apex_idx} -1" for k in range(n_sides)
    )
    base_cap = " ".join(str(k) for k in ([0] + list(range(n_sides - 1, 0, -1))))
    return f"""IndexedFaceSet {{
          coord Coordinate {{ point [ {coord} ] }}
          coordIndex [ {sides}, {base_cap} -1 ]
          creaseAngle 0
        }}"""


def cone_node(name: str, x: float, y: float) -> str:
    bounding = cone_ifs(CONE_RADIUS, CONE_HEIGHT)
    return f"""DEF {name.upper()} Solid {{
  translation {x} {y} {BELT_HEIGHT + 0.002}
  name "{name}"
  children [
    Pose {{
      translation 0 0 {CONE_HEIGHT / 2}
      children [
        Shape {{
          appearance PBRAppearance {{ baseColor 0.20 0.55 0.85 roughness 0.5 metalness 0.1 }}
          geometry Cone {{ bottomRadius {CONE_RADIUS} height {CONE_HEIGHT} }}
        }}
      ]
    }}
  ]
  boundingObject {bounding}
  physics Physics {{ density -1 mass {CONE_MASS} }}
}}"""


# Простые примитивы (куб/шар) без STL — для демонстрации СКОРОСТИ ленты без
# нагрузки на рендер: реальные товары из assets/meshes/ визуально тяжёлые
# (напр. detergent.stl/pen.stl/helmet.stl — по размеру бинарного STL это
# десятки тысяч треугольников на визуал каждый), хотя их boundingObject и так
# простой примитив из config/objects.yaml (см. sim/collision.py) — на
# физику/скорость конвейера число треугольников визуала не влияет, но
# нагружает рендер GPU при показе в открытом окне Webots. box_node/sphere_node
# ниже — Box/Sphere и как Shape, и как boundingObject напрямую, три размера.
SIMPLE_SHAPE_SIZES = {"small": 0.05, "medium": 0.10, "large": 0.20}  # м (ребро/диаметр)
SIMPLE_SHAPES = (
    [(f"box_{k}", f"Куб {int(v * 1000)}мм") for k, v in SIMPLE_SHAPE_SIZES.items()]
    + [(f"sphere_{k}", f"Шар {int(v * 1000)}мм") for k, v in SIMPLE_SHAPE_SIZES.items()]
)


def simple_shape_node(name: str, kind: str, x: float, y: float) -> str:
    shape, size_key = kind.split("_", 1)
    size = SIMPLE_SHAPE_SIZES[size_key]
    if shape == "box":
        geom = f"Box {{ size {size} {size} {size} }}"
        color = "0.75 0.45 0.20"
        damping = ""
    else:
        geom = f"Sphere {{ radius {size / 2} subdivision 2 }}"
        color = "0.25 0.55 0.80"
        # без демпфирования шар на Track-ленте физически "взрывался" (улетал
        # на десятки/сотни метров за первые доли секунды после спавна) — без
        # плоской грани контакт шара с сегментами трака нестабилен, угловой
        # демпфер гасит раскрутку до того, как она успевает разойтись.
        damping = "damping Damping { linear 0.05 angular 0.3 }"
    z = BELT_HEIGHT + size / 2 + 0.002
    return f"""DEF {name.upper()} Solid {{
  translation {x} {y} {z}
  name "{name}"
  children [
    Shape {{
      castShadows FALSE
      appearance PBRAppearance {{ baseColor {color} roughness 0.6 metalness 0.1 }}
      geometry {geom}
    }}
  ]
  boundingObject {geom}
  physics Physics {{ density {OBJECT_DENSITY} {damping} }}
}}"""


def mesh_templates_node() -> str:
    """DEF MESH_<type> на каждый реальный товар из config/objects.yaml — тот же
    приём, что mesh_templates_node в основном gen_world.py (парсить STL один
    раз при загрузке мира, спавн дальше делает `USE MESH_<type>`), НО путь —
    обычный относительный файл (assets/meshes/, два уровня вверх от
    webots/worlds/), а не http://127.0.0.1 как в основной линии.

    Основная линия раздаёт STL по HTTP ради браузерного --stream просмотра
    (assimpjs фетчит через сеть, file:// там недоступен) — этот демо-мир
    открывается напрямую в Webots GUI, HTTP тут не нужен вообще. Пробовал
    HTTP-вариант (demo_panel отдавал файлы сам) — Webots пытается скачать
    Mesh при ЗАГРУЗКЕ МИРА, до того как контроллер-супервизор успевает
    поднять HTTP-сервер ("Connection refused" на всех 16 типов) — гонка
    заложена в порядок событий Webots (мир парсится раньше, чем стартуют
    контроллеры), не лечится изнутри одного контроллера. Локальный файл
    читается сразу, без сети — гонки нет в принципе.
    """
    shapes = "\n    ".join(
        f'Shape {{ castShadows FALSE geometry DEF MESH_{name} Mesh {{ '
        f'url [ "../../assets/meshes/{name}.stl" ] }} }}'
        for name in REAL_OBJECTS
    )
    return f"""DEF MESH_TEMPLATES Solid {{
  translation 0 0 -50
  name "mesh_templates"
  children [
    {shapes}
  ]
}}"""


def real_object_node(name: str, obj_type: str, x: float, y: float, yaw_deg: float) -> str:
    """Реальный товар из config/objects.yaml (та же геометрия/коллизия, что на
    основной линии — MESH_<type> + collision_bounding_node, см. модуль).
    Ротация — ТОЛЬКО вокруг Z (yaw, как у клина): ось Z инвариантна при
    повороте вокруг Z, поэтому origin-у-дна (см. tools/prepare_stl.py) не
    нужен доп. подъём под разные повороты, в отличие от произвольных 3D-
    поворотов в object_node/supervisor_main.py (там же spawn_clearance) —
    здесь всегда просто {BELT_HEIGHT}+зазор."""
    cfg = REAL_OBJECTS[obj_type]
    r, g, b = cfg["color"]
    bnd = cfg["bounding"]
    is_cylinder = bnd["type"] == "cylinder"
    is_roller = (is_cylinder and bnd.get("axis", "z") != "z") or cfg.get("roller", False)
    if cfg.get("slippery"):
        contact_material = "slippery"
    else:
        contact_material = "round" if is_roller else "default"
    lin_damp, ang_damp = (0.05, 0.01) if is_cylinder else (0.1, 0.05)
    yaw = math.radians(yaw_deg)
    return f"""DEF {name.upper()} Solid {{
  translation {x} {y} {BELT_HEIGHT + 0.002}
  rotation 0 0 1 {yaw}
  name "{name}"
  contactMaterial "{contact_material}"
  children [
    Shape {{
      castShadows FALSE
      appearance PBRAppearance {{ baseColor {r} {g} {b} roughness 0.7 metalness 0.1 }}
      geometry USE MESH_{obj_type}
    }}
  ]
  boundingObject {collision_bounding_node(obj_type)}
  physics Physics {{
    density {OBJECT_DENSITY}
    damping Damping {{ linear {lin_damp} angular {ang_damp} }}
  }}
}}"""


def blade_nodes(as_bounding: bool) -> str:
    """N лопастей вокруг вала — одна Pose на лопасть (translation+rotation в
    ОДНОМ узле, не вложенные Pose-в-Pose: Webots отклоняет вложенный Pose
    внутри boundingObject — "Cannot insert Pose node in 'children' field of
    Pose node in bounding object", найдено первым headless-прогоном, из-за
    этого лопасти реально не имели коллизии и все тестовые объекты
    проходили насквозь). Радиальное смещение (0, r_mid, 0) поворачивается
    вокруг X на `angle` ЗАРАНЕЕ, в Python (а не вложенным узлом) — оба узла
    (радиальный снос и поворот самой плиты) вокруг оси X, поэтому это ровно
    та же геометрия, что и раньше, но одним Pose."""
    r_mid = (ROLLER_SHAFT_RADIUS + BLADE_TIP_RADIUS) / 2
    span = BLADE_TIP_RADIUS - ROLLER_SHAFT_RADIUS
    box = f"Box {{ size {ARM_LENGTH} {span} {BLADE_THICKNESS} }}"
    blades = []
    for k in range(BLADE_COUNT):
        angle = k * 2 * math.pi / BLADE_COUNT
        ty, tz = r_mid * math.cos(angle), r_mid * math.sin(angle)
        inner = box if as_bounding else f"""Shape {{
              appearance PBRAppearance {{ baseColor 0.85 0.55 0.10 roughness 0.4 metalness 0.6 }}
              geometry {box}
            }}"""
        blades.append(f"""    Pose {{
      translation 0 {ty} {tz}
      rotation 1 0 0 {angle}
      children [ {inner} ]
    }}""")
    return "\n".join(blades)


def roller_shiber_node() -> str:
    """Модифицированный шибер: статичный (не качается) щит развёрнут в
    рабочее положение DEPLOY_ANGLE, под ним — вращающийся вал с лопастями
    на отдельном HingeJoint (ось X эндпойнта = длина щита). Зазор до ленты —
    ROLLER_GAP.

    Верхний Robot неподвижен (нет физики на нём самом — тот же приём, что
    у ramp_conveyor_node в основном gen_world.py), боковая панель — его
    собственный boundingObject (статичное препятствие), вал — единственное
    динамическое тело с приводом.
    """
    ax, ay = SHIBER_X, ANCHOR_Y * SIDE
    z_roller_axis = BELT_HEIGHT + ROLLER_GAP + BLADE_TIP_RADIUS
    roller_top_z = z_roller_axis + BLADE_TIP_RADIUS + 0.01  # зазор до панели —
    # без него панель касается верхней точки дуги лопасти (нулевой клиренс),
    # что даёт постоянный самоконтакт панель/лопасть (шум "Contact joints..."
    # в логе даже без тестовых объектов рядом) — найдено первым прогоном.
    panel_zc = roller_top_z + PANEL_HEIGHT / 2
    panel_zc_local = panel_zc - z_roller_axis
    # Щит наклонён (ARM_TILT) вокруг СВОЕГО центра, который стоит строго
    # НАД валом (сдвиг только по Z) — из-за наклона плоскость щита в высоте
    # вала (z=0) проходит не через y=0, а через y=panel_zc_local*tan(ARM_TILT).
    # Раньше ось вала оставалась в y=0, т.е. была выдвинута вперёд относительно
    # плоскости щита — на этом уступе некатящиеся объекты (напр. пуфик)
    # проваливались в щель между валом и щитом, а не гладко переходили с
    # одного на другой. Сдвигаем ось вала на этот же y, чтобы она лежала
    # ровно в плоскости щита.
    axis_y = panel_zc_local * math.tan(ARM_TILT)
    # Знак ПОДОБРАН ЭМПИРИЧЕСКИ (первый headless-прогон: с +DEPLOY_ANGLE весь
    # узел уезжал ЗА пределы ленты в сторону anchor_y, а не поперёк неё — все
    # тестовые объекты проходили насквозь просто потому что вал физически не
    # пересекал ленту). Основная линия берёт знак из runtime-логики
    # supervisor_main.py, которая тут не воспроизводится — для статичного
    # демо годится любой знак, который реально заметает ленту; проверено
    # trace-логом (см. чат).
    deploy = -DEPLOY_ANGLE * SIDE

    panel_pose = f"""translation {ARM_LENGTH / 2} 0 {panel_zc_local}
      rotation 1 0 0 {ARM_TILT}"""

    shaft_shape = f"""Pose {{
          rotation 0 1 0 1.5708
          children [
            Shape {{
              appearance PBRAppearance {{ baseColor 0.30 0.32 0.36 roughness 0.7 metalness 0.5 }}
              geometry Cylinder {{ radius {ROLLER_SHAFT_RADIUS} height {ARM_LENGTH} }}
            }}
          ]
        }}"""
    shaft_bounding = f"""Pose {{
          rotation 0 1 0 1.5708
          children [ Cylinder {{ radius {ROLLER_SHAFT_RADIUS} height {ARM_LENGTH} }} ]
        }}"""

    return f"""Robot {{
  translation {ax} {ay} {z_roller_axis}
  rotation 0 0 1 {deploy}
  name "roller_shiber_demo"
  controller "roller_spin"
  controllerArgs [ "{ROLLER_VELOCITY}" ]
  children [
    Pose {{
      {panel_pose}
      children [
        Shape {{
          appearance PBRAppearance {{ baseColor 0.85 0.55 0.10 roughness 0.4 metalness 0.6 }}
          geometry Box {{ size {ARM_LENGTH} {PANEL_THICKNESS} {PANEL_HEIGHT} }}
        }}
      ]
    }}
    HingeJoint {{
      jointParameters HingeJointParameters {{
        anchor {ARM_LENGTH / 2} {axis_y} 0
        axis 1 0 0
      }}
      device [
        RotationalMotor {{
          name "roller_motor"
          maxVelocity {abs(ROLLER_VELOCITY) + 2}
          maxTorque {ROLLER_TORQUE}
          sound ""
        }}
      ]
      endPoint Solid {{
        translation {ARM_LENGTH / 2} {axis_y} 0
        name "roller shaft"
        contactMaterial "roller"
        children [
          {shaft_shape}
{blade_nodes(as_bounding=False)}
        ]
        boundingObject Group {{
          children [
            {shaft_bounding}
{blade_nodes(as_bounding=True)}
          ]
        }}
        physics Physics {{
          density -1
          mass {ROLLER_MASS}
        }}
      }}
    }}
  ]
  boundingObject Pose {{
    {panel_pose}
    children [ Box {{ size {ARM_LENGTH} {PANEL_THICKNESS} {PANEL_HEIGHT} }} ]
  }}
}}"""


def demo_panel_node() -> str:
    """Супервизор веб-панели: спавн по HTTP-команде (demo_panel.py читает
    wedge_node/cone_node из ЭТОГО ЖЕ модуля через sys.path — см. контроллер),
    без геометрии/устройств самого узла."""
    return """DEF DEMO_PANEL Robot {
  name "demo_panel"
  controller "demo_panel"
  supervisor TRUE
}"""


def main() -> None:
    belt_xc = BELT_LENGTH / 2
    world = f"""#VRML_SIM R2025a utf8
# Сгенерировано tools/gen_demo_roller_shiber.py — руками не править.
# Автономная демо-сцена (не часть основной линии sorting_line.wbt).

EXTERNPROTO "../protos/SimpleConveyor.proto"

WorldInfo {{
  title "Demo: модифицированный шибер (вал с лопастями)"
  basicTimeStep 4
  contactProperties [
    ContactProperties {{
      coulombFriction [ 1.2 ]
      rollingFriction 0.05 0.05 0.05
      bounce 0
      softCFM 0.0001
    }}
    ContactProperties {{
      material2 "roller"
      coulombFriction [ 1.0 ]
      rollingFriction 0.02 0.02 0.02
      bounce 0
      softCFM 0.0001
    }}
    ContactProperties {{
      material2 "round"
      coulombFriction [ 1.2 ]
      rollingFriction 0.02 0.02 0.02
      bounce 0
      softCFM 0.0001
    }}
    ContactProperties {{
      material2 "slippery"
      coulombFriction [ 1.2 ]
      rollingFriction 0.15 0.15 0.15
      bounce 0
      softCFM 0.0001
    }}
  ]
}}
DEF VIEW Viewpoint {{
  orientation -0.20 0.20 0.96 1.55
  position {SHIBER_X - 0.6} -1.6 1.7
}}
Background {{
  skyColor [ 0.65 0.72 0.80 ]
  luminosity 0.6
}}
DirectionalLight {{
  direction 0.3 0.4 -1
  intensity 2.5
  castShadows TRUE
}}
{mesh_templates_node()}
DEF FLOOR Solid {{
  translation {belt_xc} 0 -0.05
  name "floor"
  children [
    Shape {{
      appearance PBRAppearance {{ baseColor 0.55 0.57 0.60 roughness 0.95 metalness 0 }}
      geometry DEF FLOOR_BO Box {{ size {BELT_LENGTH + 1.0} 2.0 0.1 }}
    }}
  ]
  boundingObject USE FLOOR_BO
}}
SimpleConveyor {{
  translation {belt_xc} 0 0
  name "belt_demo"
  size {BELT_LENGTH} {BELT_WIDTH} {BELT_HEIGHT}
  speed {BELT_SPEED}
  borderHeight 0
}}
{roller_shiber_node()}
{demo_panel_node()}
"""
    WORLD_PATH.write_text(world, encoding="utf-8")
    print(f"OK: {WORLD_PATH}")


if __name__ == "__main__":
    main()
