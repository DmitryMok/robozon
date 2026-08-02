#!/usr/bin/env python3
"""Автономная демо-сцена: шибер на входе в зону — идея пользователя: закрытое
положение щита перекрывает вход в зону (товар идёт транзитом по ленте),
открытое — освобождает проём и НАПРАВЛЯЕТ товар в зону, а обратный ход
(закрытие) дополнительно ДОТАЛКИВАЕТ товар внутрь, т.е. щит "загребает"
товар К СЕБЕ (к своей стороне), а не отбрасывает от себя.

ГЕОМЕТРИЯ (ВАЖНАЯ ПРАВКА 2026-07-29, после долгого разбора в чате — см.
историю): узел `flap_node_upstream` ниже — НЕ `tools.gen_world.paddle_node`
из проды (тот жёстко кладёт полотно от анкера ВНИЗ ПО ХОДУ ленты, +X). Для
такого полотна доказано (и алгеброй, и дважды вживую): объект при скольжении
ВСЕГДА уезжает ОТ анкера — физически невозможно заставить его тем же узлом
ехать К анкеру. `flap_node_upstream` — зеркальный по X узел: полотно от
анкера идёт НАВСТРЕЧУ ленте (-X, "против хода", туда же, откуда едет товар).
Для такого полотна знак меняется на противоположный: объект, продолжая
скользить вперёд по ленте, скользит уже К анкеру. Анкер стоит ПО ХОДУ ЛЕНТЫ
ПОСЛЕ метки зоны (справа от неё, если лента едет слева направо) — метка
внутри радиуса охвата полотна (в закрытом/флеш положении полотно физически
накрывает метку).

Углы (другая роль, чем в flap_node_upstream ~ paddle_node проды):
- ЗАКРЫТО = FLUSH (0.0) — полотно лежит вдоль кромки НАЗАД от анкера, физически
  перекрывая метку зоны (барьер именно за счёт совпадения по месту, не за счёт
  диагонали поперёк полосы).
- ОТКРЫТО = ANCHOR_SIDE*deploy_angle — полотно уходит по диагонали НАЗАД-и-
  ПОПЕРЁК (ещё смотрит "против хода", но и тянется через полосу) — рассекает
  полосу и, per доказанному правилу для этого узла, тянет объект К анкеру
  (= к метке).

УПРАВЛЕНИЕ — ТОЛЬКО РУЧНОЕ (по просьбе пользователя, автоцикл убран): щит и
веб-панель объединены в ОДИН Robot (supervisor + сам моторный узел), кнопки
"Открыть"/"Закрыть" на панели напрямую дёргают paddle_flap.setPosition(...).

Мир/контроллеры отдельные (не sorting_line.wbt, ничего в основной сцене не
меняется), но читает ТЕ ЖЕ config/objects.yaml + assets/meshes + config/layout.yaml
(diverters), что и основная линия (sim.config, только на чтение) — размеры
полотна/PID/скорость мотора те же, что в проде, меняется только форма узла.

Зона в этом демо АБСТРАКТНАЯ (без раструба/RollCage, по договорённости с
пользователем) — красная метка (zone_marker_node) — просто ориентир, не
физическое препятствие. Успех = объект ушёл на сторону ANCHOR_SIDE (=
TARGET_Y_SIGN, теперь СОВПАДАЕТ со стороной анкера — не противоположна ей,
как было в предыдущей версии узла).

Правьте константы ниже и перезапускайте:

    python3 tools/gen_demo_flap_shiber.py

Затем открыть webots/worlds/demo_flap_shiber.wbt напрямую в Webots (без
run.ps1/run.sh — контроллеры на Python Webots подхватывает сам), либо через
scripts/run_demo_flap_shiber.ps1.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORLD_PATH = ROOT / "webots" / "worlds" / "demo_flap_shiber.wbt"

sys.path.insert(0, str(ROOT))
from sim.config import load_layout  # noqa: E402
from tools.gen_demo_roller_shiber import (  # noqa: E402
    REAL_OBJECTS, SIMPLE_SHAPES, cone_node, mesh_templates_node,
    real_object_node, simple_shape_node, wedge_node,
)

# --- Лента -------------------------------------------------------------
BELT_LENGTH = 5.0
BELT_WIDTH = 0.5          # как belt_a.width в config/layout.yaml
BELT_HEIGHT = 0.82
BELT_SPEED = 1.0

# --- Бортик по дальнему (от щита) краю ленты ----------------------------
# SimpleConveyor.borderHeight ставит бортики СРАЗУ с обеих сторон (нельзя
# только одну) — щит должен оставаться полностью открытым на своей стороне
# (ANCHOR_SIDE, y<0 — иначе некуда будет выезжать объекту), поэтому здесь
# свой отдельный узел только на противоположной ("левой", +Y) кромке —
# сдерживает объект, если щит при открытии заденет ещё не отъехавший
# предыдущий (не даёт укатиться/улететь за пределы ленты в другую сторону).
# Геометрия/формула — те же, что в SimpleConveyor.proto (border left).
BORDER_HEIGHT = 0.04
BORDER_THICKNESS = 0.03

# --- Щит: размеры берём из основной линии (diverters), геометрия узла —
# СВОЯ (flap_node_upstream ниже), см. docstring модуля почему ---------------
_LAYOUT = load_layout()
_DV = _LAYOUT["diverters"]
_PADDLE_FRICTION = _LAYOUT["physics"]["paddle_friction"]

# Мотор для этого демо — СВОИ значения, не трогаем прод (config/layout.yaml).
#
# ИСТОРИЯ (2026-07-29): сначала пробовал velocity=10/torque=250/PID=[400,0,10]
# — при basicTimeStep=1 работает верно, но пользователь замерил вживую
# realtime-фактор — 0.7x ДАЖЕ БЕЗ спавна объектов (1мс = 1000 шагов физики/с,
# CPU не тянет в реальном времени; после спавна падает до 0.3-0.5x — отсюда
# рывки "равномерными замираниями"). Поднять basicTimeStep нельзя — на 2мс и
# 4мс (проверено живым прогоном) щит при velocity=10 бьёт объект В ДРУГУЮ
# СТОРОНУ (не просто грубее — качественно другой контакт: наконечник полотна
# при radius=1.15м и velocity=10 рад/с идёт ~11.5 м/с — за 2мс это 23мм за
# шаг, а клин всего 25-35мм в поперечнике, шаг почти вровень с размером
# объекта). Компромисс: снизил скорость мотора (11.5 -> ~6.9 м/с на
# наконечнике) — при более медленном полотне больший шаг физики не должен
# так же легко проскакивать сквозь объект, что даёт запас поднять
# basicTimeStep и вернуть realtime-фактор к 1.0x. Направление отклонения на
# этом сочетании (velocity=6 + basicTimeStep=2) — ПРОВЕРИТЬ headless-прогоном
# перед сдачей (см. чат), реальный realtime-фактор на машине пользователя
# не измерен мной (нет GUI) — нужно подтвердить у него.
_DV_MOTOR = dict(_DV, motor_velocity=6.0, motor_torque=180,
                  motor_control_pid=[50, 0, 0])
                  # PID-тюнинг (2026-07-29/30, живой прогон пользователя):
                  # [300,0,8] дрожь в покое; [300,0,35] дрожь+раскачка;
                  # [50,0,0] — подтверждено пользователем живым прогоном,
                  # держит покой без дрожи. P снижен настолько, что D/I не
                  # нужны вовсе — прежний P=300 был избыточно жёстким для
                  # этого узла/basicTimeStep=8.

ANCHOR_SIDE = -1           # на каком краю висит анкер (-1 -> y=-0.30). ТЕПЕРЬ
                            # СОВПАДАЕТ со стороной зоны (см. TARGET_Y_SIGN) —
                            # у flap_node_upstream (полотно "против хода")
                            # объект скользит К анкеру, не от него. Крутите
                            # этот параметр, если красная метка (zone_marker)
                            # оказалась не на той стороне.
GATE_X = 2.55               # было 3.0 — подобрано вживую (2026-07-29). X анкера —
                            # ПО ХОДУ ЛЕНТЫ ПОСЛЕ метки зоны (справа от неё, если
                            # лента едет слева направо).
GATE_POS = {"x": GATE_X, "side": ANCHOR_SIDE}
TARGET_Y_SIGN = ANCHOR_SIDE  # см. docstring: для узла "полотно против хода"
                              # объект тянет К стороне анкера (не от неё).

MARKER_X = GATE_X - 0.6     # метка зоны — ПЕРЕД анкером (против хода), внутри
                            # длины полотна (arm_length=1.15), чтобы в закрытом
                            # (флеш) положении полотно физически накрывало её.
MARKER_Y = _DV["anchor_y"] * ANCHOR_SIDE  # та же кромка, что и у анкера.

CLOSED_ANGLE = 0.0          # ФЛЕШ — полотно назад от анкера вдоль кромки,
                            # физически перекрывает MARKER_X/Y — вход в зону
                            # закрыт (см. docstring).
OPEN_ANGLE = ANCHOR_SIDE * _DV["deploy_angle"]  # полотно назад-и-поперёк —
                            # рассекает полосу, тянет объект к стороне анкера.

X_FINISH = round(GATE_X + 0.5, 2)  # запасной критерий FAIL: объект миновал
                            # щит по X, ни разу не свернув (PASS ловится по
                            # Y раньше и не ждёт X_FINISH — см. demo_panel_flap).
Y_SIDE_THRESHOLD = 0.12     # |y| дальше — считаем "ушёл в зону"

SPAWN_X = 0.30

# --- Лазерный (through-beam) датчик перед щитом --------------------------
# Излучатель на одном краю ленты, луч поперёк (вдоль Y) ко второму краю —
# как реальный оптический барьерный датчик: пока луч свободен, дистанция
# читается ~BEAM_MAX_RANGE (дальше ленты, целиться некуда); товар, пересекая
# луч, обрывает его на расстоянии << BEAM_MAX_RANGE — controller ловит
# именно этот перепад (фронт), не абсолютное значение.
SENSOR_X = 1.55                        # м — фиксированная позиция по ТЗ (2026-07-29)
BEAM_MAX_RANGE = 0.55                 # м — с запасом за BELT_WIDTH=0.5
BEAM_TRIGGER_DIST = 0.45              # м — ниже = луч перекрыт товаром

# знаки пресетов — от TARGET_Y_SIGN, не захардкожены (см. правку 2026-07-29).
PRESETS = [
    ("wedge", "клин, по центру", 0.00, 0),
    ("wedge", "клин, у кромки зоны", round(0.20 * TARGET_Y_SIGN, 2), 0),
    ("wedge", "клин, у дальней кромки", round(-0.20 * TARGET_Y_SIGN, 2), 0),
    ("cone", "конус, по центру", 0.00, 0),
]


def flap_node_upstream(name: str, dv: dict, pos: dict) -> str:
    """Щит с полотном "против хода" (навстречу товару) — ЗЕРКАЛЬНАЯ по X
    версия tools.gen_world.paddle_node (та кладёт полотно ВНИЗ по ходу, от
    анкера к +X; здесь — ВВЕРХ по ходу, от анкера к -X). Единственное
    геометрическое отличие от paddle_node — знак смещения полотна по X
    (`ax - length/2` вместо `ax + length/2`, и то же в boundingObject/
    centerOfMass); все размеры/материал/PID/мотор — те же поля из
    config/layout.yaml: diverters, что и в проде.

    ФИЗИКА (см. докстринг модуля): для этой ориентации объект, скользящий по
    полотну вперёд по ленте, движется К анкеру (не от него, как у paddle_node)
    — поэтому здесь можно "загребать к себе", а у продового узла — нельзя.

    ЗНАК tilt — ОБРАТНЫЙ относительно paddle_node (там `arm_tilt*side`).
    Наклон должен класть ВЕРХ полотна НАД товаром (прижимать к ленте, не
    давать забраться сверху) — у paddle_node это подобрано для полотна вниз
    по ходу; при том же знаке на нашем зеркальном (полотну вверх по ходу)
    наклон ложится в другую сторону — визуально полотно превращается в
    рампу/горку, а не в прижимающую стенку (найдено пользователем вживую
    2026-07-29). Знак минус компенсирует это зеркалирование.
    """
    ax = pos["x"]
    ay = dv["anchor_y"] * pos["side"]
    length = dv["arm_length"]
    th = dv["arm_thickness"]
    h = dv["arm_height"]
    zb = dv["z_bottom"]
    tilt = -dv["arm_tilt"] * pos["side"]
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
            translation {ax - length / 2} {ay} {zc}
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
          translation {ax - length / 2} {ay} {zc}
          rotation 1 0 0 {tilt}
          children [
            Box {{ size {length} {th} {h} }}
          ]
        }}
        physics Physics {{
          density -1
          mass 3
          centerOfMass [
            {ax - length / 2} {ay} {zc}
          ]
        }}
      }}
    }}"""


def beam_sensor_node() -> str:
    """Излучатель у края ленты (y=+BELT_WIDTH/2), луч вдоль -Y ко второму
    краю. Ось измерения DistanceSensor в Webots — локальный +X, поэтому
    поворот -90° вокруг Z кладёт его в мировое -Y. Без boundingObject/physics
    — луч не должен физически мешать товару."""
    y = BELT_WIDTH / 2
    z = BELT_HEIGHT + 0.01  # 10 мм над лентой (ТЗ 2026-07-29)
    return f"""    Pose {{
      translation {SENSOR_X} {y} {z}
      rotation 0 0 1 -1.5708
      children [
        DistanceSensor {{
          name "beam_sensor"
          lookupTable [ 0 0 0, {BEAM_MAX_RANGE} {BEAM_MAX_RANGE} 0 ]
          type "laser"
          aperture 0.02
          children [
            Shape {{
              appearance PBRAppearance {{ baseColor 0.90 0.10 0.10 emissiveColor 0.5 0.05 0.05 roughness 0.4 metalness 0.3 }}
              geometry Cylinder {{ radius 0.008 height 0.03 }}
            }}
          ]
        }}
      ]
    }}"""


def demo_panel_node() -> str:
    """ОДИН Robot: supervisor (спавн/статус, controller demo_panel_flap) И
    ЖЕ владелец мотора щита (flap_node_upstream) — так panel-контроллер может
    напрямую дёргать paddle_flap.setPosition(...) по кнопке из веб-панели,
    без отдельного автономного контроллера/таймера."""
    body = flap_node_upstream("flap", _DV_MOTOR, GATE_POS)
    sensor = beam_sensor_node()
    return f"""DEF DEMO_PANEL Robot {{
  name "demo_panel"
  controller "demo_panel_flap"
  supervisor TRUE
  children [
{body}
{sensor}
  ]
}}"""


def spawn_marker_node() -> str:
    """Чисто визуальная плоская метка на ленте в точке спавна (SPAWN_X) —
    без boundingObject (не мешает физике) — чтобы на глаз было видно, откуда
    и в какую сторону едут объекты, до того как они доедут до щита."""
    return f"""Solid {{
  translation {SPAWN_X} 0 {BELT_HEIGHT + 0.001}
  name "spawn_marker"
  children [
    Shape {{
      appearance PBRAppearance {{ baseColor 0.20 0.85 0.30 roughness 0.9 metalness 0 emissiveColor 0.05 0.25 0.08 }}
      geometry Box {{ size 0.03 {BELT_WIDTH} 0.001 }}
    }}
  ]
}}"""


def zone_marker_node() -> str:
    """Чисто визуальная КРАСНАЯ метка в точке MARKER_X/MARKER_Y — ПЕРЕД
    постом щита (против хода ленты), на кромке анкера. В закрытом (флеш)
    положении полотно физически лежит НАД этой меткой (см. docstring
    модуля). Цель — снять путаницу с "лево/право на экране": зелёная метка
    = откуда едут объекты, красная = куда их должен доставить щит. Если
    красная метка не на своей стороне — меняется ANCHOR_SIDE."""
    return f"""Solid {{
  translation {MARKER_X} {MARKER_Y} {BELT_HEIGHT + 0.001}
  name "zone_marker"
  children [
    Shape {{
      appearance PBRAppearance {{ baseColor 0.90 0.15 0.15 roughness 0.9 metalness 0 emissiveColor 0.35 0.03 0.03 }}
      geometry Box {{ size 0.35 0.35 0.001 }}
    }}
  ]
}}"""


def border_left_node(belt_xc: float) -> str:
    """Бортик по ЛЕВОЙ (+Y, дальней от щита) кромке ленты — см. константы
    BORDER_* выше. Физический (boundingObject) — сдерживает объект, а не
    только визуальный ориентир, в отличие от spawn/zone_marker."""
    y = 0.5 * BELT_WIDTH + 0.025
    z = BELT_HEIGHT + 0.5 * BORDER_HEIGHT - 0.03
    size_z = BORDER_HEIGHT + 0.06
    return f"""Solid {{
  translation {belt_xc} {y} {z}
  name "border_left"
  children [
    Shape {{
      appearance PBRAppearance {{ baseColor 0.42 0.46 0.52 roughness 0.6 metalness 0.4 }}
      geometry DEF BORDER_LEFT_BO Box {{ size {BELT_LENGTH} {BORDER_THICKNESS} {size_z} }}
    }}
  ]
  boundingObject USE BORDER_LEFT_BO
}}"""


def main() -> None:
    belt_xc = BELT_LENGTH / 2
    world = f"""#VRML_SIM R2025a utf8
# Сгенерировано tools/gen_demo_flap_shiber.py — руками не править.
# Автономная демо-сцена (не часть основной линии sorting_line.wbt).

EXTERNPROTO "../protos/SimpleConveyor.proto"

WorldInfo {{
  title "Demo: шибер на входе в зону (закрыто=барьер, открыто=проём)"
  basicTimeStep 8
  optimalThreadCount {min(os.cpu_count() or 1, 4)}
  contactProperties [
    ContactProperties {{
      coulombFriction [ 1.2 ]
      rollingFriction 0.05 0.05 0.05
      bounce 0
      softCFM 0.0001
    }}
    ContactProperties {{
      material2 "paddle"
      coulombFriction [ {_PADDLE_FRICTION} ]
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
  position {belt_xc - 1.0} -3.4 3.0
}}
Background {{
  skyColor [ 0.65 0.72 0.80 ]
  luminosity 0.6
}}
DirectionalLight {{
  direction 0.3 0.4 -1
  intensity 2.5
  castShadows FALSE
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
{spawn_marker_node()}
{zone_marker_node()}
{border_left_node(belt_xc)}
{demo_panel_node()}
"""
    WORLD_PATH.write_text(world, encoding="utf-8")
    print(f"OK: {WORLD_PATH}")
    print(f"CLOSED_ANGLE={CLOSED_ANGLE:.3f} OPEN_ANGLE={OPEN_ANGLE:.3f} "
          f"X_FINISH={X_FINISH} anchor_side={'y<0' if ANCHOR_SIDE < 0 else 'y>0'} "
          f"target_side={'y>0' if TARGET_Y_SIGN > 0 else 'y<0'} (same as anchor) "
          f"marker=({MARKER_X},{round(MARKER_Y,3)})")


if __name__ == "__main__":
    main()
