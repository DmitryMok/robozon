#!/usr/bin/env python3
"""Автономная демо-сцена «исполнительный механизм» (шаги 1-2 из 6, см. вики
`03 Work/План финализации сдачи (раздельные демо CV и исполнительного
механизма).md` + Kanban «Новая сцена демо исполнительного механизма»).

Отдельная от sorting_line.wbt сцена (не трогает основную линию/layout.yaml),
но геометрия описывается ОТНОСИТЕЛЬНО неё — числа ниже это разница с
`config/layout.yaml: belt_a` на момент шага 1 (2026-07-30):

- старт (низкий сегмент) — КАК В ПРОДЕ (3.15м, включая CV-зону на том же
  месте, x=2.55), но на 20см ДЛИННЕЕ (3.35м) — см. ПРАВКУ ниже: буфер нужен,
  чтобы подъём не попадал в кадр CV-камер.
- подъём — сразу после старта (та же геометрия рампы, что и в проде).
- финальный (высокий) сегмент — заканчивается на 60см ДАЛЬШЕ, чем текущая
  лента sorting_line.wbt (0.5+3.15+1.0+3.10=7.75 -> 8.35).
- CV-риг (3 камеры + зеркало) — ГЕОМЕТРИЯ И ПОЗИЦИЯ КАК В ПРОДЕ, без
  изменений (x=2.55, belt_z=0.70, на низком сегменте, ДО подъёма) — камеры
  ЧИСТО ДЕКОРАТИВНЫЕ, не подключены ни к какому контроллеру/CV-пайплайну.
- добавлена визуализация положения реальных камер (маленький корпус +
  цветная точка-«объектив» по направлению взгляда) — сами камеры (Camera
  node) невидимы в 3D-виде, без этого нельзя на глаз проверить раскладку.

ПРАВКА 2026-07-30 (по итогам первого визуального прогона пользователем):
первая версия шага 1 переносила CV-риг на высокий сегмент, ПОСЛЕ подъёма —
пользователь увидел в кадре камеры "top" сам подъём (наклонная лента в поле
зрения ломает фон для CV-алгоритмов, которые рассчитаны на ровный участок) и
попросил вернуть риг на низкий сегмент, как в проде, просто сделав сам
низкий сегмент на 20см длиннее (больше запаса перед подъёмом). Заодно
исправлен `camera_marker_node`: корпус-маркер раньше стоял ПРЯМО на позиции
камеры и наполовину торчал перед объективом — в кадре live-вида это было
видно как гигантский голубой объект, закрывающий почти весь кадр (маркер
теперь целиком СЗАДИ фокальной точки камеры, ничего не торчит в кадр).

Шаг 2 — шибер 1: узел `flap_node_upstream` (см. tools/gen_demo_flap_shiber.py)
— "загребание" вместо "плуга" `paddle_node` проды (объект тянется К анкеру, а
не отбрасывается к противоположному краю), анкер на ПРОТИВОПОЛОЖНОЙ от проды
стороне (там зона D: side=1, здесь ANCHOR1_SIDE=-1), ролл-кейдж стоит на ТОЙ
ЖЕ стороне, что анкер (не на противоположной, как у paddle_node-зон),
лучевой датчик перед щитом (тот же паттерн DistanceSensor, что
demo_flap_shiber.py), бортик на противоположной от щита кромке (с начала
высокого сегмента, сразу после подъёма, до конца ленты), козырёк-скат НЕ
нужен (загребание не даёт бокового удара под острым углом о кромку).

ПРАВКА 2026-07-30 (живой прогон): ролл-кейдж пробовали развернуть узкой
гранью к ленте (0.8м) — товар часто пролетал мимо (мouth не покрывал разброс
момента срабатывания щита). Вернули ШИРОКУЮ грань к ленте (yaw=1.5708, как
зона D в проде, 1.2м вдоль хода). Шибер 1 и сенсор сдвинуты на -0.1м ближе к
началу по тому же фидбеку.

Шаг 6 — визуальные ArUco-метки (декоративные, `tools/gen_aruco_textures.py`):
2 метки на полотне зеркала + 2 метки по бортам ленты в CV-зоне, ID совпадают с
первыми маркерами досок `roboson_tools/assets/{mirror,belt}_board/*.yaml`
(только для единообразия — solvePnP тут нет, координаты не связаны). Геометрия
метки — `IndexedFaceSet` с `solid FALSE` (двусторонний рендер), не `Plane` —
осознанно не выяснялась точная сторона (лицом к камерам или нет), см.
`_aruco_marker_shape`.

Правьте константы ниже и перезапускайте:

    python3 tools/gen_demo_mechanism.py

Открыть напрямую в Webots (webots/worlds/demo_mechanism.wbt) либо через
scripts/run_demo_mechanism.ps1 (обычный или -Headless режим, панель спавна —
см. webots/controllers/demo_panel_mechanism/).
"""
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORLD_PATH = ROOT / "webots" / "worlds" / "demo_mechanism.wbt"

sys.path.insert(0, str(ROOT))
from sim.collision import collision_bounding_node  # noqa: E402
from sim.config import load_layout, load_objects  # noqa: E402
from tools.gen_demo_flap_shiber import flap_node_upstream  # noqa: E402
from tools.gen_world import (  # noqa: E402
    _camera_frame, _rotation_from_frame, angled_wall_node, belt_b_node,
    cv_camera_node, cv_mirror_node, flat_conveyor_node, ramp_conveyor_node,
    roll_cage_node, tilted_plate_node,
)

_LAYOUT = load_layout()
REAL_OBJECTS = load_objects()
OBJECT_DENSITY = _LAYOUT["physics"]["object_density_kg_m3"]
PADDLE_FRICTION = _LAYOUT["physics"]["paddle_friction"]

# --- Реальная площадка (ТЗ 2026-07-30) ------------------------------------
# Декоративная плита-пол должна быть РАВНА реальной площадке (10x6м,
# config/layout.yaml: work_zone), а не подогнана по длине ленты "с запасом",
# как было раньше. Начало ленты (SEG_LOW_X_START) должно стоять на 1.0м от
# края этой плиты — плита начинается на 1.0м РАНЬШЕ начала ленты.
_WORK_ZONE = _LAYOUT["work_zone"]
PLATE_LENGTH_X = _WORK_ZONE["length_x"]   # 10.0
PLATE_WIDTH_Y = _WORK_ZONE["width_y"]     # 6.0

# --- Лента (шаг 1) -------------------------------------------------------
BELT_WIDTH = 0.5           # как belt_a.width в проде
BELT_SPEED = 1.0

SEG_LOW_X_START = 0.5
PLATE_X_START = round(SEG_LOW_X_START - 1.0, 3)   # левый край плиты — начало
                                                    # ленты на 1.0м от края
                                                    # (ТЗ 2026-07-30, см. выше)
PROD_SEG_LOW_LENGTH = 3.15  # длина низкого сегмента в проде (belt_a.segments.low)
SEG_LOW_LENGTH = round(PROD_SEG_LOW_LENGTH + 0.20, 3)  # ПРАВКА: +20см буфера перед подъёмом
                                                        # (см. докстринг модуля) — подъём не
                                                        # должен попадать в кадр CV-камер
SEG_LOW_HEIGHT = 0.70

SEG_RAMP_LENGTH = 1.0      # геометрия подъёма — та же, что в проде (belt_a.segments.ramp)
SEG_HIGH_HEIGHT = 0.82     # высота после подъёма — та же, что в проде

PROD_BELT_END_X = 7.75     # текущая лента sorting_line.wbt: 0.5+3.15+1.0+3.10
SEG_HIGH_END_X = round(PROD_BELT_END_X + 0.60 + 0.50, 3)   # ТЗ шаг 1: +60см дальше
                        # текущей, ПЛЮС ещё +50см (шаг 5, 2026-07-30): щит 2
                        # (после своего live-сдвига) кончается всего в 0.15м
                        # от исходного конца ленты — отражателю накопителя
                        # нужен свой пробег (0.5м), иначе он физически
                        # накладывался бы на щит 2 (см. ACCUM_DEFLECT_X_START)

SEG_LOW_END_X = round(SEG_LOW_X_START + SEG_LOW_LENGTH, 3)
SEG_RAMP_X_START = SEG_LOW_END_X
SEG_RAMP_END_X = round(SEG_RAMP_X_START + SEG_RAMP_LENGTH, 3)
SEG_HIGH_X_START = SEG_RAMP_END_X
SEG_HIGH_LENGTH = round(SEG_HIGH_END_X - SEG_HIGH_X_START, 3)

SPAWN_X = round(SEG_LOW_X_START + 0.25, 3)   # появление на низком сегменте, с запасом от края

# CV-риг — БЕЗ изменений относительно проды (x=2.55, belt_z=0.70, на низком
# сегменте, ДО подъёма) — см. ПРАВКУ в докстринге модуля: первая версия
# переносила риг на высокий сегмент, но подъём попадал в кадр камеры "top" и
# ломал бы фон для CV-алгоритмов. Раскладка (углы/дистанции/наклон зеркала)
# берётся из layout.yaml как есть, никакие поля не переопределяются.
CV_RIG = dict(_LAYOUT["cv_rig"])

# --- Шибер 1 (шаг 2) ------------------------------------------------------
# Узел щита — flap_node_upstream (см. tools/gen_demo_flap_shiber.py), НЕ
# tools.gen_world.paddle_node: тот кладёт полотно ВНИЗ по ходу и работает как
# "плуг" (объект скользит ОТ анкера, к противоположному от анкера краю ленты
# — так устроены D/B в проде). flap_node_upstream — зеркальная по X версия
# (полотно НАВСТРЕЧУ ленте), доказанно (и живым прогоном demo_flap_shiber)
# работает НАОБОРОТ: "загребание" — объект тянется К анкеру, закрытое (flush)
# положение перекрывает вход барьером, открытие тянет объект к своей же
# стороне. Это и есть "разворот на 180°" из ТЗ шага 2.
_DV = _LAYOUT["diverters"]
# Тюнинг мотора — ТОТ ЖЕ, что подобран живым прогоном в demo_flap_shiber.py
# под basicTimeStep=8 (см. историю в том файле) — НЕ дефолтные значения
# proды (velocity=4/torque=120/PID=[200,0,5], подобраны под paddle_node и
# другой сценарий срабатывания).
_DV_MOTOR = dict(_DV, motor_velocity=6.0, motor_torque=180, motor_control_pid=[50, 0, 0])

# Категория товара, на которую реагирует автосрабатывание (sensor_auto) этого
# шибера — та же карта категория->зона, что sim/router.py::CATEGORY_TO_ZONE
# в проде ("round"->D, "ok"->B, "oversize"->C): шибер 1 занимает место зоны D
# (округлые), шибер 2 — зоны B (ok), см. SHIBER2_CATEGORY ниже. Без этого
# деления оба щита при включённом автосрабатывании реагировали бы на ЛЮБОЙ
# товар без разбора категории.
SHIBER1_CATEGORY = "round"

SHIBER1_X = 5.85           # было 5.9 -> 6.0 — сдвинуто ещё на -0.05м (ТЗ
                            # 2026-07-30, второй раз), на высоком сегменте
                            # (4.85-8.35), с запасом вниз по ленте под шиберы
                            # 2/3+накопитель
ANCHOR1_SIDE = -1           # ПРОТИВОПОЛОЖНАЯ сторона относительно проды (там
                            # зона D: side=1) — ТЗ шага 2 "с противоположной
                            # стороны ленты по сравнению со старой сценой"
GATE1_POS = {"x": SHIBER1_X, "side": ANCHOR1_SIDE}
CLOSED1_ANGLE = 0.0                              # ФЛЕШ — вход перекрыт барьером
OPEN1_ANGLE = ANCHOR1_SIDE * _DV["deploy_angle"]  # тянет объект к стороне анкера

SENSOR1_X = round(SHIBER1_X - 1.0, 3)   # тот же офсет (-1.0м), что GATE_X-SENSOR_X
                                         # в demo_flap_shiber.py; при SHIBER1_X=5.85
                                         # даёт SENSOR1_X=4.85 = РОВНО SEG_HIGH_X_START
                                         # (конец подъёма) — ТЗ 2026-07-30: сенсор
                                         # должен срабатывать сразу на выходе с
                                         # подъёма, чтобы у длинных предметов передний
                                         # конец не оказывался НАД сенсором ещё на
                                         # наклонном участке (там другая высота/угол).
BEAM_MAX_RANGE = 0.55
BEAM_TRIGGER_DIST = 0.45

# Ролл-кейдж зоны шибера 1 — та же геометрия бокса, что RollCage в проде (см.
# config/layout.yaml: roll_cage). ИСТОРИЯ (2026-07-30): изначально по ТЗ стоял
# узкой стороной к ленте (yaw=0) — живым прогоном нашли, что товар часто
# пролетал МИМО (мouth 0.8м вдоль хода ленты не покрывал разброс момента
# срабатывания щита), развернули широкой гранью (yaw=1.5708, 1.2м). ПРАВКА
# 2026-07-30 (повторно): пользователь попросил вернуть узкую грань — ТАК ЖЕ,
# как у кейджа 2 (см. ниже) — для единообразия; если проблема с пролётом мимо
# вернётся при живом прогоне, это первое, что нужно будет проверить.
_CAGE = _LAYOUT["roll_cage"]
CAGE1_YAW = 0.0
CAGE1_X = 5.19   # было 5.3 (=SHIBER1_X-0.6, офсет из demo_flap_shiber.py) —
                 # сдвинуто живым прогоном (ТЗ 2026-07-30), больше НЕ формула
                 # от SHIBER1_X, а самостоятельно откалиброванное значение
CAGE1_Y = round(-(_DV["anchor_y"] + _CAGE["length"] / 2), 3)   # ближняя стенка
                                       # (SIDE_L при yaw=0) начинается точно
                                       # на линии анкера (та же формула, что
                                       # CAGE2_Y — узкая грань к ленте)
CAGE1_POS = {"x": CAGE1_X, "y": CAGE1_Y, "yaw": CAGE1_YAW, "closed": True}
# Козырёк-скат (d_entry_ramp в проде) НЕ нужен — ТЗ шага 2: "другой принцип
# работы столкнёт товар без проблем" (загребание не даёт того бокового удара
# под острым углом о кромку стенки, из-за которого он потребовался для D).

BORDER_HEIGHT = 0.04    # как в demo_flap_shiber.py
BORDER_THICKNESS = 0.03
BORDER1_Y_SIGN = -ANCHOR1_SIDE   # противоположная от анкера шибера 1 сторона
BORDER1_X_START = SEG_HIGH_X_START   # ПРАВКА 2026-07-30: было SHIBER1_X — начинался
                                      # только у самого щита, пользователь попросил
                                      # сразу после подъёма (защищает весь заход на
                                      # высокий сегмент, не только зону у щита)
# BORDER1_X_END задаётся НИЖЕ, после константы CHUTE2_X_LEFT (шаг 3): бортик
# шибера 1 не может идти "до конца ленты" буквально — теперь он физически
# перекрыл бы вход в зону B (обе конструкции на стороне +Y). См. правку.

# --- Шибер 2 / вход в зону B (шаг 3) ---------------------------------------
# ПРАВКА 2026-07-30 (живой прогон пользователя, дважды): (1) позиция зоны B
# ЖЁСТКО ФИКСИРОВАНА (реальный физический приёмник, не переносится вместе с
# шиберами) — X зоны B берётся НАПРЯМУЮ из проды (belt_b.x_center=7.250, см.
# config/layout.yaml), а НЕ вычисляется от SHIBER2_X. (2) анкер шибера 2
# должен стоять НА ТОЙ ЖЕ стороне, что и зона B (подтверждено пользователем),
# а не напротив, как у paddle_node-зон проды (D/B толкают ЧЕРЕЗ ленту, у
# них анкер и цель — на РАЗНЫХ сторонах). Раз анкер и цель — на ОДНОЙ
# стороне, узел щита — flap_node_upstream (ТА ЖЕ геометрия, что у шибера 1,
# см. tools/gen_demo_flap_shiber.py), НЕ paddle_node: для "анкер=сторона цели"
# физически работает только узел с полотном НАВСТРЕЧУ ленте (объект
# скользит К анкеру, где эта же сторона — сторона зоны B), paddle_node
# (полотно ВНИЗ по ходу) скользит ОТ анкера — тянуло бы товар прочь от зоны
# B, если анкер и цель совпадают. Тюнинг мотора — тот же _DV_MOTOR
# (velocity=6/torque=180/PID=[50,0,0]), что у шибера 1 — валиден для ЭТОЙ
# геометрии узла (подобран в demo_flap_shiber.py), а не тюнинг paddle_node.
SHIBER2_CATEGORY = "ok"   # зона B в проде (см. SHIBER1_CATEGORY выше)

_CHUTE = _LAYOUT["zone_b_chute"]
_BELT_B = _LAYOUT["belt_b"]
# ПРАВКА 2026-07-30 (уточнение реального ТЗ площадки): позиция зоны B задана
# АБСОЛЮТНО через реальные измерения площадки, а не "как в проде" (то было
# приближение до появления точных цифр) — центральная ось приёмной ленты B
# должна быть на 7.0м от начала ленты A (SEG_LOW_X_START), которое, в свою
# очередь, должно быть на 1.0м от края реальной площадки 10х6м (см. FLOOR
# ниже). Совпадение с продой (7.250) было случайным приближением (6.75м от
# начала при SEG_LOW_X_START=0.5) — теперь 7.5м, точно по ТЗ.
BELT_B_X_CENTER = round(SEG_LOW_X_START + 7.0, 3)
CHUTE2_X_LEFT = round(BELT_B_X_CENTER - _CHUTE["width_end"] / 2, 3)   # фиксированная
                                              # (прямая) стенка — левая граница
                                              # сузившегося (0.5м) конца лотка,
                                              # центр которого = BELT_B_X_CENTER
SHIBER2_X = round(CHUTE2_X_LEFT + 0.15 + 0.8, 3)   # было CHUTE2_X_LEFT+0.15
                                              # (зеркальное отражение офсета
                                              # проды — там щит стоял на
                                              # x_right-1.05, т.е. на 0.15м
                                              # РАНЬШЕ скошенного угла узкого
                                              # конца) — сдвинут ещё на +0.8м
                                              # живым прогоном (ТЗ 2026-07-30);
                                              # зона B (CHUTE2_X_LEFT/
                                              # BELT_B_X_CENTER) НЕ двигается,
                                              # сдвигается только сам щит
ANCHOR2_SIDE = 1            # ТА ЖЕ сторона, что зона B (+Y) — см. правку выше
GATE2_POS = {"x": SHIBER2_X, "side": ANCHOR2_SIDE}
CLOSED2_ANGLE = 0.0                              # ФЛЕШ — вход перекрыт барьером
OPEN2_ANGLE = ANCHOR2_SIDE * _DV["deploy_angle"]   # тянет объект к стороне анкера
                                                    # (та же формула, что у
                                                    # шибера 1 — OPEN_ANGLE в
                                                    # gen_demo_flap_shiber.py,
                                                    # БЕЗ минуса — это формула
                                                    # ДРУГОГО узла, чем
                                                    # supervisor_main.py:
                                                    # deploy=-side*angle
                                                    # у paddle_node)

SENSOR2_X = round(SHIBER2_X - 1.0, 3)   # тот же офсет перед щитом, что у шибера 1

# Вход в зону B — статический раструб-лоток (см. tools.gen_world.zone_b_chute_node),
# та же геометрия (ширины/высоты/y-диапазон), что zone_b_chute в проде
# (config/layout.yaml), НО ЗЕРКАЛЬНО: скос (широкий раструб -> сужение) — на
# ПРАВОЙ (высокий X, ниже по ходу ленты) стороне, прямая направляющая — на
# ЛЕВОЙ (низкий X, CHUTE2_X_LEFT — фиксированная координата, см. выше). В
# проде наоборот — там подход "через ленту" (анкер напротив), здесь подход
# "вдоль своей стороны" (анкер=цель), геометрия захода в раструб другая,
# отсюда и другая сторона скоса (ТЗ шага 3, живой прогон подтвердит).
#
# ПРАВКА 2026-07-30 (ТЗ, живой прогон): широкий (дальний) край раструба —
# CHUTE2_X_LEFT+width_start=8.45 — вылезал за щит 2 (x=8.2). Сокращён на
# 0.25м (width_start 1.2->0.95) — локальная копия _CHUTE, ПРОДУ (layout.yaml)
# не трогаем, копия используется только для этого раструба.
_CHUTE2 = dict(_CHUTE, width_start=round(_CHUTE["width_start"] - 0.25, 3))

BORDER1_X_END = CHUTE2_X_LEFT   # бортик шибера 1 (шаг 2) не может идти "до конца
                                # ленты" буквально — обрезан до начала раструба
                                # зоны B (та же сторона, +Y) — дальше контейнирование
                                # берут на себя стенки самого раструба (wall_b2_*).

BORDER2_Y_SIGN = -ANCHOR2_SIDE   # ПРОТИВОПОЛОЖНАЯ анкеру/цели сторона (-Y) —
                                # та же логика, что BORDER1_Y_SIGN у шибера 1
                                # (анкер=цель у обоих щитов теперь): бортик
                                # защищает НЕ-целевую сторону от случайного
                                # сталкивания при открывании щита 2.
BORDER2_X_SHIFT = -1.0   # ТЗ 2026-07-30 (повторно): весь бортик сдвинут на -1м по X
BORDER2_X_START = round(SHIBER2_X + BORDER2_X_SHIFT, 3)   # было = SHIBER2_X (столбик щита 2)
BORDER2_X_END = round(SEG_HIGH_END_X + BORDER2_X_SHIFT, 3)   # было = SEG_HIGH_END_X (конец ленты)

# --- Шибер 3 / второй ролл-кейдж (шаг 4) -----------------------------------
# ТЗ: "второй ролл-кейдж сбоку ленты, как первый, тоже узкой стороной, вход
# закрывает шибер 3" — та же схема, что шибер 1/кейдж 1 (flap_node_upstream,
# анкер=сторона цели, "загребание"), НА ТОЙ ЖЕ (-Y) стороне, что кейдж 1.
#
# МЕСТО (ТЗ 2026-07-30, живое обсуждение): между шибером 1 (кейдж 1 кончается
# на x=5.79) и зоной B (раструб начинается на x=7.25) — свободно ~1.46м.
# Плотно, поэтому кейдж 2 — УЗКОЙ гранью (0.8м вдоль ленты, как в
# изначальном ТЗ; кейдж 1 после живого прогона пришлось развернуть широкой
# гранью — 1.2м может физически не поместиться здесь без конфликта с кейджем
# 1/раструбом B, начинаем с узкой, как просили, и проверяем живым прогоном
# так же, как для кейджа 1). Порядок по ленте теперь: 1(round) ->
# 3(oversize) -> 2(ok) — НЕ как в проде (D->B->C), но это не проблема: этот
# демо-механизм не использует divert_safe_radius/порядок Router.step() из
# проды, каждый щит реагирует независимо по своей категории (см.
# _nearest_category в контроллере).
_CAGE2_YAW = 0.0   # узкая грань (0.8м) к ленте — см. вывод формулы в шаге 2
                   # (при yaw=0 SIDE_L обращена к ленте узкой гранью)
CAGE2_X = 6.4      # было 6.4 -> 6.5 -> 6.4 — сдвинуто на +0.1м (ТЗ), затем
                   # обратно на -0.1м (ТЗ 2026-07-30, повторно)
CAGE2_Y = round(-(_DV["anchor_y"] + _CAGE["length"] / 2), 3)   # -0.90 — та же
                   # формула, что была у кейджа 1 до разворота (узкая грань,
                   # ближняя стенка SIDE_L на линии анкера)
CAGE2_POS = {"x": CAGE2_X, "y": CAGE2_Y, "yaw": _CAGE2_YAW, "closed": True}

SHIBER3_CATEGORY = "oversize"   # последняя из трёх категорий (round/ok уже
                                # заняты шиберами 1/2) — аналог зоны C в проде

SHIBER3_X = 7.06   # было формулой CAGE2_X+0.66 — раскреплено (ТЗ 2026-07-30:
                   # кейдж двигали БЛИЖЕ к щиту, если оставить формулу, щит
                   # уехал бы вместе с кейджем и зазор не изменился бы);
                   # значение — то, что формула давала при CAGE2_X=6.4 (щит
                   # остаётся на месте, зазор кейдж->щит сократился 0.66->0.56)
ANCHOR3_SIDE = -1           # ТА ЖЕ сторона, что кейдж 1 (-Y) — ТЗ шага 4
                            # "как первый ролл-кейдж"
GATE3_POS = {"x": SHIBER3_X, "side": ANCHOR3_SIDE}
CLOSED3_ANGLE = 0.0                              # ФЛЕШ — вход перекрыт барьером
OPEN3_ANGLE = ANCHOR3_SIDE * _DV["deploy_angle"]  # тянет объект к стороне анкера

SENSOR3_X = round(SHIBER3_X - 1.0, 3)   # тот же офсет перед щитом, что у шиберов 1/2

# Бортик "напротив шибера" (ТЗ шага 4) — НЕ добавляем новый: BORDER1 (шаг 2)
# уже покрывает +Y (противоположную от анкера/цели шибера 3 сторону) на всём
# диапазоне от SEG_HIGH_X_START(4.85) до начала раструба B(7.25) — шибер 3
# (x≈7.06) целиком внутри этого диапазона, второй бортик был бы дублирующим.

# --- Накопитель (шаг 5) -----------------------------------------------------
# ТЗ (уточнено 2026-07-30 после шага 4): места по прямой не осталось (после
# щита 2 на x=8.2 до края плиты — 1.15м, впритык). Решение: конвейер
# поворачивает НАПРАВО (-Y — та же сторона, что кейджи 1/2, там всего ±3.0м,
# использовано только до y≈-1.7) лентой, едущей под 90°, тем же способом, что
# приёмная лента belt_b у входа в зону B.
#
# ПЕРВОЕ ПРИБЛИЖЕНИЕ, НЕ ПРОВЕРЕНО ЖИВЫМ ПРОГОНОМ (это принципиально НОВАЯ для
# сцены задача): раньше 90°-повороты (зона B) всегда получали объект уже с
# боковой (Y) скоростью от АКТИВНОГО толчка шибера — статический раструб там
# лишь ДОВОДИТ уже боковое движение. Здесь же объект прилетает с ЧИСТО
# продольной (X) скоростью (никто его не толкает — это "всё, что доехало до
# конца"), редиректить нужно с нуля. Решение — статический ОТРАЖАТЕЛЬ:
# диагональная стенка через ВСЮ ширину ленты у самого её конца (тот же приём
# angled_wall_node, что стенки раструба B, только здесь стенка ВСЕГДА
# "развёрнута", без мотора — накопителю не нужна избирательность, туда
# должно попадать всё). Угол отражателя (45° к ходу ленты, круче 35°
# deploy_angle щитов — тем нужен лишь небольшой боковой увод, а тут требуется
# развернуть само направление движения) — первая прикидка, детали контакта
# (соскользнёт ли товар по стенке в -Y, а не просто впечатается) — проверить
# живым прогоном, как и остальные щиты/кейджи в этой сцене.
ACCUM_DEFLECT_X_SHIFT = -0.1   # ТЗ 2026-07-30: сама направляющая сдвинута на -0.1м
ACCUM_DEFLECT_X_START = round(SHIBER2_X + 0.15 + ACCUM_DEFLECT_X_SHIFT, 3)   # начало
                                                       # отражателя — с запасом ПОСЛЕ
                                                       # щита 2 (иначе накладывался бы
                                                       # на его стойку/полотно), плюс сдвиг
ACCUM_DEFLECT_X_END = round(SEG_HIGH_END_X + ACCUM_DEFLECT_X_SHIFT, 3)   # конец —
                                                       # у края ленты (0.5м пробег), плюс сдвиг
ACCUM_X_CENTER = 8.35   # РАСКРЕПЛЕНО от ACCUM_DEFLECT_X_END (ТЗ 2026-07-30: "смести
                        # НАПРАВЛЯЮЩУЮ" — если оставить формулой, накопитель уехал бы
                        # ВМЕСТЕ с ней, хотя просили сдвинуть только направляющую; тот
                        # же урок, что с CAGE2_X/SHIBER3_X). Значение — то, что формула
                        # ACCUM_DEFLECT_X_END-0.3-0.2 давала ДО этого сдвига направляющей.
ACCUM_WIDTH = 1.0     # "расширение до 1м" (ТЗ)
ACCUM_LENGTH = 1.0    # "накопитель 1м длиной" (ТЗ)
ACCUM_DESCENT_LENGTH = 0.3   # длина ската спуска (вдоль -Y)
ACCUM_HEIGHT_DROP = 0.20     # "спуск 20см" (ТЗ)
ACCUM_HEIGHT_LOW = round(SEG_HIGH_HEIGHT - ACCUM_HEIGHT_DROP, 3)
ACCUM_DESCENT_START_DROP = 0.05   # было 0.02 — ребро всё ещё выпирало (товары
                                  # цеплялись) — опущено ЕЩЁ на 3см (ТЗ 2026-07-30,
                                  # повторно), итого 5см ниже поверхности ленты
ACCUM_BORDER_HEIGHT = 0.04   # "невысокие бортики" (ТЗ 2026-07-30) — как BORDER_HEIGHT
                             # у щитов, только для площадки накопителя (3 стороны,
                             # без входа со стороны ската)

ACCUM_Y_EDGE = round(-BELT_WIDTH / 2, 3)                       # -0.25, кромка ленты (-Y)
ACCUM_Y_SPUSK_END = round(ACCUM_Y_EDGE - ACCUM_DESCENT_LENGTH, 3)   # конец ската/начало платформы
ACCUM_Y_END = round(ACCUM_Y_SPUSK_END - ACCUM_LENGTH, 3)            # дальний край накопителя


# Порт HTTP-панели demo_panel_mechanism.py (PANEL_PORT там же) — тот же сервер отдаёт
# STL-меши по /meshes/ (см. mesh_templates_node ниже, правка 2026-07-31).
PANEL_HTTP_PORT = 8022


def mesh_templates_node() -> str:
    """DEF MESH_<type> на каждый реальный товар — тот же приём, что в
    tools/gen_demo_roller_shiber.py (STL парсится один раз при загрузке мира,
    спавн дальше делает `USE MESH_<type>`).

    url — массив с ДВУМЯ URL (HTTP + относительный-fallback), не один:
      1. HTTP `"http://127.0.0.1:{PANEL_HTTP_PORT}/meshes/<name>.stl"` —
         браузерный webotsJS (web/wwi/MeshLoader.js) при `MeshLoader.stream`
         добавляет `worldsPath` (путь мира) к относительному URL, что даёт
         неправильный путь (404 от :1235 — историческая причина перехода на
         HTTP). HTTP-URL не попадает в эту ветку (`!url.startsWith('http')`),
         резолвится напрямую — web-стрим работает.
      2. относительный `"../../assets/meshes/<name>.stl"` — нативный Webots
         (оконный режим) читает STL с диска при парсинге мира, до старта
         панели (порт 8022 открывается позже). Нативный Webots fallback'ит на
         этот URL, если первый (HTTP) недоступен (`Connection refused` при
         парсинге до старта панели) — без него белый/пустой меш в нативном окне.

    Конфликт: нативный Webots пробует url по порядку с fallback (если первый
    недоступен — берёт второй); webotsJS берёт только первый. Поэтому HTTP
    первым (web-стрим), относительный вторым (fallback для нативного). Для
    текстур меток (`_aruco_marker_shape`) — НАОБОРОТ, относительный первым:
    `ImageLoader.js` при стриме резолвит `prefix + url` (без worldsPath) и
    относительный даёт правильный `http://localhost:8022/assets/...`, а
    нативный Webots читает его с диска сразу — без гонки с панелью. Разница:
    `MeshLoader` добавляет `worldsPath` (ломает относительный), `ImageLoader`
    при стриме — нет.

    Спавн объекта (supervisor в demo_panel_mechanism.py) дальше делает
    `geometry USE MESH_<type>` вместо собственного `Mesh { url }`: STL
    скачивается и парсится один раз при загрузке мира, не на каждый спавн.
    """
    shapes = "\n    ".join(
        f'Shape {{ castShadows FALSE geometry DEF MESH_{name} Mesh {{ '
        f'url [ "http://127.0.0.1:{PANEL_HTTP_PORT}/meshes/{name}.stl", '
        f'"../../assets/meshes/{name}.stl" ] }} }}'
        for name in REAL_OBJECTS
    )
    return f"""DEF MESH_TEMPLATES Solid {{
  translation 0 0 -50
  name "mesh_templates"
  children [
    {shapes}
  ]
}}"""


SIMPLE_SHAPE_SIZES = {"small": 0.05, "medium": 0.10, "large": 0.20}  # м (ребро/диаметр)
# ВАЖНО: "cube_"/"ball_", НЕ "box_"/"sphere_" — config/objects.yaml уже
# содержит РЕАЛЬНЫЕ STL-типы "box_small"/"box_large" (найдено по фидбеку
# пользователя: эти простые формы с именами "box_small"/"box_large"
# перехватывали спавн реальных STL-объектов того же имени в
# demo_panel_mechanism.py — kind.startswith("box_") проверялся РАНЬШЕ,
# чем kind in REAL_OBJECTS, поэтому реальный короб никогда не долетал
# до real_object_node). Уникальный префикс убирает коллизию навсегда,
# а не только переставляет порядок проверки.
SIMPLE_SHAPES = (
    [(f"cube_{k}", f"Куб {int(v * 1000)}мм") for k, v in SIMPLE_SHAPE_SIZES.items()]
    + [(f"ball_{k}", f"Шар {int(v * 1000)}мм") for k, v in SIMPLE_SHAPE_SIZES.items()]
)


def simple_shape_node(name: str, kind: str, x: float, y: float) -> str:
    shape, size_key = kind.split("_", 1)
    size = SIMPLE_SHAPE_SIZES[size_key]
    if shape == "cube":
        geom = f"Box {{ size {size} {size} {size} }}"
        color = "0.75 0.45 0.20"
        damping = ""
    else:
        geom = f"Sphere {{ radius {size / 2} subdivision 2 }}"
        color = "0.25 0.55 0.80"
        damping = "damping Damping { linear 0.05 angular 0.3 }"
    z = SEG_LOW_HEIGHT + size / 2 + 0.002
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


def real_object_node(name: str, obj_type: str, x: float, y: float, yaw_deg: float) -> str:
    """Реальный товар из config/objects.yaml — та же геометрия/коллизия, что
    на основной линии (см. tools/gen_demo_roller_shiber.py::real_object_node,
    формулы не дублируются заново, только адаптирован BELT_HEIGHT -> появление
    всегда на низком сегменте, SEG_LOW_HEIGHT)."""
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
  translation {x} {y} {SEG_LOW_HEIGHT + 0.002}
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


def camera_marker_node(name: str, cam: dict, rig: dict) -> str:
    """Чисто визуальный маркер положения РЕАЛЬНОЙ камеры (не зеркальной —
    зеркальные позиции виртуальные, у них уже есть свой визуал через рамку
    зеркала cv_mirror_node): тёмный корпус-кубик + цветная точка-«объектив»
    по направлению взгляда (локальная +X = forward, см. docstring
    tools.gen_world._camera_frame). Camera node сам по себе невидим в 3D-виде
    Webots — без этого маркера нельзя на глаз проверить раскладку рига.

    ВАЖНО (найдено 2026-07-30 живым прогоном пользователя): корпус/объектив
    должны стоять ЦЕЛИКОМ СЗАДИ фокальной точки камеры (локальная -X, назад
    от направления взгляда), а не вокруг/перед ней — иначе для КАМЕРЫ,
    которую показывает live-вид панели (см. demo_panel_mechanism.py), сам
    маркер торчит перед объективом и в кадре выглядит как гигантский
    цветной объект, закрывающий почти весь кадр (был центрирован ровно на
    позиции камеры, половина геометрии оказывалась перед ней)."""
    if cam.get("mirror"):
        return ""
    theta = math.radians(cam["angle"])
    d = cam["distance"]
    x = rig["x"]
    y = d * math.cos(theta)
    z = rig["belt_z"] + d * math.sin(theta)
    axis, angle = _rotation_from_frame(*_camera_frame(theta))
    body_depth = 0.06
    gap = 0.01   # запас, чтобы даже задняя грань объектива не касалась фокальной точки
    body_x = -(body_depth / 2 + gap)
    lens_x = -gap
    return f"""Solid {{
  translation {x} {y} {z}
  rotation {axis[0]} {axis[1]} {axis[2]} {angle}
  name "cv_marker_{name}"
  children [
    Pose {{
      translation {body_x} 0 0
      children [
        Shape {{
          appearance PBRAppearance {{ baseColor 0.12 0.12 0.14 roughness 0.5 metalness 0.6 }}
          geometry Box {{ size {body_depth} 0.05 0.05 }}
        }}
      ]
    }}
    Pose {{
      translation {lens_x} 0 0
      children [
        Shape {{
          appearance PBRAppearance {{ baseColor 0.15 0.85 0.95 roughness 0.2 metalness 0.3 emissiveColor 0.05 0.35 0.4 }}
          geometry Sphere {{ radius 0.014 subdivision 1 }}
        }}
      ]
    }}
  ]
}}"""


def cv_markers_node() -> str:
    markers = "\n".join(
        camera_marker_node(name, cam, CV_RIG) for name, cam in CV_RIG["cameras"].items()
    )
    mirror = cv_mirror_node(CV_RIG)
    return f"{markers}\n{mirror}"


# --- Шаг 6: визуальные ArUco-метки (декоративные, tools/gen_aruco_textures.py) ---
# Метки НЕ читаются никаким кодом сцены (нет solvePnP/детектора) — только визуальное
# оформление, как и сами cv-камеры/зеркало (см. cv_markers_node выше). Полноценная
# задача с реальными мировыми координатами под будущий solvePnP — отдельный Backlog
# [[Добавить визуальные ArUco-метки в сцену Webots]], здесь её не открываем.
ARUCO_MARKER_SIDE = 0.10   # 100мм — как в реальном риге roboson_tools


def _aruco_marker_shape(texture_file: str, side: float, asset_dir: str = "aruco") -> str:
    """Shape с текстурой метки на квадрате в локальной плоскости X-Y (z=0) —
    `IndexedFaceSet` с ДВУМЯ гранями (прямой + обратный winding), чтобы метка
    была видна с ОБЕИХ сторон: правильная сторона (лицом к ленте/камерам) для
    зеркальных марок и марок на бортах ленты не очевидна без живой визуальной
    проверки (см. [[feedback-less-self-verification-more-handoff]] — не гонять
    headless ради этого), а удвоенная геометрия снимает вопрос совсем — не
    нужно гадать со знаком поворота. `asset_dir` — подкаталог `assets/` (по
    умолчанию `aruco`, переиспользуется и для не-ArUco декоративных текстур
    вроде подписей ролл-кейджей, см. `cage_label_node`).

    URL текстуры — массив с ДВУМЯ URL (HTTP + относительный-fallback), не один,
    та же схема, что `mesh_templates_node` (HTTP первым): браузерный webotsJS
    (web/wwi/Parser.js) берёт ПЕРВЫЙ URL из массива, а `ImageLoader.js` при
    стриме резолвит `prefix + url`, где `prefix` = origin **стрим-сервера**
    (`http://localhost:1235/`, см. webots.js:154 `this.prefix = httpServerUrl
    + '/'`) — НЕ панели (`:8022`). Относительный `"../../assets/..."` резолвится
    в `http://localhost:1235/assets/...`, а стрим-сервер `/assets/` не отдаёт →
    404 → "missed texture" (обнаружено живым прогоном в --headless). HTTP-URL
    на панель `:8022` (которая отдаёт `/assets/` через STATIC_ROUTES) работает
    напрямую, без резолвинга через prefix.
      1. HTTP `"http://127.0.0.1:{PANEL_HTTP_PORT}/assets/<dir>/<file>"` —
         web-стрим (браузер грузит с панели :8022, CORS-заголовок уже есть).
      2. относительный `"../../assets/<dir>/<file>"` — нативный Webots (оконный
         режим) читает PNG с диска сразу при парсинге мира, до старта панели
         (порт 8022 открывается позже). Нативный Webots fallback'ит на этот
         URL, если первый (HTTP) недоступен (`Connection refused` при парсинге
         до старта панели) — без него белые квадраты в нативном окне.
    Проверено: оба режима работают (headless — через панель по HTTP, оконный —
    с диска по относительному fallback'у).

    Поле `solid FALSE` (VRML/X3V Webots'ом не поддерживается —
    `IndexedFaceSet` в Webots имеет только coord/normal/texCoord/ccw/convex/
    normalPerVertex/coordIndex/normalIndex/texCoordIndex/creaseAngle, см.
    ~/opt/webots/resources/nodes/IndexedFaceSet.wrl), парсер выдаёт
    `Skipped unknown 'solid' field` на каждой метке — нефатально, но
    двустороннего рендера НЕ даёт. Вместо него — вторая грань с обратным
    порядком вершин (0 3 2 1)."""
    h = side / 2
    return f"""Shape {{
          appearance PBRAppearance {{
            baseColorMap ImageTexture {{ url [ "http://127.0.0.1:{PANEL_HTTP_PORT}/assets/{asset_dir}/{texture_file}", "../../assets/{asset_dir}/{texture_file}" ] }}
            roughness 1
            metalness 0
          }}
          geometry IndexedFaceSet {{
            coord Coordinate {{ point [ {-h} {-h} 0, {h} {-h} 0, {h} {h} 0, {-h} {h} 0 ] }}
            coordIndex [ 0 1 2 3 -1, 0 3 2 1 -1 ]
            texCoord TextureCoordinate {{ point [ 0 0, 1 0, 1 1, 0 1 ] }}
            texCoordIndex [ 0 1 2 3 -1, 0 1 2 3 -1 ]
          }}
        }}"""


def mirror_aruco_markers_node(rig: dict) -> str:
    """4 метки по углам полотна зеркала (ID 22/23/24/25 — те же, что доски
    `mirror_edge_left` (top=22, bottom=23) и `mirror_edge_right` (top=24,
    bottom=25) в roboson_tools, только для единообразия ID, координаты этой
    сцены с теми yaml никак не связаны). Та же геометрия рамки (позиция/
    поворот панели), что `cv_mirror_node` — пересчитана здесь заново (а не
    выведена из вызова той функции, которая возвращает готовую VRML-строку
    без промежуточных чисел)."""
    tilt = math.radians(rig["mirror_tilt_from_vertical_deg"])
    lean = rig["mirror_lean_sign"]
    d_y = lean * math.sin(tilt)
    d_z = math.cos(tilt)
    bottom_y, bottom_z = rig["mirror_bottom_edge"]
    sx, _st, sz = rig["mirror_size"]
    mid_y = bottom_y + 0.5 * sz * d_y
    mid_z = bottom_z + 0.5 * sz * d_z
    x = rig["x"]
    y = mid_y
    z = rig["belt_z"] + mid_z
    rot_angle = math.atan2(-d_y, d_z)

    margin = 0.08  # отступ центра метки от края полотна, м
    x_off = sx / 2 - margin
    z_off = sz / 2 - margin

    def marker(x_sign: float, z_sign: float, texture: str) -> str:
        return f"""    Pose {{
      translation {x_sign * x_off} 0 {z_sign * z_off}
      rotation 1 0 0 -1.5708
      children [ {_aruco_marker_shape(texture, ARUCO_MARKER_SIDE)} ]
    }}"""

    corners = "\n".join([
        marker(-1, +1, "marker_22_mirror_left_top.png"),
        marker(-1, -1, "marker_23_mirror_left_bottom.png"),
        marker(+1, +1, "marker_24_mirror_right_top.png"),
        marker(+1, -1, "marker_25_mirror_right_bottom.png"),
    ])
    return f"""Solid {{
  translation {x} {y} {z}
  rotation 1 0 0 {rot_angle}
  name "mirror_aruco_markers"
  children [
{corners}
  ]
}}"""


def belt_aruco_markers_node(rig: dict, belt_height: float) -> str:
    """2 доски по 2 метки на бортах ленты в CV-зоне (ID 26/27 — доска
    `belt_left`, ID 28/29 — доска `belt_right` в roboson_tools, см. докстринг
    mirror_aruco_markers_node про единообразие ID). Лежат ПЛАШМЯ в плоскости
    ленты (без дополнительного поворота — Shape уже в локальной плоскости
    X-Y, которая здесь горизонтальна), на 1мм выше поверхности, как декаль —
    не стоят вертикально, как в прошлой версии."""
    x = rig["x"]
    z = belt_height + 0.001
    # +0.05м от кромки ленты (по замечанию пользователя — метки лежали НА ленте)
    y_edge = BELT_WIDTH / 2 + 0.05
    # база 0.25 + 0.2 в каждую сторону (по замечанию пользователя — раздвинуть дальше)
    separation = 0.25 + 2 * 0.2

    def marker(name: str, x_pos: float, y: float, texture: str) -> str:
        return f"""Solid {{
  translation {x_pos} {y} {z}
  name "{name}"
  children [ {_aruco_marker_shape(texture, ARUCO_MARKER_SIDE)} ]
}}"""

    return "\n".join([
        marker("belt_marker_left_top", x - separation / 2, -y_edge, "marker_26_belt_left_top.png"),
        marker("belt_marker_left_bottom", x + separation / 2, -y_edge, "marker_27_belt_left_bottom.png"),
        marker("belt_marker_right_top", x - separation / 2, y_edge, "marker_28_belt_right_top.png"),
        marker("belt_marker_right_bottom", x + separation / 2, y_edge, "marker_29_belt_right_bottom.png"),
    ])


CAGE_LABEL_SIDE = 0.3   # м — размер подписи C/D на боковой стенке кейджа
# RollCage.proto: floorHeight по умолчанию 0.15 (не переопределяется
# roll_cage_node) — та же величина нужна здесь для высоты подписи по центру
# стенки, не выведена из proto напрямую (генератор не парсит .proto).
_CAGE_FLOOR_HEIGHT = 0.15
_CAGE_WALL_CZ = _CAGE_FLOOR_HEIGHT + (_CAGE["height"] - _CAGE_FLOOR_HEIGHT) / 2


CAGE_LABEL_Y = -1.505   # абсолютная мировая Y — задана пользователем напрямую
                        # (живая проверка позиции/поворота в Webots), не
                        # выведена формулой от геометрии кейджа

def cage_label_node(name: str, cage_x: float, texture_file: str) -> str:
    """Подпись (C/D — соответствие зонам продакшн-сортировщика, см.
    `sim/router.py::CATEGORY_TO_ZONE`) на боковой стенке кейджа. Y и поворот
    подобраны пользователем живым прогоном (см. `CAGE_LABEL_Y` выше) —
    оба кейджа используют одну и ту же Y (их центры совпадают: CAGE1_Y ==
    CAGE2_Y == -0.9), только X различается. `solid FALSE` (как в
    `_aruco_marker_shape`) — не выясняем, какая сторона стенки физически
    смотрит на камеры/зрителя."""
    return f"""Solid {{
  translation {cage_x} {CAGE_LABEL_Y} {_CAGE_WALL_CZ}
  rotation 1 0 0 1.5708
  name "{name}"
  children [ {_aruco_marker_shape(texture_file, CAGE_LABEL_SIDE, asset_dir="cage_labels")} ]
}}"""


def beam_sensor_node(name: str, x: float, height: float) -> str:
    """Лучевой (through-beam) датчик перед шибером — тот же паттерн, что
    `beam_sensor_node` в tools/gen_demo_flap_shiber.py (излучатель на одной
    кромке, луч поперёк ленты вдоль -Y ко второй), параметризован по x/высоте
    ленты в этой точке — нужен на несколько шиберов (2/3 — следующие шаги),
    поэтому не константа, а функция с именем устройства на вход."""
    y = BELT_WIDTH / 2
    z = height + 0.01   # 10мм над лентой, как в demo_flap_shiber.py
    return f"""    Pose {{
      translation {x} {y} {z}
      rotation 0 0 1 -1.5708
      children [
        DistanceSensor {{
          name "{name}"
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


def side_border_node(name: str, x_start: float, x_end: float, y_sign: float, height: float) -> str:
    """Статический бортик по одной кромке ленты — тот же приём, что
    `border_left_node` в tools/gen_demo_flap_shiber.py (SimpleConveyor.borderHeight
    ставит бортики сразу с обеих сторон, шиберу нужна открытая сторона —
    поэтому отдельный узел только на противоположной от анкера кромке),
    параметризован по X-диапазону и стороне (нужен на несколько шиберов)."""
    length = x_end - x_start
    xc = (x_start + x_end) / 2
    y = y_sign * (0.5 * BELT_WIDTH + 0.025)
    z = height + 0.5 * BORDER_HEIGHT - 0.03
    size_z = BORDER_HEIGHT + 0.06
    return f"""Solid {{
  translation {xc} {y} {z}
  name "{name}"
  children [
    Shape {{
      appearance PBRAppearance {{ baseColor 0.42 0.46 0.52 roughness 0.6 metalness 0.4 }}
      geometry DEF {name.upper()}_BO Box {{ size {length} {BORDER_THICKNESS} {size_z} }}
    }}
  ]
  boundingObject USE {name.upper()}_BO
}}"""


def zone_b_chute_node_mirrored(chute: dict, x_left: float, lane_y_end: float) -> str:
    """Зеркальный (лево<->право) вариант tools.gen_world.zone_b_chute_node —
    см. комментарий у CHUTE2_X_LEFT: скос (широкий раструб -> сужение) на
    ПРАВОЙ (высокий X) стороне, прямая направляющая — на ЛЕВОЙ (низкий X,
    x_left — фиксированная координата). В проде фиксирована ПРАВАЯ сторона
    (`x_right`), скос — слева; здесь ровно наоборот, ширины/высоты/Y-диапазон
    берутся из того же `chute` (config/layout.yaml: zone_b_chute) без
    изменений — меняется только то, какая сторона фиксирована/скошена."""
    y_lo, y_hi = chute["y_start"], chute["y_end"]
    z_lo, z_hi = chute["height_start"], chute["height_end"]
    w_start, w_end = chute["width_start"], chute["width_end"]
    thickness = 0.05

    dy, dz = y_hi - y_lo, z_hi - z_lo
    true_length = math.hypot(dy, dz)
    phi = math.atan2(dz, dy)
    mid_y, mid_z = (y_lo + y_hi) / 2, (z_lo + z_hi) / 2
    ty = mid_y + (thickness / 2) * math.sin(phi)
    tz = mid_z - (thickness / 2) * math.cos(phi)
    floor_width = w_start
    tx = x_left + floor_width / 2

    floor = f"""Solid {{
  translation {tx} {ty} {tz}
  rotation 1 0 0 {phi}
  name "zone_b2_chute_floor"
  contactMaterial "chute"
  children [
    Shape {{
      appearance PBRAppearance {{ baseColor 0.16 0.16 0.18 roughness 0.9 metalness 0 }}
      geometry DEF CHUTE2_FLOOR_BO Box {{ size {floor_width} {true_length} {thickness} }}
    }}
  ]
  boundingObject USE CHUTE2_FLOOR_BO
}}"""

    z_wall_lo, z_wall_hi = 0.55, 0.90
    left_wall = angled_wall_node("wall_b2_left", x_left, y_lo, x_left, lane_y_end,
                                  0.02, z_wall_lo, z_wall_hi)
    right_taper = angled_wall_node("wall_b2_right_taper", x_left + w_start, y_lo,
                                    x_left + w_end, y_hi, 0.02, z_wall_lo, z_wall_hi)
    right_straight = angled_wall_node("wall_b2_right_straight", x_left + w_end, y_hi,
                                       x_left + w_end, lane_y_end, 0.02, z_wall_lo, z_wall_hi)
    return "\n".join([floor, left_wall, right_taper, right_straight])


def accum_deflector_node() -> str:
    """Статический отражатель у конца ленты (см. ПРАВКУ 2026-07-30 у
    констант ACCUM_*) — диагональная стенка через ВСЮ ширину ленты (от +Y до
    -Y кромки), БЕЗ мотора: в отличие от щитов 1-3, накопителю не нужна
    избирательность — сюда должно попадать абсолютно всё, что доехало до
    конца.

    ПРАВКА 2026-07-30 (ТЗ, живой прогон): товары упирались в направляющую и
    останавливались, НЕ соскальзывая в накопитель — трение по умолчанию
    (`angled_wall_node`, tools.gen_world, БЕЗ contactMaterial) слишком
    высокое для скольжения вдоль диагональной стенки, ровно та же причина,
    что была у "щит отфутболивал объекты" (см. историю) — только наоборот
    (там трение снижали от избытка резкости, здесь — от избытка сцепления).
    Полотно щита-раздатчика (`material2 "paddle"`, coulombFriction=0.01,
    ГЛАДКОЕ специально, чтобы объект скользил вдоль него) — та же по сути
    задача (диагональная стенка, вдоль которой должен ехать объект), поэтому
    переиспользуем тот же `contactMaterial "paddle"`, а не отдельную
    `angled_wall_node` (та не параметризована по материалу) — здесь
    самостоятельная копия той же геометрии стенки с добавленным полем."""
    x0, y0 = ACCUM_DEFLECT_X_START, BELT_WIDTH / 2
    x1, y1 = ACCUM_DEFLECT_X_END, -BELT_WIDTH / 2
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)
    angle = math.atan2(dy, dx)
    mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2
    z_lo, z_hi = 0.55, 0.90
    zc, h = (z_lo + z_hi) / 2, z_hi - z_lo
    thickness = 0.02
    return f"""Solid {{
  translation {mid_x} {mid_y} {zc}
  rotation 0 0 1 {angle}
  name "accum_deflector"
  contactMaterial "paddle"
  children [
    Shape {{
      appearance PBRAppearance {{ baseColor 0.42 0.46 0.52 roughness 0.6 metalness 0.4 }}
      geometry DEF ACCUM_DEFLECTOR_BO Box {{ size {length} {thickness} {h} }}
    }}
  ]
  boundingObject USE ACCUM_DEFLECTOR_BO
}}"""


def accum_descent_node() -> str:
    """Спуск (20см) с расширением до 1м — статическая наклонная плита, тот же
    приём, что d_entry_ramp в проде (tools.gen_world.tilted_plate_node):
    ширина (x0,x1) не меняется по длине ската, только Y/Z.

    ПРАВКА 2026-07-30 (ТЗ, живой прогон): верхнее ребро (стык с лентой) чуть
    выпирало ВЫШЕ поверхности ленты — стартовая высота ската опущена на
    ACCUM_DESCENT_START_DROP (2см) НИЖЕ SEG_HIGH_HEIGHT вместо ровно вровень."""
    x0 = ACCUM_X_CENTER - ACCUM_WIDTH / 2
    x1 = ACCUM_X_CENTER + ACCUM_WIDTH / 2
    z0 = round(SEG_HIGH_HEIGHT - ACCUM_DESCENT_START_DROP, 3)
    return tilted_plate_node("accum_descent", x0, x1, ACCUM_Y_EDGE, ACCUM_Y_SPUSK_END,
                              z0, ACCUM_HEIGHT_LOW, 0.05)


def accum_platform_node() -> str:
    """Сама площадка накопителя (1x1м, плоская).

    ПРАВКА 2026-07-30 (ТЗ, живой прогон): изначально — ОТДЕЛЬНЫЙ узел (не
    tilted_plate_node), намеренно с дефолтным (не "chute") трением — идея
    была "площадка, где товар должен ОСЕСТЬ, а не продолжать скользить".
    Пользователь попросил обратное — низкое трение, как у ската (тот же
    "chute", что и accum_descent_node/tools.gen_world.tilted_plate_node) —
    вероятно, резкий скачок трения на стыке скат->площадка сам по себе
    цеплял объекты (тот же класс проблемы, что "цепляние" за ребро ската
    выше)."""
    xc = ACCUM_X_CENTER
    yc = (ACCUM_Y_SPUSK_END + ACCUM_Y_END) / 2
    thickness = 0.05
    zc = ACCUM_HEIGHT_LOW - thickness / 2
    length = ACCUM_Y_SPUSK_END - ACCUM_Y_END
    return f"""Solid {{
  translation {xc} {yc} {zc}
  name "accumulator"
  contactMaterial "chute"
  children [
    Shape {{
      appearance PBRAppearance {{ baseColor 0.35 0.40 0.46 roughness 0.7 metalness 0.3 }}
      geometry DEF ACCUM_PLATFORM_BO Box {{ size {ACCUM_WIDTH} {length} {thickness} }}
    }}
  ]
  boundingObject USE ACCUM_PLATFORM_BO
}}"""


def accum_borders_node() -> str:
    """Невысокие бортики по периметру площадки накопителя (ТЗ 2026-07-30) —
    3 стороны (левая/правая/дальняя), БЕЗ стороны ската (там должен быть
    свободный въезд с спуска, см. accum_descent_node)."""
    x0 = ACCUM_X_CENTER - ACCUM_WIDTH / 2
    x1 = ACCUM_X_CENTER + ACCUM_WIDTH / 2
    z_lo = ACCUM_HEIGHT_LOW
    z_hi = round(ACCUM_HEIGHT_LOW + ACCUM_BORDER_HEIGHT, 3)
    left = angled_wall_node("accum_border_left", x0, ACCUM_Y_SPUSK_END, x0, ACCUM_Y_END,
                             0.02, z_lo, z_hi)
    right = angled_wall_node("accum_border_right", x1, ACCUM_Y_SPUSK_END, x1, ACCUM_Y_END,
                              0.02, z_lo, z_hi)
    far = angled_wall_node("accum_border_far", x0, ACCUM_Y_END, x1, ACCUM_Y_END,
                            0.02, z_lo, z_hi)
    return "\n".join([left, right, far])


def demo_panel_node() -> str:
    """Супервизор веб-панели (спавн по HTTP) — см.
    webots/controllers/demo_panel_mechanism/. CV-камеры (Camera-узлы) И щит
    шибера 1 (HingeJoint с мотором+датчиком угла) И его лучевой датчик — ДЕТИ
    этого же Robot (не отдельного узла): контроллер видит через getDevice()
    только устройства СВОЕГО Robot (тот же приём, что DEMO_PANEL в
    tools/gen_demo_flap_shiber.py — supervisor и владелец мотора щита в
    одном узле)."""
    cameras = "\n".join(cv_camera_node(name, cam, CV_RIG) for name, cam in CV_RIG["cameras"].items())
    shiber1 = flap_node_upstream("shiber1", _DV_MOTOR, GATE1_POS)
    sensor1 = beam_sensor_node("beam_sensor_1", SENSOR1_X, SEG_HIGH_HEIGHT)
    shiber2 = flap_node_upstream("shiber2", _DV_MOTOR, GATE2_POS)
    sensor2 = beam_sensor_node("beam_sensor_2", SENSOR2_X, SEG_HIGH_HEIGHT)
    shiber3 = flap_node_upstream("shiber3", _DV_MOTOR, GATE3_POS)
    sensor3 = beam_sensor_node("beam_sensor_3", SENSOR3_X, SEG_HIGH_HEIGHT)
    return f"""DEF DEMO_PANEL Robot {{
  name "demo_panel"
  controller "demo_panel_mechanism"
  supervisor TRUE
  children [
{cameras}
{shiber1}
{sensor1}
{shiber2}
{sensor2}
{shiber3}
{sensor3}
  ]
}}"""


def main() -> None:
    belt_length_total = SEG_HIGH_END_X - SEG_LOW_X_START
    belt_xc = SEG_LOW_X_START + belt_length_total / 2
    plate_xc = PLATE_X_START + PLATE_LENGTH_X / 2
    low = flat_conveyor_node("belt_low", SEG_LOW_X_START, SEG_LOW_LENGTH, BELT_WIDTH,
                              SEG_LOW_HEIGHT, BELT_SPEED)
    ramp = ramp_conveyor_node("belt_ramp", SEG_RAMP_X_START, SEG_RAMP_LENGTH, BELT_WIDTH,
                               SEG_LOW_HEIGHT, SEG_HIGH_HEIGHT, BELT_SPEED)
    high = flat_conveyor_node("belt_high", SEG_HIGH_X_START, SEG_HIGH_LENGTH, BELT_WIDTH,
                               SEG_HIGH_HEIGHT, BELT_SPEED)
    belt_b = belt_b_node(dict(_BELT_B, x_center=BELT_B_X_CENTER))
    chute2 = zone_b_chute_node_mirrored(_CHUTE2, CHUTE2_X_LEFT, _BELT_B["y_end"])

    world = f"""#VRML_SIM R2025a utf8
# Сгенерировано tools/gen_demo_mechanism.py — руками не править.
# Автономная демо-сцена исполнительного механизма (не часть sorting_line.wbt).
# Шаг 1/6: профиль ленты (низкий сегмент как в проде +20см буфера -> подъём ->
# высокий сегмент, конец на +60см дальше проды) + CV-зона на низком сегменте
# (как в проде, декоративно) + визуализация камер.
# Шаг 2/6: шибер 1 (загребание, узел flap_node_upstream — 180° от paddle_node
# проды) + повёрнутый (узкой стороной к ленте) ролл-кейдж + лучевой датчик +
# бортик на противоположной от щита кромке.
# Шаги 3-5/6: шибер 2 (зона B) + шибер 3 (oversize) + временная зона накопления
# (поворот через отражатель + скат). Шаг 6/6: декоративные ArUco-метки на
# зеркале и бортах ленты (не читаются кодом).

EXTERNPROTO "../protos/SimpleConveyor.proto"
EXTERNPROTO "../protos/RollCage.proto"

WorldInfo {{
  title "Demo: исполнительный механизм (шаги 1-6 — лента + 3 шибера + накопитель + ArUco-метки)"
  basicTimeStep 4
  optimalThreadCount {min(os.cpu_count() or 1, 4)}
  contactProperties [
    ContactProperties {{
      # basicTimeStep снижен с 8 до 4 (ТЗ 2026-07-30): "WARNING: The current
      # physics step could not be computed correctly" при резком столкновении
      # мелких объектов с бортиками — это ОТДЕЛЬНАЯ проблема от предупреждения
      # про >100k вершин STL (то — только рендер, boundingObject у товаров
      # свой, упрощённый convex-decomposition, см. sim/collision.py — на
      # физику вершины визуального меша не влияют, отключать физику STL не
      # нужно). Мельче шаг — прямая рекомендация самого предупреждения
      # Webots, безопасна и для тайминга щитов (velocity=6 у flap-щитов уже
      # подобран под ЗАПАС на больший шаг (8мс) без туннелирования — меньший
      # шаг (4мс) только снижает риск туннелирования, не повышает).
      # softCFM здесь тоже поднят (0.0001->0.001, мягче дефолтного жёсткого
      # контакта) — та же идея, что уже сработала для щита (см. material2
      # "paddle" ниже): резкий удар мелкого объекта об бортик частично
      # гасится, а не разрешается один жёстким импульсом.
      coulombFriction [ 1.2 ]
      rollingFriction 0.05 0.05 0.05
      bounce 0
      softCFM 0.001
    }}
    ContactProperties {{
      # softCFM поднят с 0.0001 (ТЗ 2026-07-30): щит "отфутболивал" объекты —
      # при жёстком контакте ODE резко гасит проникновение за один шаг, и
      # относительная скорость лопасти (~7 м/с на конце полотна) против
      # объекта (лента ~1 м/с) резко переходит в объект одним импульсом.
      # Больший softCFM делает контакт податливее (ODE решает столкновение
      # мягче, за несколько шагов) — первое приближение, не тюнинг torque/
      # velocity/PID щита (те завязаны на тайминг разворота, трогать
      # рискованнее) — проверить живым прогоном, при недостаточном эффекте
      # можно поднять ещё (0.01+) или дополнительно снизить motor_torque.
      material2 "paddle"
      coulombFriction [ {PADDLE_FRICTION} ]
      rollingFriction 0.02 0.02 0.02
      bounce 0
      softCFM 0.005
    }}
    ContactProperties {{
      material2 "chute"
      coulombFriction [ 0.02 ]
      rollingFriction 0.0 0.0 0.0
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
  position {belt_xc - 1.0} -9.5 8.5
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
  translation {plate_xc} 0 -0.05
  name "floor"
  children [
    Shape {{
      appearance PBRAppearance {{ baseColor 0.55 0.57 0.60 roughness 0.95 metalness 0 }}
      geometry DEF FLOOR_BO Box {{ size {PLATE_LENGTH_X} {PLATE_WIDTH_Y} 0.1 }}
    }}
  ]
  boundingObject USE FLOOR_BO
}}
{low}
{ramp}
{high}
{cv_markers_node()}
{mirror_aruco_markers_node(CV_RIG)}
{belt_aruco_markers_node(CV_RIG, SEG_LOW_HEIGHT)}
{roll_cage_node("1", _CAGE, CAGE1_POS)}
{cage_label_node("cage1_label_D", CAGE1_X, "label_D.png")}
{side_border_node("border_shiber1", BORDER1_X_START, BORDER1_X_END, BORDER1_Y_SIGN, SEG_HIGH_HEIGHT)}
{roll_cage_node("2", _CAGE, CAGE2_POS)}
{cage_label_node("cage2_label_C", CAGE2_X, "label_C.png")}
{chute2}
{belt_b}
{side_border_node("border_shiber2", BORDER2_X_START, BORDER2_X_END, BORDER2_Y_SIGN, SEG_HIGH_HEIGHT)}
{accum_deflector_node()}
{accum_descent_node()}
{accum_platform_node()}
{accum_borders_node()}
{demo_panel_node()}
"""
    WORLD_PATH.write_text(world, encoding="utf-8")
    print(f"OK: {WORLD_PATH}")
    print(f"low=[{SEG_LOW_X_START},{SEG_LOW_END_X}] ramp=[{SEG_RAMP_X_START},{SEG_RAMP_END_X}] "
          f"high=[{SEG_HIGH_X_START},{SEG_HIGH_END_X}] cv_rig.x={CV_RIG['x']} spawn_x={SPAWN_X}")


if __name__ == "__main__":
    main()
