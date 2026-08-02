"""Supervisor-контроллер участка сортировки.

Обязанности:
  * HTTP-сервер: веб-панель (web/index.html) + JSON API (spawn/status/reset);
  * спавн объектов на ленте по команде с панели;
  * классификация (этап 1 — KnownTypeClassifier по типу объекта);
  * управление шиберами (логика очереди — sim.router.Router);
  * журнал событий и подтверждение доставки в зону;
  * нештатные ситуации: объект замер у шибера -> встряхивание щита.
"""
import json
import math
import os
import queue
import random
import shutil
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
from controller import Supervisor

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from sim.classifier import KnownTypeClassifier          # noqa: E402
from sim.collision import collision_bounding_node        # noqa: E402
from sim.config import load_layout, load_objects        # noqa: E402
from sim.cv_grid import CELL_CAMERA_MOMENT, assemble_grid, cells_metadata, render_composite  # noqa: E402
from sim.cv_grid_v2 import (                              # noqa: E402
    assemble_grid_v2, build_camera_geom, compute_target_hint_from_top,
)  # Фаза 3.2 (top-локализация+проекция), endpoint /api/cv/moment_grid_v2/
from sim.cv_localization import build_background_model, compute_mask  # noqa: E402
from sim.cv_moments import (                              # noqa: E402
    MomentScheduler, moment_thresholds, threshold_captures,
)
from sim.cv_top_tracking import TopTracker, localize_top_components  # noqa: E402
from sim.router import (                                 # noqa: E402
    CATEGORY_TO_ZONE, DELIVERED, DISPATCHED, MISSED, Router, TrackedObject,
)

WEB_DIR = ROOT / "web"
MESH_DIR = ROOT / "assets" / "meshes"
RECORD_DIR = Path("/tmp/robozon_records")
MAX_EVENTS = 300
FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None
# Settle-задержка (тиков) перед saveImage автоматического захвата момента
# (Фаза 2.2) — НЕ тот же countdown=3, что у редких ручных /api/cv/frame|mask
# (там лишний рендер за один разовый запрос незаметен). Здесь захват идёт
# автоматически 5 раз на каждый объект (на "центр" — 3 камеры сразу), и всё
# время, пока камера enable()'нута, Webots рендерит её на КАЖДОМ тике —
# countdown=3 значит 3 полных рендера 2592x1944 на камеру, хотя используется
# только последний кадр. Пользователь заметил реальные фризы при проезде
# зоны CV — измерено (/api/status "time" vs реальные секунды): rate~0.44
# (в 2+ раза медленнее реального времени) именно в зоне CV против ~0.9-1.0
# до/после. Уменьшено до 1 (минимум, чтобы рендер после enable() не был
# кадром ДО него) — снижает лишний рендер втрое.
MOMENT_CAPTURE_SETTLE_TICKS = 1
# Допуск сопоставления компоненты top-кадра с уже отслеживаемым uid
# (sim/cv_top_tracking.py::TopTracker) — заметно больше смещения объекта за
# один тик (belt_a.speed=1.0 м/с × basicTimeStep=8мс = 8мм, допуск с большим
# запасом на пропущенные кадры/шум детекции), но заметно меньше минимального
# расстояния между объектами на ленте (spawn.min_interval_s=0.5с × 1.0м/с =
# 0.5м) — не должен путать соседей.
TOP_TRACKER_MAX_MATCH_DISTANCE_M = 0.15

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".wasm": "application/wasm",
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".jpg": "image/jpeg",
    ".stl": "application/octet-stream",
    ".data": "application/octet-stream",
    ".mp4": "video/mp4",
}

# Статика: префикс URL -> корневая папка
STATIC_ROUTES = {"/wwi/": WEB_DIR / "wwi", "/meshes/": MESH_DIR, "/records/": RECORD_DIR}


def _jpeg_writer_loop(jobs: "queue.Queue") -> None:
    """Фоновый поток: JPEG-кодирование + запись на диск для автоматического
    захвата моментов (Фаза 2.2). Webots API (enable/getImage/disable) должен
    вызываться из главного потока — здесь принимает уже готовый сырой
    BGRA-буфер (`Camera.getImage()`, снят в главном потоке ДО этого) и делает
    именно ту часть, что раньше делал `saveImage()` синхронно в главном
    цикле (~0.1-0.16с/кадр на 2592x1944) — сама причина остаточных
    подтормаживаний после того, как расчёт маски уже убран из горячего пути
    (см. заметку задачи). Из главного цикла теперь остаётся только дешёвый
    `getImage()` (~5мс).

    `mirror=True` (Фаза 3, зеркальные камеры участвуют в захвате моментов) —
    сразу после записи применяет `_flip_mirror_frame` (тот же обязательный
    вертикальный flip, что и для ручного `/api/cv/frame/<camera>`, см. её
    docstring) — здесь, а не в главном цикле, т.к. Image.open/transpose/save
    тоже дисковый I/O, который не должен блокировать `robot.step()`."""
    while True:
        job = jobs.get()
        if job is None:
            return
        raw, width, height, path, mirror = job
        frame_bgr = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 4))[:, :, :3]
        ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if ok:
            path.write_bytes(buf.tobytes())
            if mirror:
                _flip_mirror_frame(path)


def _flip_mirror_frame(path: Path) -> None:
    """Зеркальные cv_*_mirror камеры прицелены точно на объект, но ОРИЕНТАЦИЯ
    строится с ВОССТАНОВЛЕННЫМ (proper, det=+1) `up` вместо истинного
    отражённого `left` (Webots не может закодировать несобственное
    преобразование поворотом) — см. tools/gen_world.py::_mirror_camera_rotation.
    Из-за этого сырой рендер зеркально перевёрнут ПО ВЕРТИКАЛИ (не по
    горизонтали — восстанавливается именно `up`, который в конвенции осей
    Webots этого рига соответствует ВЕРТИКАЛЬНОЙ оси кадра, см. docstring
    `_camera_frame`) относительно истинного вида через зеркало; этот flip —
    обязательная компенсация для ЛЮБОГО потребителя пикселей mirror-камеры
    (сегментация/реконструкция), не только для визуальной проверки."""
    from PIL import Image

    img = Image.open(path)
    img.transpose(Image.FLIP_TOP_BOTTOM).save(path, quality=90)


def random_orientation() -> tuple:
    """Полностью случайная ориентация (равномерная ось + угол 0..180°) —
    объект может появиться в любом развороте, включая «вверх ногами».
    Любой относительный поворот представим такой парой (ось, угол<=180°),
    так что диапазон покрывает всё SO(3) (без строгой хааровской
    равномерности по углу, но для визуального разнообразия она не нужна).
    """
    x, y, z = random.gauss(0, 1), random.gauss(0, 1), random.gauss(0, 1)
    n = math.sqrt(x * x + y * y + z * z) or 1.0
    return (x / n, y / n, z / n, random.uniform(0, math.pi))


def random_orientation_for(cfg: dict, max_attempts: int = 20) -> tuple:
    """Случайная ориентация под ограничения формы объекта.

    Для цилиндра с осью z (pouf Ø490, plate, helmet — стоят на донышке):
    отклоняем ориентации, где локальная ось z объекта сильно отклонена от
    мировой вертикали — иначе плоский/крупный диск встаёт на ребро и
    выкатывается за борт ленты (pouf под 173.7° уезжал на y=0.78 и
    проваливался, см. ТЗ «исключить появление товара в позиции, приводящей
    стабильно к застреванию»). Порог 45° — допускает заметный наклон
    («лежа на боку чуть под углом»), но не стойку на ребре. Для цилиндров
    axis=x/y (бутылка, цилиндр-каток) и коробов — без фильтра (полная
    случайность, как раньше).
    """
    if not (cfg.get("bounding", {}).get("type") == "cylinder"
            and cfg.get("bounding", {}).get("axis", "z") == "z"):
        return random_orientation()
    bnd = cfg["bounding"]
    # roller: true (шлем) — объект должен катиться, как bottle. Не фильтруем
    # ориентацию — пусть ложится на бок случайно, тогда щит катит его в зону.
    # Без roller (pouf, plate) — фильтр: стоят на донышке, не на ребре.
    if cfg.get("roller"):
        return random_orientation()
    # Порог наклона оси z от вертикали: плоские диски (plate, pouf — height ≤
    # 0.30м) допускают наклон до 30° (не встают на ребро, но дают разнообразие);
    # высокие цилиндры без roller — строже, 5°: стоят прямо на донышке.
    h = bnd.get("height", 0.0)
    max_tilt_deg = 5.0 if h > 0.30 else 30.0
    threshold_cos = math.cos(math.radians(max_tilt_deg))
    for _ in range(max_attempts):
        rot = random_orientation()
        z_local = _rotate_point((0.0, 0.0, 1.0), rot[:3], rot[3])
        if abs(z_local[2]) >= threshold_cos:
            return rot
    # не нашли за max_attempts — берём последнюю (редкий случай, всё равно
    # случайная, просто может оказаться на ребре)
    return rot


def _rotate_point(p: tuple, axis: tuple, angle: float) -> tuple:
    """Поворот точки p вокруг единичной оси axis на angle (формула Родрига)."""
    x, y, z = p
    kx, ky, kz = axis
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    dot = kx * x + ky * y + kz * z
    crossx, crossy, crossz = ky * z - kz * y, kz * x - kx * z, kx * y - ky * x
    return (
        x * cos_a + crossx * sin_a + kx * dot * (1 - cos_a),
        y * cos_a + crossy * sin_a + ky * dot * (1 - cos_a),
        z * cos_a + crossz * sin_a + kz * dot * (1 - cos_a),
    )


def _canonical_points(bnd: dict) -> list:
    """Опорные точки boundingObject в локальной (неповёрнутой) системе —
    те же самые, что строит bounding_node: origin у дна объекта. Для
    цилиндра берём точки на ободах — достаточно для поиска экстремумов
    выпуклой формы (капсула/цилиндр не имеют более дальних точек внутри).
    """
    if bnd["type"] == "box":
        sx, sy, sz = bnd["size"]
        return [(xs, ys, zs)
                for xs in (-sx / 2, sx / 2)
                for ys in (-sy / 2, sy / 2)
                for zs in (0, sz)]
    if bnd["type"] == "cylinder":
        r, h, axis = bnd["radius"], bnd["height"], bnd.get("axis", "z")
        n = 16
        rim = [(r * math.cos(2 * math.pi * i / n), r * math.sin(2 * math.pi * i / n))
               for i in range(n)]
        if axis == "z":
            return [(cx, cy, zs) for cx, cy in rim for zs in (0, h)]
        if axis == "y":
            return [(cx, ys, r + cz) for cx, cz in rim for ys in (-h / 2, h / 2)]
        return [(xs, cx, r + cz) for cx, cz in rim for xs in (-h / 2, h / 2)]
    if bnd["type"] == "sphere":
        r = bnd["radius"]
        n = 12
        pts = []
        for i in range(n):
            theta = math.pi * i / (n - 1)   # 0..pi
            for j in range(n):
                phi = 2 * math.pi * j / n
                pts.append((r * math.sin(theta) * math.cos(phi),
                            r * math.sin(theta) * math.sin(phi),
                            r + r * math.cos(theta)))   # origin у дна
        return pts
    raise ValueError(f"Неизвестный bounding: {bnd['type']}")


def spawn_clearance(bnd: dict, rotation: tuple) -> float:
    """Насколько нужно приподнять origin объекта над лентой, чтобы при данном
    случайном повороте ни одна точка формы не ушла под ленту (для типичных
    «естественных» ориентаций — почти 0, для крайних вроде угла вниз — больше).
    """
    ax, ay, az, angle = rotation
    min_z = min(_rotate_point(p, (ax, ay, az), angle)[2] for p in _canonical_points(bnd))
    return max(0.0, -min_z)


def object_node(uid: int, obj_type: str, cfg: dict, x: float, y: float, z: float,
                rotation: tuple = (0.0, 0.0, 1.0, 0.0), density: float = 200.0) -> str:
    r, g, b = cfg["color"]
    # Геометрия — USE на DEF MESH_<type> из mesh_templates (gen_world.py):
    # STL уже скачан и распарсен при загрузке мира, повторный спавн того же
    # типа не платит за это заново (см. README/обсуждение задержки спавна).
    # bounding: используется ТОЛЬКО как грубая эвристика формы для трения/
    # демпфирования/фильтра ориентации ниже (is_cylinder/is_roller) и для
    # spawn_clearance — реальная форма коллизии больше не строится из него,
    # см. collision_bounding_node()/sim/collision.py (convex decomposition
    # прямо по STL, кэш assets/collision/, tools/prepare_collision.py).
    bnd = cfg["bounding"]
    is_cylinder = bnd["type"] == "cylinder"
    # "Каталка" — цилиндр, лежащий на боку (ось x/y): у него качение вдоль
    # ленты/под шибером — нормальное реалистичное поведение (как бутылка),
    # поэтому трение качения занижено отдельным contactMaterial "round".
    # Цилиндр на оси z (тарелка, пуфик, шлем) стоит/лежит плашмя на донышке —
    # у него так же заниженное трение качения дало бы непредсказуемое качение
    # в любую сторону (тарелка на ребре хаотично крутится и застревает у
    # щита) — для него оставляем обычное трение качения (contactMaterial
    # "default"), но всё равно катится по кромке.
    is_roller = (is_cylinder and bnd.get("axis", "z") != "z") or cfg.get("roller", False)
    if cfg.get("slippery"):
        # Гладкая скорлупа (шлем и т.п.) — снижено именно трение СКОЛЬЖЕНИЯ
        # (contactMaterial "round" снижает только rollingFriction и не
        # спасает объект, прижатый к бортику зоны/направляющей: без запаса
        # по скольжению он там залипает вместо того, чтобы соскользнуть).
        contact_material = "slippery"
    else:
        contact_material = "round" if is_roller else "default"
    # Демпфирование — низкое для ЛЮБОГО цилиндра (обеих осей), не только
    # "катящегося": высокое демпфирование гасит именно ту неустойчивость,
    # что в реальности заставляет предмет, вставший на ребро, завалиться —
    # без него пуфик может неестественно долго ехать на ребре под углом, не
    # опрокидываясь. Для коробов демпфирование выше — стабильность важнее.
    lin_damp, ang_damp = (0.05, 0.01) if is_cylinder else (0.1, 0.05)
    rx, ry, rz, angle = rotation
    return f"""DEF OBJ_{uid} Solid {{
  translation {x} {y} {z}
  rotation {rx} {ry} {rz} {angle}
  name "obj_{uid}"
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
    density {density}
    damping Damping {{ linear {lin_damp} angular {ang_damp} }}
  }}
}}"""


class Api:
    """Состояние, разделяемое между HTTP-потоком и циклом симуляции."""

    def __init__(self, objects_cfg: dict):
        self.objects_cfg = objects_cfg
        self.spawn_queue: queue.Queue = queue.Queue()
        self.manual_queue: queue.Queue = queue.Queue()   # элементы: "B"/"C"/"D" или "auto"
        self.reset_requested = threading.Event()
        self.screenshot_requested = threading.Event()
        self.screenshot_done = threading.Event()
        self.screenshot_path = Path("/tmp/robozon_view.jpg")
        # По слоту НА КАМЕРУ (не общий) — конкурентные запросы к разным cv_*
        # камерам (снять со всех одновременно, пока объект в зоне CV) не должны
        # гоняться за одно состояние и получать чужой кадр.
        self.cv_frame_lock = threading.Lock()
        self.cv_frame_requests: dict = {}   # camera_name -> {"done", "ok", "pending"}
        self.cv_frame_dir = Path("/tmp/robozon_cv_frames")
        self.cv_camera_names: set[str] = set()
        self.cv_mirror_names: set[str] = set()
        # Модель фона (Фаза 2.1, прямые камеры; Фаза 3 — и зеркальные), в
        # памяти процесса (пересчитывается при старте и на /api/reset, см.
        # Sorter).
        self.cv_backgrounds: dict = {}   # camera_name -> BackgroundModel
        self.bg_subtraction_cfg: dict = {}   # cv_rig.background_subtraction (см. Sorter.__init__)
        self.cv_rig_cfg: dict = {}   # весь cv_rig (см. Sorter.__init__) — нужен sim/cv_grid.py::assemble_grid
        # Фаза 2.2/3 — автозахват моментов+сборка сетки ВЫКЛЮЧЕНЫ по
        # умолчанию: пока Фаза 4 (сегментация) не подключена, некому
        # потреблять эти кадры при обычной работе стенда — включать явно
        # (POST /api/cv/moments_enabled) только на время тестирования/снятия
        # офлайн-отчёта (tools/capture_moments.py/assemble_grids.py делают
        # это сами).
        # Обычное чтение/запись bool между потоками — атомарно под GIL,
        # отдельная блокировка не нужна (тот же паттерн, что manual_mode).
        self.moments_enabled = False
        self.calibrate_bg_requested = threading.Event()
        self.calibrate_bg_done = threading.Event()
        self.record_start_requested = threading.Event()
        self.record_stop_requested = threading.Event()
        self.lock = threading.Lock()
        self.status: dict = {"time": 0, "objects": [], "events": [], "stats": {}}

    def snapshot(self) -> bytes:
        with self.lock:
            return json.dumps(self.status, ensure_ascii=False).encode()


def _build_moment_grid(api: Api, uid: int) -> bool:
    """Ленивая сборка сетки 9 ракурсов (Фаза 3, `sim/cv_grid.py`) — тот же
    паттерн, что `moment_mask` выше: читает уже сохранённые
    `moment_<uid>_<camera>_<moment>[_prev].jpg` и калиброванные фоны с диска,
    кэширует композит (`moment_<uid>_grid.png`) + метаданные
    (`moment_<uid>_grid.json`). Возвращает False, если хотя бы один из 9
    обязательных object-кадров ещё не захвачен (момент ещё не наступил).

    Замеряет и сохраняет тайминги сборки (`meta["timings_s"]`) — та же логика,
    что уже оправдала себя в Фазе 2.2 (компонент `compute_mask` был причиной
    реальных фризов, см. docstring `_jpeg_writer_loop`): здесь `assemble_grid`
    зовёт `compute_mask` до 9 раз (по одной на ячейку с откалиброванным
    фоном), поэтому важно знать раздельно I/O (чтение JPEG с диска) и сам
    расчёт кропа/масштаба, не только суммарное время запроса."""
    t0 = time.perf_counter()
    object_frames: dict[str, np.ndarray] = {}
    prev_frames: dict[str, np.ndarray] = {}
    for cell_name, (camera, moment) in CELL_CAMERA_MOMENT.items():
        path = api.cv_frame_dir / f"moment_{uid}_{camera}_{moment}.jpg"
        if not path.is_file():
            return False
        object_frames[cell_name] = cv2.imread(str(path))
        prev_path = api.cv_frame_dir / f"moment_{uid}_{camera}_{moment}_prev.jpg"
        if prev_path.is_file():
            prev_frames[cell_name] = cv2.imread(str(prev_path))

    background_frames: dict[str, np.ndarray] = {}
    for camera in {cam for cam, _moment in CELL_CAMERA_MOMENT.values()}:
        bg_path = api.cv_frame_dir / f"{camera}_background.jpg"
        if bg_path.is_file():
            background_frames[camera] = cv2.imread(str(bg_path))

    # X объекта НА МОМЕНТ КАЖДОГО кадра — из уже залогированных событий
    # cv_moment (см. Sorter._advance_moment_capture), не из физики напрямую:
    # этот код выполняется в потоке HTTP-обработчика, где Supervisor
    # недоступен (тот же принцип, что moment_mask выше).
    positions_x: dict[str, float] = {}
    with api.lock:
        events = list(api.status.get("events", []))
    for cell_name, (camera, moment) in CELL_CAMERA_MOMENT.items():
        for e in events:
            if e.get("event") == "cv_moment" and e.get("uid") == uid \
                    and e.get("camera") == camera and e.get("moment") == moment:
                positions_x[cell_name] = e.get("x")
                break
    t1 = time.perf_counter()

    rig = api.cv_rig_cfg
    grid_cfg = rig["grid"]
    result = assemble_grid(object_frames, prev_frames, background_frames, positions_x, rig,
                            grid_cfg["cell_px"], grid_cfg["sam3_max_dim"], grid_cfg["crop_margin_factor"])
    t2 = time.perf_counter()
    cv2.imwrite(str(api.cv_frame_dir / f"moment_{uid}_grid.png"), render_composite(result.cells))
    t3 = time.perf_counter()
    meta = {
        "scale": result.scale,
        "cells": cells_metadata(result.cells),
        "timings_s": {
            "io_load": round(t1 - t0, 4),          # чтение 9(+prev+фон) JPEG с диска
            "assemble_grid": round(t2 - t1, 4),    # crop/compute_mask×≤9/resize — сам расчёт кропа
            "composite_write": round(t3 - t2, 4),  # плитка-превью + imwrite
            "total": round(t3 - t0, 4),
        },
    }
    (api.cv_frame_dir / f"moment_{uid}_grid.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return True


def _build_moment_grid_v2(api: Api, uid: int) -> bool:
    """Ленивая сборка сетки 9 ракурсов Фазой 3.2 (sim/cv_grid_v2.py::
    assemble_grid_v2 — top-локализация + проекция в остальные камеры). Тот же
    паттерн кэширования/таймингов, что _build_moment_grid, но отдельные файлы
    moment_<uid>_grid_v2.{png,json}, чтобы не затирать эталон Фазы 3.1 (можно
    сравнивать). positions_x НЕ нужен v2 (локализация по top, не по физике)."""
    t0 = time.perf_counter()
    object_frames: dict[str, np.ndarray] = {}
    prev_frames: dict[str, np.ndarray] = {}
    for cell_name, (camera, moment) in CELL_CAMERA_MOMENT.items():
        path = api.cv_frame_dir / f"moment_{uid}_{camera}_{moment}.jpg"
        if not path.is_file():
            return False
        object_frames[cell_name] = cv2.imread(str(path))
        prev_path = api.cv_frame_dir / f"moment_{uid}_{camera}_{moment}_prev.jpg"
        if prev_path.is_file():
            prev_frames[cell_name] = cv2.imread(str(prev_path))

    background_frames: dict[str, np.ndarray] = {}
    for camera in {cam for cam, _moment in CELL_CAMERA_MOMENT.values()}:
        bg_path = api.cv_frame_dir / f"{camera}_background.jpg"
        if bg_path.is_file():
            background_frames[camera] = cv2.imread(str(bg_path))
    t1 = time.perf_counter()

    rig = api.cv_rig_cfg
    grid_cfg = rig["grid"]
    # target_hint из независимой локализации КАЖДОГО top-момента (не только
    # center_top на все 9 ячеек) — см. заметку задачи "3D-реконструкция
    # объекта по кропам сетки ракурсов...", находка про cylinder на side.
    target_hint = compute_target_hint_from_top(object_frames, background_frames, rig)
    result = assemble_grid_v2(object_frames, prev_frames, background_frames, rig,
                               grid_cfg["cell_px"], grid_cfg["sam3_max_dim"],
                               grid_cfg["crop_margin_factor"], target_hint=target_hint,
                               narrow_strip=False)
    t2 = time.perf_counter()
    cv2.imwrite(str(api.cv_frame_dir / f"moment_{uid}_grid_v2.png"),
                render_composite(result.cells))
    # Композит ФОНА — те же окна/масштаб кропа, что у объекта (иначе
    # object_crop-background_crop не даст силуэт при вычитании), нужен для
    # 3D-реконструкции (Фаза 5, см. заметку задачи "3D-реконструкция объекта
    # по кропам сетки ракурсов..."). Пропускается, если фон недоступен хотя
    # бы для одной камеры (background_frames неполный — редкий случай, до
    # калибровки фона).
    has_bg = all(c.background_crop is not None for c in result.cells.values())
    if has_bg:
        cv2.imwrite(str(api.cv_frame_dir / f"moment_{uid}_grid_v2_bg.png"),
                    render_composite(result.cells, field="background_crop"))
    t3 = time.perf_counter()
    meta = {
        "scale": result.scale,
        "has_background_grid": has_bg,
        "cells": cells_metadata(result.cells),
        "timings_s": {
            "io_load": round(t1 - t0, 4),
            "assemble_grid_v2": round(t2 - t1, 4),
            "composite_write": round(t3 - t2, 4),
            "total": round(t3 - t0, 4),
        },
    }
    (api.cv_frame_dir / f"moment_{uid}_grid_v2.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return True


def make_handler(api: Api):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # тихий сервер
            pass

        def _send(self, code: int, body: bytes, ctype: str = "application/json; charset=utf-8"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def _serve_static(self) -> bool:
            for prefix, root in STATIC_ROUTES.items():
                if not self.path.startswith(prefix):
                    continue
                rel = self.path[len(prefix):].split("?")[0]
                target = (root / rel).resolve()
                if root.resolve() not in target.parents or not target.is_file():
                    self._send(404, b'{"error": "not found"}')
                    return True
                ctype = CONTENT_TYPES.get(target.suffix, "application/octet-stream")
                self._send(200, target.read_bytes(), ctype)
                return True
            return False

        def do_GET(self):
            if self._serve_static():
                pass
            elif self.path in ("/", "/index.html"):
                page = (WEB_DIR / "index.html").read_bytes()
                self._send(200, page, "text/html; charset=utf-8")
            elif self.path == "/api/objects":
                items = [{"type": t, "label": c["label"], "category": c["category"],
                          "zone": CATEGORY_TO_ZONE[c["category"]], "color": c["color"]}
                         for t, c in api.objects_cfg.items()]
                self._send(200, json.dumps(items, ensure_ascii=False).encode())
            elif self.path == "/api/status":
                self._send(200, api.snapshot())
            elif self.path == "/api/screenshot":
                api.screenshot_done.clear()
                api.screenshot_requested.set()
                if api.screenshot_done.wait(timeout=3) and api.screenshot_path.exists():
                    self._send(200, api.screenshot_path.read_bytes(), "image/jpeg")
                else:
                    self._send(503, b'{"error": "screenshot failed"}')
            elif self.path.startswith("/api/cv/frame/"):
                cam_name = self.path[len("/api/cv/frame/"):].split("?")[0]
                if cam_name not in api.cv_camera_names:
                    self._send(404, b'{"error": "unknown camera"}')
                    return
                with api.cv_frame_lock:
                    entry = api.cv_frame_requests.setdefault(
                        cam_name, {"done": threading.Event()})
                    entry["done"].clear()
                    entry["ok"] = False
                    entry["pending"] = True
                    entry["mode"] = "frame"
                if entry["done"].wait(timeout=3) and entry["ok"]:
                    path = api.cv_frame_dir / f"{cam_name}.jpg"
                    self._send(200, path.read_bytes(), "image/jpeg")
                else:
                    self._send(503, b'{"error": "cv frame failed"}')
            elif self.path.startswith("/api/cv/mask/"):
                # Фаза 2.1 — только прямые камеры (top/diag/side): зеркальные
                # маски (top_mirror/diag_mirror) не входят в эту задачу, см.
                # заметку "Фоновая субтракция и маска объекта на CV-камерах".
                cam_name = self.path[len("/api/cv/mask/"):].split("?")[0]
                if cam_name not in api.cv_camera_names or cam_name in api.cv_mirror_names:
                    self._send(404, b'{"error": "unknown or unsupported camera"}')
                    return
                if cam_name not in api.cv_backgrounds:
                    self._send(503, b'{"error": "background not calibrated"}')
                    return
                with api.cv_frame_lock:
                    entry = api.cv_frame_requests.setdefault(
                        cam_name, {"done": threading.Event()})
                    entry["done"].clear()
                    entry["ok"] = False
                    entry["pending"] = True
                    entry["mode"] = "mask"
                if entry["done"].wait(timeout=3) and entry["ok"]:
                    path = api.cv_frame_dir / f"{cam_name}_mask.png"
                    self._send(200, path.read_bytes(), "image/png")
                else:
                    self._send(503, b'{"error": "cv mask failed"}')
            elif self.path.startswith("/api/cv/moment_frame/"):
                # Фаза 2.2 — отдаёт уже сохранённый JPEG автоматического
                # захвата момента (см. Sorter._advance_moment_capture):
                # /api/cv/moment_frame/<uid>/<camera>/<moment>. Не рендерит по
                # запросу (в отличие от /api/cv/frame выше) — либо файл уже
                # готов (захват прошёл раньше, асинхронно от этого запроса),
                # либо 404 (момент ещё не наступил/объект потерян).
                parts = self.path[len("/api/cv/moment_frame/"):].split("?")[0].split("/")
                if len(parts) != 3 or not parts[0].isdigit():
                    self._send(400, b'{"error": "expected /uid/camera/moment"}')
                    return
                uid_s, cam_name, moment = parts
                if cam_name not in api.cv_camera_names or moment not in ("start", "center", "end"):
                    self._send(404, b'{"error": "unknown camera or moment"}')
                    return
                path = api.cv_frame_dir / f"moment_{uid_s}_{cam_name}_{moment}.jpg"
                if path.is_file():
                    self._send(200, path.read_bytes(), "image/jpeg")
                else:
                    self._send(404, b'{"error": "moment not captured (yet)"}')
            elif self.path.startswith("/api/cv/moment_mask/"):
                # Фаза 2.2 — маска считается ЛЕНИВО (не автоматически при
                # захвате): compute_mask на 2592x1944 стоит 0.15-0.34с — на
                # каждый из 7 моментов/объект это давало заметные фризы
                # симуляции (замечено пользователем, замерено — см. заметку
                # задачи). Здесь она не нужна сразу — деталь детекции момента
                # (физическая X, уже дёшева) отделена от точной сегментации,
                # которая пригождается только для офлайн-сверки. Расчёт идёт
                # прямо в потоке HTTP-обработчика (НЕ через Webots API,
                # camera/robot недоступны вне главного потока) — читает уже
                # сохранённый JPEG с диска (та же связка, что /api/cv/mask/
                # выше), НЕ трогает цикл симуляции. Результат кэшируется на
                # диск при первом запросе — повторные запросы читают файл.
                parts = self.path[len("/api/cv/moment_mask/"):].split("?")[0].split("/")
                if len(parts) != 3 or not parts[0].isdigit():
                    self._send(400, b'{"error": "expected /uid/camera/moment"}')
                    return
                uid_s, cam_name, moment = parts
                if (cam_name not in api.cv_camera_names or cam_name in api.cv_mirror_names
                        or moment not in ("start", "center", "end")):
                    self._send(404, b'{"error": "unknown/unsupported camera or moment"}')
                    return
                mask_path = api.cv_frame_dir / f"moment_{uid_s}_{cam_name}_{moment}_mask.png"
                if not mask_path.is_file():
                    frame_path = api.cv_frame_dir / f"moment_{uid_s}_{cam_name}_{moment}.jpg"
                    bg = api.cv_backgrounds.get(cam_name)
                    if not frame_path.is_file() or bg is None:
                        self._send(404, b'{"error": "frame not captured or background not calibrated"}')
                        return
                    frame = cv2.imread(str(frame_path))
                    bs = api.bg_subtraction_cfg
                    mask, _bbox = compute_mask(
                        frame, bg, bs["h_threshold"], bs["s_threshold"],
                        bs["min_area_px"], bs["morph_kernel"], v_threshold=bs.get("v_threshold"))
                    cv2.imwrite(str(mask_path), mask)
                self._send(200, mask_path.read_bytes(), "image/png")
            elif self.path.startswith("/api/cv/moment_grid/"):
                # Фаза 3 — сетка 9 ракурсов (sim/cv_grid.py), тот же ленивый+
                # кэширующий паттерн, что moment_mask выше: /uid, без камеры/
                # момента (сетка — по ВСЕМ 9 сразу). Composite PNG — только
                # для визуальной проверки; данные для Фазы 4/5 — в
                # moment_grid_meta ниже (метаданные crop/scale на ячейку).
                uid_s = self.path[len("/api/cv/moment_grid/"):].split("?")[0]
                if not uid_s.isdigit():
                    self._send(400, b'{"error": "expected /uid"}')
                    return
                uid = int(uid_s)
                png_path = api.cv_frame_dir / f"moment_{uid}_grid.png"
                if not png_path.is_file() and not _build_moment_grid(api, uid):
                    self._send(404, b'{"error": "not all 9 moments captured yet"}')
                    return
                self._send(200, png_path.read_bytes(), "image/png")
            elif self.path.startswith("/api/cv/moment_grid_meta/"):
                uid_s = self.path[len("/api/cv/moment_grid_meta/"):].split("?")[0]
                if not uid_s.isdigit():
                    self._send(400, b'{"error": "expected /uid"}')
                    return
                uid = int(uid_s)
                json_path = api.cv_frame_dir / f"moment_{uid}_grid.json"
                if not json_path.is_file() and not _build_moment_grid(api, uid):
                    self._send(404, b'{"error": "not all 9 moments captured yet"}')
                    return
                self._send(200, json_path.read_bytes())
            elif self.path.startswith("/api/cv/moment_grid_v2/"):
                # Фаза 3.2 — top-локализация+проекция (sim/cv_grid_v2.py), те же
                # файлы moment_<uid>_grid_v2.{png,json}, что у moment_grid, но с
                # суффиксом _v2 — чтобы сравнивать с Фазой 3.1 не затирая её.
                uid_s = self.path[len("/api/cv/moment_grid_v2/"):].split("?")[0]
                if not uid_s.isdigit():
                    self._send(400, b'{"error": "expected /uid"}')
                    return
                uid = int(uid_s)
                png_path = api.cv_frame_dir / f"moment_{uid}_grid_v2.png"
                if not png_path.is_file() and not _build_moment_grid_v2(api, uid):
                    self._send(404, b'{"error": "not all 9 moments captured yet"}')
                    return
                self._send(200, png_path.read_bytes(), "image/png")
            elif self.path.startswith("/api/cv/moment_grid_v2_meta/"):
                uid_s = self.path[len("/api/cv/moment_grid_v2_meta/"):].split("?")[0]
                if not uid_s.isdigit():
                    self._send(400, b'{"error": "expected /uid"}')
                    return
                uid = int(uid_s)
                json_path = api.cv_frame_dir / f"moment_{uid}_grid_v2.json"
                if not json_path.is_file() and not _build_moment_grid_v2(api, uid):
                    self._send(404, b'{"error": "not all 9 moments captured yet"}')
                    return
                self._send(200, json_path.read_bytes())
            elif self.path.startswith("/api/cv/moment_grid_v2_bg/"):
                # Композит ФОНА (те же окна/масштаб, что moment_grid_v2) —
                # для 3D-реконструкции (Фаза 5): вычитание object_crop-
                # background_crop даёт силуэт. 404, если фон был недоступен
                # хотя бы для одной камеры при сборке (см. has_background_grid
                # в meta), не только если моменты ещё не захвачены.
                uid_s = self.path[len("/api/cv/moment_grid_v2_bg/"):].split("?")[0]
                if not uid_s.isdigit():
                    self._send(400, b'{"error": "expected /uid"}')
                    return
                uid = int(uid_s)
                png_path = api.cv_frame_dir / f"moment_{uid}_grid_v2_bg.png"
                if not png_path.is_file() and not _build_moment_grid_v2(api, uid):
                    self._send(404, b'{"error": "not all 9 moments captured yet"}')
                    return
                if not png_path.is_file():
                    self._send(404, b'{"error": "background grid unavailable for this capture"}')
                    return
                self._send(200, png_path.read_bytes(), "image/png")
            elif self.path == "/api/record/list":
                RECORD_DIR.mkdir(parents=True, exist_ok=True)
                files = sorted((f.name for f in RECORD_DIR.glob("*.mp4")), reverse=True)
                self._send(200, json.dumps(files).encode())
            else:
                self._send(404, b'{"error": "not found"}')

        def do_POST(self):
            if self.path == "/api/spawn":
                length = int(self.headers.get("Content-Length", 0))
                try:
                    data = json.loads(self.rfile.read(length) or b"{}")
                    obj_type = data["type"]
                    if obj_type not in api.objects_cfg:
                        raise KeyError(obj_type)
                    # rotation:"identity" — детерминированный (не случайный)
                    # спавн для офлайн-проверки CV (см. tools/report_cv_masks.py):
                    # даёт предсказуемую ориентацию для сравнения габаритов маски
                    # с истинными размерами объекта. rotation:[rx,ry,rz,angle] —
                    # явный кватернион (ось+угол) для ВОСПРОИЗВОДИМОГО повтора
                    # конкретного случайного розыгрыша (см. событие "spawn" в
                    # логе — там теперь пишется полный rotation) при A/B-сравнении
                    # разных порогов срабатывания шибера на ОДНОМ и том же
                    # физическом старте, без перезапуска симуляции. По умолчанию
                    # (без поля) — прежнее поведение, случайная ориентация.
                    rotation_mode = data.get("rotation")
                    if isinstance(rotation_mode, list):
                        if len(rotation_mode) != 4 or not all(
                                isinstance(v, (int, float)) for v in rotation_mode):
                            raise ValueError(rotation_mode)
                        rotation_mode = tuple(float(v) for v in rotation_mode)
                    elif rotation_mode not in (None, "identity"):
                        raise ValueError(rotation_mode)
                except (json.JSONDecodeError, KeyError, ValueError):
                    self._send(400, b'{"error": "bad type"}')
                    return
                api.spawn_queue.put((obj_type, rotation_mode))
                self._send(200, b'{"queued": true}')
            elif self.path == "/api/cv/calibrate_background":
                api.calibrate_bg_done.clear()
                api.calibrate_bg_requested.set()
                if api.calibrate_bg_done.wait(timeout=5):
                    self._send(200, b'{"calibrated": true}')
                else:
                    self._send(503, b'{"error": "calibration timeout"}')
            elif self.path == "/api/reset":
                api.reset_requested.set()
                self._send(200, b'{"reset": true}')
            elif self.path == "/api/cv/moments_enabled":
                # Фаза 2.2 — включает/выключает автозахват старт/центр/конец
                # (см. Api.moments_enabled). Выключено по умолчанию — обычная
                # работа стенда не должна платить за захват, которым сейчас
                # никто не пользуется (Фаза 3 не подключена). Включать явно
                # перед тестированием/офлайн-отчётом (tools/capture_moments.py
                # делает это сама).
                length = int(self.headers.get("Content-Length", 0))
                try:
                    data = json.loads(self.rfile.read(length) or b"{}")
                    enabled = bool(data["enabled"])
                except (json.JSONDecodeError, KeyError, ValueError):
                    self._send(400, b'{"error": "expected {\\"enabled\\": bool}"}')
                    return
                api.moments_enabled = enabled
                self._send(200, json.dumps({"moments_enabled": enabled}).encode())
            elif self.path == "/api/route":
                length = int(self.headers.get("Content-Length", 0))
                try:
                    data = json.loads(self.rfile.read(length) or b"{}")
                    zone = data["zone"]
                    if zone not in ("B", "C", "D"):
                        raise KeyError(zone)
                except (json.JSONDecodeError, KeyError):
                    self._send(400, b'{"error": "bad zone"}')
                    return
                api.manual_queue.put(zone)
                self._send(200, b'{"queued": true}')
            elif self.path == "/api/auto":
                api.manual_queue.put("auto")
                self._send(200, b'{"auto": true}')
            elif self.path == "/api/record/start":
                if not FFMPEG_AVAILABLE:
                    self._send(409, b'{"error": "ffmpeg not available"}')
                    return
                api.record_start_requested.set()
                self._send(200, b'{"queued": true}')
            elif self.path == "/api/record/stop":
                api.record_stop_requested.set()
                self._send(200, b'{"queued": true}')
            else:
                self._send(404, b'{"error": "not found"}')

    return Handler


class Sorter:
    def __init__(self):
        self.robot = Supervisor()
        self.timestep = int(self.robot.getBasicTimeStep())
        self.layout = load_layout()
        self.density = self.layout["physics"]["object_density_kg_m3"]
        self.objects_cfg = load_objects()
        self.classifier = KnownTypeClassifier(self.objects_cfg)

        dv = self.layout["diverters"]
        self.dv = dv
        # x-позиции шиберов для защиты от перехвата (router._other_zone_object_near_divert)
        divert_positions = {name: pos["x"] for name, pos in dv["positions"].items()}
        self.router = Router(
            trigger_x=dv["trigger_x"],
            zones=self.layout["zones"],
            release_timeout_s=dv["release_timeout_s"],
            trigger_x_per_zone=dv.get("trigger_x_per_zone", {}),
            divert_positions=divert_positions,
            divert_safe_radius=dv.get("divert_safe_radius", 0.0),
            release_enter={zone: tuple(spec) for zone, spec in dv.get("release_enter", {}).items()},
        )

        # Шиберы: имя зоны -> (мотор, датчик, рабочий угол)
        self.paddles = {}
        for name, pos in dv["positions"].items():
            motor = self.robot.getDevice(f"paddle_{name}")
            sensor = self.robot.getDevice(f"paddle_{name}_sensor")
            sensor.enable(self.timestep)
            motor.setVelocity(dv["motor_velocity"])
            motor.setPosition(0.0)
            deploy = -pos["side"] * dv["deploy_angle"]
            self.paddles[name] = {"motor": motor, "sensor": sensor, "deploy": deploy}

        # Обзорная камера — только по запросу /api/screenshot: постоянный
        # софтверный рендер 1280x720 съедает половину реального времени
        self.camera = self.robot.getDevice("overview")
        self._cam_countdown = -1

        # CV-риг: те же 5 устройств, что gen_world.py::cv_camera_node положил в мир
        # (cv_<name>), по конфигу cv_rig.cameras — постоянно выключены, включаются
        # по запросу /api/cv/frame/<camera>, как overview для /api/screenshot.
        cv = self.layout["cv_rig"]
        self.cv_cameras = {
            name: {"device": self.robot.getDevice(f"cv_{name}"),
                   "mirror": bool(cfg.get("mirror", False))}
            for name, cfg in cv["cameras"].items()
        }
        self._cv_frame_countdowns: dict = {}   # camera_name -> счётчик кадров до saveImage
        self.bg_subtraction_cfg = cv["background_subtraction"]
        self._bg_calib_countdowns: dict = {}   # camera_name -> счётчик кадров (только на время калибровки)
        self._bg_calib_active = False

        # Детектор моментов старт/центр/конец (Фаза 2.2) — отдельная от
        # cv_frame_requests/_cv_frame_countdowns очередь захвата: та обслуживает
        # ручные/отладочные запросы с панели (по одному кадру на камеру за раз),
        # эта — авто-триггер по физической X-позиции, независимо для top/side
        # (diag — только момент "центр"), см. sim/cv_moments.py. Раздельные
        # очереди, чтобы отладочный /api/cv/frame/<camera> не конфликтовал за
        # состояние камеры с автоматическим захватом моментов.
        self.moment_thresholds = moment_thresholds(cv)
        self.moment_captures = threshold_captures()
        self.moment_scheduler = MomentScheduler(self.moment_thresholds, self.moment_captures)
        self._moment_cameras = sorted({cam for pairs in self.moment_captures.values()
                                        for cam, _ in pairs})   # top, side, diag

        # CV-триггер по top-локализации (замена физики, часть 1 задачи
        # "CV-триггер момента по top-локализации") — вместо node.getPosition()
        # источником X для moment_scheduler.step() служит положение объекта,
        # восстановленное по кадру ВЫДЕЛЕННОЙ низкоразрешающей камеры
        # "cv_top_track" (см. cv_rig.cameras.top_track в layout.yaml) — та же
        # позиция/ориентация, что "top", но узкое native-разрешение. Не
        # переиспользуем полноразмерную "top" (та осталась как раньше —
        # включается кратко только на сам момент снимка сетки): постоянный
        # enable() полноразмерной камеры даёт ~11x просадку realtime-factor
        # (rate 0.92->0.083, живой замер) — `Camera.getImage()` в Python не
        # кэш, а запрос через границу процессов НА КАЖДЫЙ вызов (см. заметку
        # задачи, раздел про исследование производительности). top_track
        # включена ПОСТОЯННО (не по порогу, как остальные cv_*) — см.
        # _advance_top_tracking, иначе не по чему обнаружить сам момент
        # пересечения порога.
        self.top_geom = build_camera_geom(
            "top_track", {**cv, "resolution": cv["cameras"]["top_track"]["resolution"]})
        self.top_tracker = TopTracker(max_match_distance_m=TOP_TRACKER_MAX_MATCH_DISTANCE_M)
        # bg_subtraction_cfg.min_area_px откалиброван под full-res "top"
        # (2592 строк, 1.04мм/px) — на native-разрешении top_track (1296
        # строк, 2.08мм/px) та же ФИЗИЧЕСКАЯ площадь объекта занимает в 4 раза
        # (квадрат отношения линейных масштабов) меньше пикселей: ручка
        # (13×148мм) — ~444px на top_track против ~1776px на full-res,
        # практически на пороге min_area_px=400 (0 срабатываний живым
        # прогоном — найдено при живой проверке этой задачи). Пересчитанный
        # порог сохраняет тот же физический минимум обнаруживаемого объекта.
        _track_scale_ratio = (cv["resolution"][0]
                               / cv["cameras"]["top_track"]["resolution"][0])
        self._top_track_bs_cfg = {**self.bg_subtraction_cfg,
                                   "min_area_px": self.bg_subtraction_cfg["min_area_px"]
                                   / (_track_scale_ratio ** 2)}
        self._top_track_bg_frame: np.ndarray | None = None   # RAW BGR-фон top_track (не BackgroundModel)
        self._top_track_continuous_active = False
        self._top_track_settle_countdown = -1
        self._moment_queue: list = []          # [(uid, camera, moment), ...] FIFO по каждой камере
        self._moment_active: dict = {}         # camera -> {"uid", "moment", "countdown"}
        # JPEG-кодирование+запись автозахвата — в фоновом потоке (см.
        # _jpeg_writer_loop), не в главном цикле Webots: getImage() в
        # главном потоке остаётся (обязателен для Webots API), но сам
        # кодек+диск больше не блокируют шаг симуляции.
        self._moment_jpeg_jobs: queue.Queue = queue.Queue()
        threading.Thread(target=_jpeg_writer_loop, args=(self._moment_jpeg_jobs,), daemon=True).start()

        self.api = Api(self.objects_cfg)
        self.api.cv_camera_names = set(self.cv_cameras)
        self.api.cv_mirror_names = {n for n, i in self.cv_cameras.items() if i["mirror"]}
        self.api.bg_subtraction_cfg = self.bg_subtraction_cfg
        self.api.cv_rig_cfg = cv
        self.api.cv_frame_dir.mkdir(parents=True, exist_ok=True)
        # uid_seq (ниже) стартует с 0 при КАЖДОМ перезапуске Webots — без этой
        # очистки oставшиеся с прошлого запуска moment_<uid>_*.* (в т.ч. уже
        # собранные grid_v2.png/json/bg.png) молча "затеняют" свежий захват
        # при переиспользовании того же номера uid: `_build_moment_grid_v2`/
        # `_build_moment_grid` пропускают пересборку, если файл уже существует
        # на диске (кэш по имени, не по содержимому) — найдено живым прогоном
        # (сетка для нового объекта показывала ЧУЖОЙ, из прошлого запуска
        # Webots с тем же uid), см. заметку задачи "3D-реконструкция объекта
        # по кропам сетки ракурсов...". Фоновые *_background.jpg НЕ трогаем
        # (per-камера, не per-uid, калибровка всё равно перезапишет их ниже).
        for stale in self.api.cv_frame_dir.glob("moment_*"):
            stale.unlink()
        # Фон снят на пустой ленте сразу после старта — до открытия HTTP-панели
        # спавн физически невозможен (см. self.api ниже), так что первая
        # калибровка гарантированно застаёт пустую ленту.
        self.api.calibrate_bg_requested.set()
        self.nodes: dict = {}          # uid -> узел Webots
        self.uid_seq = 0
        self.last_spawn_time = -1e9
        self.events: list = []
        self.current_target = "C"      # ручной режим: выбранная цель (оба шибера убраны по умолчанию)
        self.manual_mode = False       # ручное управление шиберами (см. /api/route, /api/auto)
        self._stall: dict = {}         # uid -> {"pos", "since", "wiggle_until"} — по каждому лотку своя
        self._frozen: set = set()      # uid товаров в C/D с отключённой физикой (см. _freeze)
        self._last_pos: dict = {}      # uid -> последняя известная позиция для _frozen
        self._retire_at: dict = {}     # uid -> sim-время, когда можно физически node.remove() (см. _dispose)
        self.record_state = "idle"     # idle -> recording -> encoding -> ready/failed
        self.record_file: str | None = None

        # self._warmup()  # временно отключено — проверяем, не из-за него ли
        # зависает диалог "Opening world file" на Windows (см. переписку)

        port = self.layout["http"]["port"]
        server = ThreadingHTTPServer(("0.0.0.0", port), make_handler(self.api))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        print(f"[sorter] панель управления: http://localhost:{port}/")

    # -- прогрев --------------------------------------------------------------
    def _warmup(self):
        """Первый спавн товара в сессии заметно медленнее следующих — тормозит
        не физика, а рендер: геометрия/PBRAppearance каждого типа грузится в
        GPU лениво, при первом появлении в кадре, а не при генерации мира
        (там только парсится VRML). На демо для жюри это читается как «не
        работает», хотя это разовая задержка на весь сеанс — поэтому прогреваем
        по одному экземпляру каждого типа здесь, ДО открытия HTTP-панели
        (запрос спавна физически невозможен раньше, чем сервер начнёт слушать
        порт), а не дожидаемся первого реального клика."""
        root_children = self.robot.getRoot().getField("children")
        warm_uids = []
        for i, (obj_type, cfg) in enumerate(self.objects_cfg.items()):
            uid = 900_000 + i
            node_str = object_node(uid, obj_type, cfg, -50.0, i * 2.0, 2.0, density=self.density)
            root_children.importMFNodeFromString(-1, node_str)
            warm_uids.append(uid)
        for _ in range(3):
            self.robot.step(self.timestep)
        for uid in warm_uids:
            node = self.robot.getFromDef(f"OBJ_{uid}")
            if node:
                node.remove()
        self.robot.step(self.timestep)

    # -- события ------------------------------------------------------------
    def log(self, event: str, **details):
        entry = {"t": round(self.robot.getTime(), 2), "event": event, **details}
        self.events.append(entry)
        del self.events[:-MAX_EVENTS]
        print(f"[sorter] {entry}")

    # -- шиберы -------------------------------------------------------------
    def _paddle_deploy(self, zone: str):
        p = self.paddles.get(zone)
        if p:
            p["motor"].setPosition(p["deploy"])

    def _paddle_retract(self, zone: str):
        p = self.paddles.get(zone)
        if p:
            p["motor"].setPosition(0.0)

    def set_route(self, zone: str):
        """Ручной режим: развернуть один лоток целевой зоны, остальные убрать."""
        self.current_target = zone
        for name in self.paddles:
            (self._paddle_deploy if name == zone else self._paddle_retract)(name)
        self.log("route", target=zone)

    # -- спавн --------------------------------------------------------------
    def try_spawn(self):
        if self.api.spawn_queue.empty():
            return
        now = self.robot.getTime()
        if now - self.last_spawn_time < self.layout["spawn"]["min_interval_s"]:
            return
        obj_type, rotation_mode = self.api.spawn_queue.get()
        cfg = self.objects_cfg[obj_type]
        self.uid_seq += 1
        uid = self.uid_seq
        sp = self.layout["spawn"]
        if isinstance(rotation_mode, tuple):
            rotation = rotation_mode
        elif rotation_mode == "identity":
            rotation = (0.0, 0.0, 1.0, 0.0)
        else:
            rotation = random_orientation_for(cfg)
        z = (self.layout["belt_a"]["segments"][0]["height"]
             + spawn_clearance(cfg["bounding"], rotation) + sp["z_margin"])
        node_str = object_node(uid, obj_type, cfg, sp["x"], 0.0, z, rotation, density=self.density)
        root_children = self.robot.getRoot().getField("children")
        root_children.importMFNodeFromString(-1, node_str)
        self.nodes[uid] = self.robot.getFromDef(f"OBJ_{uid}")
        # Регистрация в top-трекере (CV-триггер, см. Sorter.__init__) —
        # объект ЕЩЁ НЕ наблюдался top-камерой, добавляется в очередь
        # ожидания первого сопоставления (без физической позиции).
        self.top_tracker.add_pending(uid)
        # Небольшой случайный угловой импульс: объект, упавший точно в
        # неустойчивом равновесии (на ребре/углу), иначе может застыть в
        # этом положении — сила тяжести даёт нулевой момент ровно в этой
        # точке, а без асимметричного возмущения решатель ничем его оттуда
        # не выведет (в реальности эту роль играют микронеровности/воздух).
        # При rotation="identity" импульс НЕ даём: цель этого режима —
        # детерминированная поза для сверки масок с истинными габаритами
        # (см. tools/report_cv_masks.py), объект и так стоит на плоской грани
        # (не на ребре/углу случайного разворота), устойчив без возмущения —
        # даже небольшой остаточный угловой момент к моменту cv_rig.x давал
        # систематическое занижение измеренного bbox против номинала.
        # Амплитуда ±0.6 рад/с — достаточна, чтобы стронуть коробку с ребра
        # (было ±0.15 — коробки ехали на ребре, не падали в устойчивое
        # положение), но не закручивает объект слишком сильно.
        if rotation_mode is None:
            self.nodes[uid].setVelocity([0.0, 0.0, 0.0,
                                          random.uniform(-0.6, 0.6),
                                          random.uniform(-0.6, 0.6),
                                          random.uniform(-0.6, 0.6)])
        self.last_spawn_time = now

        category = self.classifier.classify({"type": obj_type})
        zone = CATEGORY_TO_ZONE[category]
        self.router.add(TrackedObject(uid=uid, type=obj_type, category=category,
                                      zone=zone, spawn_time=now))
        self.log("spawn", uid=uid, type=obj_type, category=category, target=zone,
                  rot_deg=round(math.degrees(rotation[3]), 1),
                  rotation=[round(v, 6) for v in rotation])

    # -- утилизация ------------------------------------------------------
    RETIRE_GRACE_S = 0.5   # см. _dispose: сколько держать физически удалённое,
                            # но ещё не node.remove()'нутое тело перед реальным remove()

    def _dispose(self, uid: int):
        """Зона B — конвейер, а не конечная точка: доставленный товар убираем
        из мира совсем, как будто уехал дальше по цепочке. Так же поступаем
        с MISSED — товар физически неизвестно где, показывать нечего.

        РАНЬШЕ делали node.remove() немедленно, пока у узла ЕЩЁ активна
        физика (может быть в контакте с лентой/щитом в этот самый такт) —
        подозревается в частых крашах webots-bin.exe (ACCESS_VIOLATION,
        читение по протухшему указателю, один и тот же адрес во ВСЕХ
        дампах — см. заметку задачи "Тайминг шиберов B/D": похоже, ODE
        держит ссылку на geom/body, снесённые node.remove() посреди шага
        физики). Теперь — как _freeze(), но с последующим полным remove():
        1) скорость в ноль, 2) увезти за пределы поля (как _warmup_meshes),
        3) снять Physics (чистое отсоединение от ODE), 4) сам node.remove()
        — не сразу, а спустя RETIRE_GRACE_S (см. _process_retirements) —
        даём ODE как минимум несколько шагов на то, чтобы отпустить тело,
        прежде чем сносить сам узел."""
        node = self.nodes.get(uid)
        if node is None:
            return
        node.setVelocity([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        node.getField("translation").setSFVec3f([-50.0, (uid % 50) * 2.0, 2.0])
        phys = node.getField("physics")
        if phys is not None:
            phys.removeSF()
        self._retire_at[uid] = self.robot.getTime() + self.RETIRE_GRACE_S

    def _process_retirements(self, now: float) -> None:
        """Довершает _dispose(): физически убирает из мира узлы, у которых
        уже сняли физику и увезли за пределы поля не меньше RETIRE_GRACE_S
        назад (см. docstring _dispose — не делаем remove() в тот же такт,
        что и снятие физики)."""
        due = [uid for uid, t in self._retire_at.items() if now >= t]
        for uid in due:
            node = self.nodes.get(uid)
            if node:
                node.remove()
            self.nodes[uid] = None
            del self._retire_at[uid]

    def _freeze(self, uid: int, pos: tuple | None):
        """Зоны C/D — ролл-кейджи: наполнение должно оставаться видимым для
        реалистичности, поэтому узел не удаляем. Но раз он уже подтверждённо
        (confirm_time_s) лежит неподвижно в корзине, дальнейшая его
        физическая симуляция не нужна — убираем только Physics, boundingObject
        остаётся, так что следующий товар в ту же корзину по-прежнему
        сталкивается с ним/опирается на него как на статичное препятствие.
        Именно накопление НЕудалённых и НЕзамороженных физических тел в
        корзинах и было основной причиной замедления: ODE каждый такт заново
        строит контакты и интегрирует динамику всех лежащих товаров, даже
        неподвижных, и это растёт с числом спавненных товаров. Встроенное
        автоусыпление Webots (WorldInfo.physicsDisableTime, по умолчанию 1с)
        тут не спасает: ODE усыпляет/будит тела островами контактов целиком,
        а каждый новый упавший в ту же корзину товар создаёт новый контакт
        с островом и будит всех соседей заново — при спавне раз в
        min_interval_s остальные толком не успевают заснуть."""
        node = self.nodes.get(uid)
        if node is None or uid in self._frozen:
            return
        phys = node.getField("physics")
        if phys is not None:
            phys.removeSF()
        self._frozen.add(uid)
        if pos is not None:
            self._last_pos[uid] = pos

    def _maybe_clear_zone(self, zone: str) -> None:
        """Очистка зоны C/D при переполнении: если доставленных (и
        замороженных) товаров в зоне > zone_clear_threshold — самый старый
        удаляется из мира (как B). Без этого ролл-кейджи переполняются
        физическими телами — ODE тормозит и крашится при частом спавне.
        Порог — в config/layout.yaml: diverters.zone_clear_threshold.

        ВАЖНО: после удаления узла из мира надо убрать объект и из
        router.objects — иначе он остаётся в состоянии DELIVERED, при
        следующем _maybe_clear_zone снова попадает в in_zone, count растёт
        бесконечно, и один и тот же uid 'удаляется' снова и снова (был баг:
        zone_clear на uid=1 срабатывал 15 раз подряд)."""
        threshold = self.dv.get("zone_clear_threshold", 3)
        in_zone = [o for o in self.router.objects.values()
                   if o.state == DELIVERED and o.result_zone == zone]
        if len(in_zone) <= threshold:
            return
        # самый старый (минимальный spawn_time) — удалить
        oldest = min(in_zone, key=lambda o: o.spawn_time)
        node = self.nodes.get(oldest.uid)
        if node:
            node.remove()
        self.nodes[oldest.uid] = None
        if oldest.uid in self._frozen:
            self._frozen.discard(oldest.uid)
        if oldest.uid in self._last_pos:
            del self._last_pos[oldest.uid]
        # Убрать из router.objects — иначе остаётся DELIVERED, count растёт
        # бесконечно (баг: 15× zone_clear на uid=1).
        self.router.objects.pop(oldest.uid, None)
        self.log("zone_clear", uid=oldest.uid, zone=zone,
                 count=len(in_zone), threshold=threshold)

    # -- сброс --------------------------------------------------------------
    def do_reset(self):
        # Снять физику со ВСЕХ узлов ПЕРЕД node.remove() (см. _dispose) —
        # reset часто прилетает, пока объекты ещё активно едут/в контакте
        # с лентой/щитом (наш собственный тестовый скрипт так и делает
        # между прогонами). Отдельным проходом, а не заодно с remove() —
        # то же подозрение на ACCESS_VIOLATION при сносе Physics+Solid одним
        # действием на теле, ещё участвующем в контактах текущего шага.
        for node in self.nodes.values():
            if node:
                phys = node.getField("physics")
                if phys is not None:
                    phys.removeSF()
        for node in self.nodes.values():
            if node:
                node.remove()
        self.nodes.clear()
        self._retire_at.clear()
        self.router.objects.clear()
        self.router._busy.clear()
        self._stall.clear()
        self._frozen.clear()
        self._last_pos.clear()
        self.manual_mode = False
        self.set_route("C")
        for cam_name in self._moment_active:
            self.cv_cameras[cam_name]["device"].disable()
        self._moment_active.clear()
        self._moment_queue.clear()
        self.moment_scheduler = MomentScheduler(self.moment_thresholds, self.moment_captures)
        self.top_tracker = TopTracker(max_match_distance_m=TOP_TRACKER_MAX_MATCH_DISTANCE_M)
        if self._top_track_continuous_active:
            # объекты убраны мгновенно — дать камере тик на "успокоение"
            # рендера перед первой локализацией после reset (тот же resettle,
            # что при первом enable(), см. _advance_top_tracking).
            self._top_track_settle_countdown = MOMENT_CAPTURE_SETTLE_TICKS
        # Лента гарантированно пуста сразу после reset — переснять модель фона
        # (геометрия рига не меняется между запусками, см. заметку задачи).
        self.api.calibrate_bg_requested.set()
        self.log("reset")

    # -- нештатные ситуации --------------------------------------------------
    def check_stall(self, positions: dict):
        """Объект застрял — у шибера или где угодно ещё (например, завис на
        кромке ролл-кейджа после приземления в неудачном случайном развороте)
        — растолкать, иначе занятый им лоток (B/D, см. sim/router.py — теперь
        независимые по зонам) держится занятым до истечения release_timeout_s
        и блокирует следующий объект этой же зоны. У шибера шевелим сам щит
        (проверенно соскальзывает); иначе толкаем сам объект импульсом скорости.
        Лотков может быть занято несколько одновременно — проверяем каждый
        DISPATCHED-объект независимо.

        Зона C (прямой проезд без шибера) НЕ нуджится: у неё нет лотка, она
        никого не блокирует — застрявший C-объект просто дождётся
        release_timeout_s -> missed -> dispose. Раньше нудж C-объектов
        бесконечным setVelocity (каждые ~0.4с) перегружал ODE и валил Webots.

        Частота nudge ограничена: после каждого nudge (wiggle щита или толчок
        объекта) счётчик `since` сбрасывается на now, и следующий nudge не
        раньше `nudge_after_s`. Кроме того, после wiggle щита явно сбрасываем
        `since` (прежде этого не делалось — nudge срабатывал мгновенно снова
        каждые 0.4с, заливая лог и перегружая физику). Максимум nudge_count
        на объект — после него прекращаем (объект безнадёжно застрял, ждём
        missed/release_timeout_s).
        """
        now = self.robot.getTime()
        active = {o.uid: o for o in self.router.objects.values() if o.state == DISPATCHED}
        for uid in list(self._stall):
            if uid not in active:
                del self._stall[uid]

        max_nudges = self.dv.get("max_nudge_count", 6)
        for uid, obj in active.items():
            # Зона C — не нуджить (нет шибера, не блокирует очередь).
            if obj.zone == "C":
                continue
            pos = positions.get(uid)
            if pos is None:
                continue
            st = self._stall.setdefault(uid, {"pos": None, "since": now,
                                              "wiggle_until": 0.0, "count": 0})

            if st["wiggle_until"]:
                if now >= st["wiggle_until"]:
                    paddle = self.paddles.get(obj.zone)
                    if paddle:
                        paddle["motor"].setPosition(paddle["deploy"])
                    st["wiggle_until"] = 0.0
                    # ЯВНО сбрасываем таймер: нудж отработал, следующий не
                    # раньше nudge_after_s (прежде этого не было — nudge
                    # срабатывал мгновенно снова, см. docstring).
                    st["since"] = now
                continue

            if st["count"] >= max_nudges:
                # Объект безнадёжно застрял после всех nudge. Раньше шибер
                # держался развёрнутым до release_timeout_s (45с) — всё это
                # время лоток занят и блокирует следующий объект той же зоны
                # (пользователь: «шибер после срабатывания не возвращался»).
                # Освобождаем _busy и убираем щит СРАЗУ — следующий объект
                # сможет быть обслужен, а застрявший дождётся missed по
                # release_timeout_s (или выедет из-за развёрнутого щита, если
                # нудж в итоге стронет — _busy уже None, это безопасно).
                if self.router._busy.get(obj.zone) == uid:
                    self.router._busy[obj.zone] = None
                    self._paddle_retract(obj.zone)
                    self.log("stall_giveup", uid=uid, target=obj.zone,
                             reason="max_nudges_reached")
                continue

            moved = (st["pos"] is None
                     or max(abs(a - b) for a, b in zip(pos, st["pos"])) > 0.015)
            if moved:
                st.update(pos=pos, since=now)
                # объект поехал после прошлого nudge — счётчик не растём, но и
                # не сбрасываем (хроническое застревание того же объекта
                # должно упереться в max_nudges).
                continue
            if now - st["since"] <= self.dv["nudge_after_s"]:
                continue

            st["since"] = now
            st["count"] += 1
            paddle = self.paddles.get(obj.zone)
            px = self.dv["positions"].get(obj.zone, {}).get("x")
            near_paddle = (paddle is not None and px is not None
                           and abs(pos[0] - px) < 1.2 and abs(pos[1]) < 0.6 and pos[2] > 0.5)
            if near_paddle:
                back = paddle["deploy"] * (1 - self.dv["nudge_angle"] / abs(paddle["deploy"]))
                paddle["motor"].setPosition(back)
                st["wiggle_until"] = now + 0.4
            else:
                node = self.nodes.get(uid)
                if node:
                    # Толчок объекта. Если объект уже ЗА шибером (x > px+0.3)
                    # — щит его не касается, wiggle бесполезен. Толкаем ВБОК
                    # в сторону целевой зоны (-Y для D, +Y для B), чтобы
                    # стронуть с края ленты. Иначе (объект ещё у шибера или
                    # в произвольной точке) — вдоль ленты + вверх.
                    side_y = 0.0
                    if paddle is not None and px is not None and pos[0] > px + 0.3:
                        side = self.dv["positions"].get(obj.zone, {}).get("side", 1)
                        side_y = -float(side) * 0.4   # D side=+1 -> -Y, B side=-1 -> +Y
                        # 0.4 м/с вбок — стронуть с края ленты, не слетая с неё
                        # (0.8 было слишком сильно — шлем улетал на y=-0.9).
                    import random as _r
                    node.setVelocity([0.4, side_y, 0.3,
                                      _r.uniform(-0.3, 0.3),
                                      _r.uniform(-0.3, 0.3),
                                      _r.uniform(-0.3, 0.3)])
            self.log("nudge", uid=uid, target=obj.zone, count=st["count"])

    # -- основной цикл ------------------------------------------------------
    def run(self):
        while self.robot.step(self.timestep) != -1:
            if self.api.reset_requested.is_set():
                self.api.reset_requested.clear()
                self.do_reset()
            while not self.api.manual_queue.empty():
                cmd = self.api.manual_queue.get()
                if cmd == "auto":
                    self.manual_mode = False
                    self.log("mode", manual=False)
                else:
                    self.manual_mode = True
                    self.set_route(cmd)
                    self.log("mode", manual=True, zone=cmd)
            if self.api.screenshot_requested.is_set():
                if self._cam_countdown < 0:
                    self.camera.enable(self.timestep)
                    self._cam_countdown = 3   # дать камере отрисовать кадр
                elif self._cam_countdown > 0:
                    self._cam_countdown -= 1
                else:
                    self.camera.saveImage(str(self.api.screenshot_path), 90)
                    self.camera.disable()
                    self._cam_countdown = -1
                    self.api.screenshot_requested.clear()
                    self.api.screenshot_done.set()
            # Калибровка фона (Фаза 2.1, прямые камеры; с Фазы 3 — и
            # зеркальные top_mirror/diag_mirror, нужны для фон-сетки на все 9
            # ячеек, см. sim/cv_grid.py) — по одному кадру с каждой камеры на
            # пустой ленте, та же схема enable/countdown/saveImage, что у
            # screenshot/cv_frame (рендер по запросу, не постоянно).
            if self.api.calibrate_bg_requested.is_set() and not self._bg_calib_active:
                self.api.calibrate_bg_requested.clear()
                self._bg_calib_active = True
                self._bg_calib_countdowns = {name: -1 for name in self.cv_cameras}
            if self._bg_calib_active:
                pending = False
                for cam_name in list(self._bg_calib_countdowns):
                    cd = self._bg_calib_countdowns[cam_name]
                    cam_info = self.cv_cameras[cam_name]
                    if cd < 0:
                        cam_info["device"].enable(self.timestep)
                        self._bg_calib_countdowns[cam_name] = 3
                        pending = True
                    elif cd > 0:
                        self._bg_calib_countdowns[cam_name] = cd - 1
                        pending = True
                    else:
                        path = self.api.cv_frame_dir / f"{cam_name}_background.jpg"
                        cam_info["device"].saveImage(str(path), 90)
                        # top_track под непрерывным трекингом (CV-триггер, см.
                        # _advance_top_tracking) не отключаем — иначе каждая
                        # калибровка фона рвёт "постоянно включена", требуя
                        # повторного settle.
                        if not (cam_name == "top_track" and self._top_track_continuous_active):
                            cam_info["device"].disable()
                        if cam_info["mirror"]:
                            _flip_mirror_frame(path)   # см. docstring — ориентация
                                                        # фона должна совпадать с
                                                        # захваченными зеркальными кадрами
                        frame = cv2.imread(str(path))
                        self.api.cv_backgrounds[cam_name] = build_background_model(frame)
                        if cam_name == "top_track":
                            self._top_track_bg_frame = frame   # RAW BGR — см. Sorter.__init__
                        del self._bg_calib_countdowns[cam_name]
                if not pending:
                    self._bg_calib_active = False
                    self.api.calibrate_bg_done.set()
            # Все камеры, запрошенные конкурентно (в одном или соседних тиках),
            # включаются/снимаются/выключаются ВМЕСТЕ — иначе последовательная
            # обработка растягивается на секунды реального времени, и объект
            # (1 м/с) успевает уехать из кадра между первой и последней камерой.
            for cam_name, entry in list(self.api.cv_frame_requests.items()):
                if not entry.get("pending"):
                    continue
                cam_info = self.cv_cameras[cam_name]
                cd = self._cv_frame_countdowns.get(cam_name, -1)
                if cd < 0:
                    cam_info["device"].enable(self.timestep)
                    self._cv_frame_countdowns[cam_name] = 3   # дать камере отрисовать кадр
                elif cd > 0:
                    self._cv_frame_countdowns[cam_name] = cd - 1
                else:
                    path = self.api.cv_frame_dir / f"{cam_name}.jpg"
                    cam_info["device"].saveImage(str(path), 90)
                    cam_info["device"].disable()
                    if cam_info["mirror"]:
                        _flip_mirror_frame(path)
                    self._cv_frame_countdowns[cam_name] = -1
                    ok = True
                    if entry.get("mode") == "mask":
                        bg = self.api.cv_backgrounds.get(cam_name)
                        if bg is None:
                            ok = False
                        else:
                            frame = cv2.imread(str(path))
                            bs = self.bg_subtraction_cfg
                            mask, _bbox = compute_mask(
                                frame, bg, bs["h_threshold"], bs["s_threshold"],
                                bs["min_area_px"], bs["morph_kernel"],
                                v_threshold=bs.get("v_threshold"))
                            mask_path = self.api.cv_frame_dir / f"{cam_name}_mask.png"
                            cv2.imwrite(str(mask_path), mask)
                    entry["ok"] = ok
                    entry["pending"] = False
                    entry["done"].set()
            if self.api.record_start_requested.is_set():
                self.api.record_start_requested.clear()
                if self.record_state in ("idle", "ready", "failed"):
                    RECORD_DIR.mkdir(parents=True, exist_ok=True)
                    path = RECORD_DIR / f"record_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
                    self.robot.movieStartRecording(str(path), 1280, 720, 0, 100, 1, False)
                    self.record_state = "recording"
                    self.record_file = path.name
                    self.log("record_start", file=path.name)
            if self.api.record_stop_requested.is_set():
                self.api.record_stop_requested.clear()
                if self.record_state == "recording":
                    self.robot.movieStopRecording()
                    self.record_state = "encoding"
                    self.log("record_stop", file=self.record_file)
            if self.record_state == "encoding":
                if self.robot.movieFailed():
                    self.record_state = "failed"
                    self.log("record_failed", file=self.record_file)
                elif self.robot.movieIsReady():
                    self.record_state = "ready"
                    self.log("record_ready", file=self.record_file)
            self.try_spawn()

            positions = {}
            for uid, node in self.nodes.items():
                if uid in self._frozen:
                    positions[uid] = self._last_pos.get(uid)
                    continue
                if node:
                    positions[uid] = tuple(node.getPosition())

            self._advance_top_tracking()
            positions_x_cv = {uid: self.top_tracker.last_x(uid) for uid in self.nodes}
            physics_positions_x = {uid: (pos[0] if pos else None) for uid, pos in positions.items()}
            self._advance_moment_capture(positions_x_cv, physics_positions_x)

            before = {u: o.state for u, o in self.router.objects.items()}
            busy_before = {z: self.router._busy.get(z) for z in self.paddles}
            commands = self.router.step(positions, self.robot.getTime())
            if not self.manual_mode:
                for zone in commands:
                    self._paddle_deploy(zone)
                    self.log("route", target=zone)
                for zone, prev_uid in busy_before.items():
                    if prev_uid is not None and self.router._busy.get(zone) is None:
                        self._paddle_retract(zone)   # обслужен — лоток в парковку
            self.check_stall(positions)

            # Зоны, куда в этом такте была доставка C/D — для очистки при
            # переполнении ПОСЛЕ цикла (нельзя менять router.objects во время
            # итерации по нему — RuntimeError dict changed size, краш supervisor).
            for uid, obj in self.router.objects.items():
                if before.get(uid) != obj.state:
                    if obj.state == DELIVERED:
                        ok = obj.result_zone == obj.zone
                        self.log("delivered", uid=uid, type=obj.type,
                                 zone=obj.result_zone, correct=ok)
                        # Все зоны (B/C/D) — товар исчезает после доставки,
                        # как будто уехал дальше по цепочке (B — конвейер,
                        # C/D — ролл-кейджи уехали на упаковку). См. _dispose:
                        # физика снимается сразу, сам node.remove() — с
                        # отсрочкой (RETIRE_GRACE_S).
                        self._dispose(uid)
                    elif obj.state == MISSED:
                        self.log("missed", uid=uid, type=obj.type, target=obj.zone)
                        self._dispose(uid)
                    self.moment_scheduler.forget(uid)   # уже проехал зону CV, пороги не нужны
                    self.top_tracker.forget(uid)

            self._process_retirements(self.robot.getTime())
            self.publish(positions)

    # -- CV-триггер по top-локализации (замена физики) -----------------------
    def _advance_top_tracking(self) -> None:
        """Поддерживает ВЫДЕЛЕННУЮ низкоразрешающую камеру "cv_top_track"
        ПОСТОЯННО включённой, пока `moments_enabled` (в отличие от остальных
        cv_* камер, которые включаются по конкретному запросу/порогу) —
        иначе нечем обнаружить сам момент пересечения порога, см. заметку
        задачи "CV-триггер момента по top-локализации (замена физики, Фаза 4,
        CV-пайплайн Webots)".

        НЕ переиспользует полноразмерную "top" (та осталась как раньше —
        включается кратко только на сам момент снимка сетки): постоянный
        enable() полноразмерной 2592×1944 камеры даёт ~11x просадку
        realtime-factor (0.92->0.083, живой замер) — `Camera.getImage()` в
        Python не кэш, а живой запрос через границу процессов (controller —
        отдельный OS-процесс) НА КАЖДЫЙ вызов, издержки ~пропорциональны
        размеру кадра, не мощности GPU (подтверждено исходником
        `lib/controller/python/controller/camera.py`: `image` — property,
        каждый обращение = новый `wb_camera_get_image()` + копия буфера).
        top_track — та же позиция/ориентация, что top, но native-разрешение
        1296×320 (без доп. ds8-занижения — `factor=1`: камера уже достаточно
        мала, повторный downscale потерял бы тонкие объекты вроде ручки).

        На каждом установившемся тике: getImage() → `localize_top_components`
        → `self.top_tracker.update`. Результат читается вызывающей стороной
        через `self.top_tracker.last_x`, не возвращается отсюда напрямую."""
        device = self.cv_cameras["top_track"]["device"]

        if not self.api.moments_enabled:
            if self._top_track_continuous_active:
                device.disable()
                self._top_track_continuous_active = False
                self._top_track_settle_countdown = -1
            return

        if not self._top_track_continuous_active:
            device.enable(self.timestep)
            self._top_track_continuous_active = True
            self._top_track_settle_countdown = MOMENT_CAPTURE_SETTLE_TICKS
            return

        if self._top_track_settle_countdown > 0:
            self._top_track_settle_countdown -= 1
            return

        if self._top_track_bg_frame is None:
            return   # фон top_track ещё не откалиброван — локализация невозможна

        raw = device.getImage()
        if raw is None:
            return
        width, height = device.getWidth(), device.getHeight()
        frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 4))[:, :, :3]
        components = localize_top_components(
            frame, self._top_track_bg_frame, self._top_track_bs_cfg, self.top_geom,
            cv_rig_x=self.api.cv_rig_cfg["x"], factor=1)
        self.top_tracker.update(components)

    # -- детектор моментов старт/центр/конец (Фаза 2.2) ---------------------
    def _advance_moment_capture(self, positions_x: dict, physics_positions_x: dict):
        """Планирует и продвигает захват кадров на порогах `moment_thresholds`
        (sim/cv_moments.py). Не более ОДНОГО активного захвата на камеру
        одновременно — при нескольких объектах в зоне CV подряд следующий
        запрос той же камеры просто ждёт своей очереди (FIFO per-camera);
        лента однонаправленная (FIFO по x), так что порядок в очереди всегда
        совпадает с порядком проезда объектов.

        `positions_x`: uid -> мировая X (или None, если ещё нет ни одного
        CV-наблюдения) — из `_advance_top_tracking`/`top_tracker` (CV-триггер,
        см. Sorter.__init__), ИМЕННО ЭТОТ источник управляет порогами
        (`moment_scheduler.step` ниже), НЕ физика (в прод-версии физика
        недоступна, см. заметку задачи "CV-триггер момента по
        top-локализации..."). `physics_positions_x`: uid -> физическая X В
        ТОТ ЖЕ МОМЕНТ (`node.getPosition()`, доступна только в симуляторе) —
        используется ТОЛЬКО для логирования (`cv_moment.x_physics`), чтобы
        офлайн можно было сверить, насколько CV-триггер (по построению —
        приближённый) отклоняется от точного физического порога (Приёмка
        задачи, п.1: допуск ±20мм); саму логику срабатывания не трогает."""
        if not self.api.moments_enabled:
            return   # выключено по умолчанию — см. Api.moments_enabled

        events = self.moment_scheduler.step(positions_x)
        self._moment_queue.extend(events)

        # Не более ОДНОГО завершения захвата (getImage) за тик — на моменте
        # "центр" 5 камер (top/side/diag/top_mirror/diag_mirror, Фаза 3) готовы
        # одновременно, без этого ограничения все пять обрабатывались бы
        # подряд внутри одного вызова.
        finalized_this_tick = False
        for cam_name in self._moment_cameras:
            active = self._moment_active.get(cam_name)
            if active is None:
                for i, (uid, camera, moment) in enumerate(self._moment_queue):
                    if camera == cam_name:
                        del self._moment_queue[i]
                        self.cv_cameras[cam_name]["device"].enable(self.timestep)
                        self._moment_active[cam_name] = {"uid": uid, "moment": moment,
                                                          "countdown": MOMENT_CAPTURE_SETTLE_TICKS,
                                                          "enabled_at": time.perf_counter()}
                        break
                continue
            if active["countdown"] > 0:
                active["countdown"] -= 1
                if active["countdown"] == 0:
                    # "Предыдущий" кадр той же камеры (Фаза 3) — ОДНИМ тиком
                    # раньше финального, снят с уже enable()'нутого устройства
                    # (без отдельного enable/disable — settle уже показал, что
                    # доп. getImage() на 2592x1944 дёшев, ~5мс, см. docstring
                    # _jpeg_writer_loop) — заготовка под будущее разделение
                    # объектов при многообъектном кадре (сам алгоритм — Фаза 4,
                    # см. заметку задачи "Сборка сетки 9 ракурсов...").
                    prev_device = self.cv_cameras[cam_name]["device"]
                    prev_raw = prev_device.getImage()
                    prev_path = (self.api.cv_frame_dir
                                 / f"moment_{active['uid']}_{cam_name}_{active['moment']}_prev.jpg")
                    self._moment_jpeg_jobs.put((prev_raw, prev_device.getWidth(), prev_device.getHeight(),
                                                 prev_path, self.cv_cameras[cam_name]["mirror"]))
                continue
            if finalized_this_tick:
                continue   # готова, но бюджет тика исчерпан — заберём на следующем
            finalized_this_tick = True

            uid, moment = active["uid"], active["moment"]
            cam_info = self.cv_cameras[cam_name]
            device = cam_info["device"]
            path = self.api.cv_frame_dir / f"moment_{uid}_{cam_name}_{moment}.jpg"
            # Маска НЕ считается здесь — раньше compute_mask (0.15-0.34с на
            # 2592x1944) шёл синхронно в этом же цикле симуляции на КАЖДЫЙ
            # из 7 кадров/объект (реальные фризы, см. заметку задачи) —
            # теперь считается ЛЕНИВО, по запросу, в потоке HTTP-обработчика
            # из уже сохранённого JPEG (см. GET /api/cv/moment_mask/... ниже).
            # JPEG-кодирование+запись — тоже НЕ здесь: `getImage()` (сырой
            # буфер, ~5мс) обязан вызываться из главного потока (Webots API),
            # но сам кодек+диск (~0.1-0.17с) вынесены в фоновый поток
            # (`_jpeg_writer_loop`) — из главного цикла остаётся только
            # дешёвый getImage()+disable().
            t_wait = time.perf_counter()
            raw = device.getImage()
            width, height = device.getWidth(), device.getHeight()
            t_save = time.perf_counter()
            device.disable()
            self._moment_jpeg_jobs.put((raw, width, height, path, cam_info["mirror"]))
            # x логируется НА МОМЕНТ ФАКТИЧЕСКОГО getImage() (не порог-цель) —
            # честная точка сравнения для офлайн-отчёта (tools/capture_moments.py):
            # содержимое кадра соответствует именно этой позиции, а не идеальному
            # порогу (settle-задержка camera.enable(), редко — очередь камеры при
            # нескольких объектах разом, см. docstring метода).
            x_val = positions_x.get(uid)
            x_physics = physics_positions_x.get(uid)
            self.log("cv_moment", uid=uid, camera=cam_name, moment=moment,
                      x=round(x_val, 4) if x_val is not None else None,
                      x_physics=round(x_physics, 4) if x_physics is not None else None,
                      wait_s=round(t_wait - active["enabled_at"], 3),
                      save_s=round(t_save - t_wait, 3))
            del self._moment_active[cam_name]

    def publish(self, positions: dict):
        objs = []
        for uid, obj in self.router.objects.items():
            pos = positions.get(uid)
            x_cv = self.top_tracker.last_x(uid)
            objs.append({
                "uid": uid, "type": obj.type, "label": self.objects_cfg[obj.type]["label"],
                "category": obj.category, "target": obj.zone, "state": obj.state,
                "result": obj.result_zone,
                "x": round(pos[0], 2) if pos else None,
                "y": round(pos[1], 2) if pos else None,
                "z": round(pos[2], 2) if pos else None,
                # Диагностика CV-триггера (top-локализация) — см. заметку
                # задачи "CV-триггер момента по top-локализации...".
                "x_cv": round(x_cv, 4) if x_cv is not None else None,
            })
        contacts = []
        for busy in self.router._busy.values():
            node = self.nodes.get(busy) if busy is not None else None
            if node:
                for cp in node.getContactPoints():
                    p = cp.getPoint()
                    contacts.append([round(p[0], 3), round(p[1], 3), round(p[2], 3)])
        with self.api.lock:
            self.api.status = {
                "time": round(self.robot.getTime(), 1),
                "route": self.current_target,
                "manual": self.manual_mode,
                # 0 при оконном запуске (run.sh без --headless) — у Webots
                # уже есть нативное окно, панели рендерить 3D-вид повторно
                # незачем по умолчанию (см. web/index.html).
                "headless": os.environ.get("ROBOZON_HEADLESS") != "0",
                "record_state": self.record_state,
                "record_file": self.record_file,
                "ffmpeg_available": FFMPEG_AVAILABLE,
                "cv_bg_calibrated": all(name in self.api.cv_backgrounds
                                         for name in self.cv_cameras),
                "cv_moments_enabled": self.api.moments_enabled,
                "contacts": contacts[:12],
                "paddles": {n: round(p["sensor"].getValue(), 2)
                            for n, p in self.paddles.items()},
                "objects": objs[-30:],
                "events": self.events[-40:],
                "stats": self.router.stats(),
            }


if __name__ == "__main__":
    Sorter().run()
