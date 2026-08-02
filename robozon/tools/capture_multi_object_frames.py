#!/usr/bin/env python3
"""Захват многообъектных тестовых кадров для сетки ракурсов (Фаза 4).

Дополнение к tools/capture_cv_frames.py (снимает ОДИН объект в момент
прохождения cv_rig.x) — спавнит 2-3 объекта с интервалом 0.5с (как
config/layout.yaml: spawn.min_interval_s, скорость ленты 1.0 м/с → 0.5м
между объектами), дожидается прохождения ЦЕЛЕВЫМ объектом всех 5 порогов
moment_thresholds (sim/cv_moments.py), и снимает 9 кадров (как
Sorter._advance_moment_capture, но по целевому uid через HTTP API, не
автозахват) + prev-кадр на ~0.33с раньше порога + ground-truth objects.json
(все видимые объекты с uid/x/y/z из /api/status).

Имена файлов — те же, что у автозахвата:
`moment_<uid>_<camera>_<moment>[_prev].jpg` (uid целевого). Дополнительно
`moment_<uid>_<camera>_<moment>[_prev]_objects.json` — ground-truth.

Фоны переиспользуем из assets/cv_grid_test_frames/backgrounds/ — не
калибруем.

Требует запущенной симуляции (`scripts/run.ps1`). Грабли (см. 01 Sources/03
Conventions.md): прокси Windows ломает HTTP — в скрипте прокси-переменные
снимаются явно (urllib opener без proxy). 503 на первый запрос CV-камер
после старта (холодный рендер) — ретрай.

Пример (все 4 набора):
    python3 tools/capture_multi_object_frames.py --out assets/cv_grid_test_frames/multi

Один набор:
    python3 tools/capture_multi_object_frames.py \\
        --sets pen_boxlarge --out /tmp/multi
"""
import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.config import load_layout                                                     # noqa: E402
from sim.cv_moments import moment_thresholds, threshold_captures                        # noqa: E402

# Прокси ломает локальные запросы (см. 01 Sources/03 Conventions.md) — opener
# без proxy. Глобально для всех urllib-вызовов в скрипте.
_PROXY_HANDLER = urllib.request.ProxyHandler({})
_OPENER = urllib.request.build_opener(_PROXY_HANDLER)

# Сеты по выбору пользователя (2026-07-24): 2-3 пары + 1 тройка. Целевой —
# ПЕРВЫЙ в списке (первый едет по ленте, его моменты фиксируем). Порядок
# спавна: целевой первым, соседи позади (по X меньше — отстанут на 0.5м).
SETS = {
    "pen_boxlarge":               ["pen", "box_large"],
    "boxsmall_boxlarge":          ["box_small", "box_large"],
    "cylinder_boxsmall":          ["cylinder", "box_small"],
    "triple_pen_boxlarge_boxsmall": ["pen", "box_large", "box_small"],
}

# prev-кадр: ~0.33с до порога (объект на ~0.33м до порога) — аналог реальных
# 30 fps камер (решение пользователя 2026-07-24).
_PREV_DELTA_M = 0.33

# Допуск по X на срабатывание события (м). Поллинг /api/status ~200 Гц
# (sleep 5мс) → шаг ~5мм; 0.03м даёт запас на jitter поллинга. Раньше 0.10м
# при 30мс-поллинге — опоздание на ~0.10-0.30м, критично для side-камер с
# узким FOV (delta_max=0.316м): end_side (порог 2.868) снимался на x=3.33
# (объект уже ВНЕ кадра side) — найдено на multi_1m boxsmall_boxlarge.
_EVENT_TOLERANCE_M = 0.03

# Ретрай при 503 (холодный рендер CV-камер после старта Webots — см.
# Конвенции).
_HTTP_RETRIES = 3
_HTTP_RETRY_DELAY = 0.5


def _get(url: str, timeout: float = 10.0) -> bytes:
    req = urllib.request.Request(url)
    last_exc = None
    for _ in range(_HTTP_RETRIES):
        try:
            with _OPENER.open(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            last_exc = e
            if e.code in (502, 503):
                time.sleep(_HTTP_RETRY_DELAY)
                continue
            raise
        except urllib.error.URLError as e:
            last_exc = e
            time.sleep(_HTTP_RETRY_DELAY)
            continue
    raise RuntimeError(f"HTTP {url} failed after {_HTTP_RETRIES} retries: {last_exc}")


def _post(url: str, body: dict | None = None, timeout: float = 10.0) -> dict:
    data = json.dumps(body or {}).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    with _OPENER.open(req, timeout=timeout) as r:
        return json.loads(r.read())


def api_status(port: int) -> dict:
    return json.loads(_get(f"http://localhost:{port}/api/status"))


def api_reset(port: int) -> None:
    _post(f"http://localhost:{port}/api/reset")


def api_spawn(port: int, obj_type: str, rotation: str = "identity") -> None:
    _post(f"http://localhost:{port}/api/spawn", {"type": obj_type, "rotation": rotation})


def save_camera(port: int, name: str, out_path: Path) -> None:
    data = _get(f"http://localhost:{port}/api/cv/frame/{name}", timeout=15.0)
    out_path.write_bytes(data)


def spawn_and_get_uid(port: int, obj_type: str, before_uids: set[int],
                      timeout: float = 15.0) -> int:
    api_spawn(port, obj_type)
    deadline = time.time() + timeout
    while time.time() < deadline:
        for o in api_status(port)["objects"]:
            if o["uid"] not in before_uids:
                return o["uid"]
        time.sleep(0.05)
    raise RuntimeError(f"объект {obj_type} не заспавнился за {timeout}с")


def _find_uid(status: dict, uid: int) -> dict | None:
    for o in status["objects"]:
        if o["uid"] == uid:
            return o
    return None


def capture_event(port: int, target_uid: int, cameras: list[str], out_dir: Path,
                   stem: str) -> None:
    """[unused] Снимает `cameras` (конкурентно через /api/cv/frame/<name>),
    сохраняет `moment_<stem>_<camera>.jpg` + `moment_<stem>_<camera>_objects.json`.
    Оставлено как референс; основной путь — инлайн в run_set (имя файла с
    моментом, не stem)."""
    import concurrent.futures
    status = api_status(port)
    gt = [{"uid": o["uid"], "type": o["type"], "x": o["x"], "y": o["y"], "z": o["z"]}
          for o in status["objects"]]
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(cameras))) as ex:
        futs = {ex.submit(save_camera, port, name,
                          out_dir / f"moment_{stem}_{name}.jpg"): name
                for name in cameras}
        for fut in concurrent.futures.as_completed(futs):
            fut.result()
    (out_dir / f"moment_{stem}_objects.json").write_text(
        json.dumps({"target_uid": target_uid, "objects": gt}, ensure_ascii=False, indent=2),
        encoding="utf-8")


def run_set(port: int, set_name: str, obj_types: list[str], out_root: Path,
            spawn_interval_s: float = 0.5, include_prev: bool = True) -> int:
    """Один подпроезд: reset, спавн N объектов с интервалом, обход событий по
    X целевого, снятие 9 кадров + 9 prev + 18 objects.json. Возвращает uid
    целевого объекта."""
    out_dir = out_root / set_name
    out_dir.mkdir(parents=True, exist_ok=True)

    rig = load_layout()["cv_rig"]
    th = moment_thresholds(rig)
    captures = threshold_captures()

    # События (отсортированы по X целевого): prev (~0.33м до порога) и main.
    # Каждое событие = (X порога, th_name, момент "start"/"center"/"end",
    # список камер, is_prev). th_name (start_top/start_side/center/end_side/
    # end_top) — уникальный идентификатор порога/события, нужен для GT-файла
    # per-события (см. ниже — moment "end" снимается двумя событиями при
    # РАЗНЫХ X). `captures[th_name]` = [(camera, moment), ...] — moment общий
    # для всех камер события. Имя кадра:
    # moment_<uid>_<camera>_<moment>[_prev].jpg (как автозахват).
    events: list[tuple[float, str, str, list[str], bool]] = []
    for th_name in ("start_top", "start_side", "center", "end_side", "end_top"):
        moment_label = captures[th_name][0][1]   # "start"/"center"/"end"
        cams = [c for c, _m in captures[th_name]]
        if include_prev:
            events.append((th[th_name] - _PREV_DELTA_M, th_name, moment_label, cams, True))
        events.append((th[th_name], th_name, moment_label, cams, False))
    events.sort(key=lambda e: e[0])

    print(f"\n=== {set_name}: {obj_types} ===")
    api_reset(port)
    time.sleep(1.0)

    # Спавн N объектов с интервалом. Целевой — первый. rotation=identity для
    # детерминированной позы (как tools/capture_moments.py).
    target_uid = None
    before = {o["uid"] for o in api_status(port)["objects"]}
    for i, obj_type in enumerate(obj_types):
        uid = spawn_and_get_uid(port, obj_type, before)
        before.add(uid)
        if i == 0:
            target_uid = uid
        print(f"  спавн {obj_type} -> uid={uid}{' (целевой)' if i == 0 else ''}")
        if i < len(obj_types) - 1:
            time.sleep(spawn_interval_s)

    if target_uid is None:
        raise RuntimeError("целевой объект не заспавнен")

    # Обход событий по X целевого. Ждём x >= event_x (с допуском), снимаем.
    # Каждое событие идентифицируется th_name (start_top/start_side/center/
    # end_side/end_top) — УНИКАЛЬНО per-события, в отличие от moment_label
    # ("end" общий для end_side и end_top, которые срабатывают при РАЗНЫХ X
    # порогах 2.868 vs 3.426 — целевой в разном положении). GT-файл per-события
    # сохраняет корректный X целевого на момент снятия КАЖДОЙ камеры, не
    # перезаписывается следующим событием того же moment (найдено на multi_1m:
    # end_side кадр снят на x=2.92, но GT "end" перезаписан end_top на x=3.49).
    for event_x, th_name, moment_label, cameras, is_prev in events:
        # Ждём, пока целевой не достигнет event_x (объект движется в +X).
        # Допуск: снимаем как только x >= event_x - tolerance. Если уже проехал
        # (после предыдущего долгого снятия) — снимаем сразу.
        deadline = time.time() + 60.0
        while time.time() < deadline:
            o = _find_uid(api_status(port), target_uid)
            if o is None or o["x"] is None:
                time.sleep(0.005)
                continue
            if o["x"] >= event_x - _EVENT_TOLERANCE_M:
                break
            time.sleep(0.005)
        else:
            raise RuntimeError(f"целевой uid={target_uid} не дошёл до x={event_x:.3f}")

        suffix = "_prev" if is_prev else ""
        # Камеры события — конкурентно (одним тиком, иначе объект уедет между
        # камерами center: 5 камер рендерятся ~0.1-0.3с суммарно).
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(cameras))) as ex:
            futs = {ex.submit(save_camera, port, cam,
                              out_dir / f"moment_{target_uid}_{cam}_{moment_label}{suffix}.jpg"): cam
                    for cam in cameras}
            for fut in concurrent.futures.as_completed(futs):
                fut.result()
        # objects.json — per-события (th_name): snapshot /api/status одного
        # тика, корректный X целевого на момент снятия ЭТИХ камер. Имя
        # moment_<uid>_<th_name>[_prev]_objects.json — уникально per-события.
        # Дополнительно сохраняем старое имя moment_<uid>_<moment>[_prev]_objects.json
        # для совместимости (каждое событие момента перезаписывает его —
        # последний выигрывает; использовать per-th_name для точности).
        status = api_status(port)
        gt = [{"uid": o["uid"], "type": o["type"], "x": o["x"], "y": o["y"], "z": o["z"]}
              for o in status["objects"]]
        gt_payload = {"target_uid": target_uid,
                       "threshold": th_name, "moment": moment_label,
                       "threshold_x": event_x, "objects": gt}
        (out_dir / f"moment_{target_uid}_{th_name}{suffix}_objects.json"
         ).write_text(json.dumps(gt_payload, ensure_ascii=False, indent=2),
                      encoding="utf-8")
        # Совместимость: старое имя (без th_name) — перезаписывается, последний
        # event момента выигрывает. Для center (одно событие) — корректно.
        (out_dir / f"moment_{target_uid}_{moment_label}{suffix}_objects.json"
         ).write_text(json.dumps(gt_payload, ensure_ascii=False, indent=2),
                      encoding="utf-8")
        o = _find_uid(status, target_uid)
        x_now = o["x"] if o else None
        tag = "prev" if is_prev else "main"
        print(f"  [{tag}] {th_name:10s} {moment_label:10s} x={x_now:.3f} "
              f"(цель={event_x:.3f}) камеры={cameras}")

    print(f"  готово: {out_dir} (uid={target_uid})")
    return target_uid


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8008)
    ap.add_argument("--out", required=True, help="корневая папка для наборов (создаются подпапки по имени сета)")
    ap.add_argument("--sets", default="all",
                    help="наборы через запятую (по умолчанию all = все 4): "
                         + ",".join(SETS))
    ap.add_argument("--spawn-interval", type=float, default=0.5,
                    help="интервал между спавнами, с (0.5 = как spawn.min_interval_s)")
    ap.add_argument("--no-prev", action="store_true",
                    help="не снимать prev-кадры (ускоряет — без двойного обхода "
                         "событий; end_side на узком FOV side точнее к порогу)")
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.sets == "all":
        sets = list(SETS.items())
    else:
        names = [s.strip() for s in args.sets.split(",")]
        missing = [n for n in names if n not in SETS]
        if missing:
            print(f"неизвестные наборы: {missing}; доступные: {list(SETS)}", file=sys.stderr)
            return 1
        sets = [(n, SETS[n]) for n in names]

    # Проверка связи с симулятором.
    try:
        api_status(args.port)
    except Exception as e:
        print(f"симулятор не отвечает на localhost:{args.port}: {e}", file=sys.stderr)
        print("запустите: scripts/run.ps1", file=sys.stderr)
        return 1

    summary = []
    for set_name, obj_types in sets:
        try:
            uid = run_set(args.port, set_name, obj_types, out_root,
                          spawn_interval_s=args.spawn_interval,
                          include_prev=not args.no_prev)
            summary.append((set_name, uid, "ok"))
        except Exception as e:
            summary.append((set_name, None, f"FAIL: {e}"))
            print(f"  ОШИБКА: {e}", file=sys.stderr)

    print("\n=== Итог ===")
    for set_name, uid, status in summary:
        print(f"  {set_name:40s} uid={uid} {status}")
    return 0 if all(s == "ok" for _, _, s in summary) else 1


if __name__ == "__main__":
    sys.exit(main())