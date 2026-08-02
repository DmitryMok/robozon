#!/usr/bin/env python3
"""Снять кадры с CV-камер и/или обзорный кадр площадки через HTTP API работающей
симуляции (сначала запустить `scripts/run.sh --headless`).

Примеры:
    # обзорный кадр площадки (проверить геометрию, шиберы, кейджи)
    python3 tools/capture_cv_frames.py --overview-only --out /tmp/overview

    # объект + все 5 CV-камер (top/diag/side/top_mirror/diag_mirror) в момент
    # прохождения им точки наблюдения cv_rig.x
    python3 tools/capture_cv_frames.py --type box_small --out /tmp/cv_frames

    # только часть камер
    python3 tools/capture_cv_frames.py --type bottle --cameras top,side --out /tmp/x

    # + маска фоновой субтракции (Фаза 2.1, только прямые камеры top/diag/side)
    # и overlay.jpg (полупрозрачная маска поверх кадра) для визуальной проверки
    python3 tools/capture_cv_frames.py --type box_small --with-mask --out /tmp/cv_frames
"""
import argparse
import concurrent.futures
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.config import load_layout  # noqa: E402


def api_status(port: int) -> dict:
    with urllib.request.urlopen(f"http://localhost:{port}/api/status", timeout=10) as r:
        return json.loads(r.read())


def api_post(port: int, path: str, body: dict | None = None) -> dict:
    data = json.dumps(body or {}).encode()
    req = urllib.request.Request(f"http://localhost:{port}/api/{path}", data=data, method="POST")
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def save_camera(port: int, name: str, out_dir: Path) -> None:
    with urllib.request.urlopen(f"http://localhost:{port}/api/cv/frame/{name}", timeout=10) as r:
        (out_dir / f"{name}.jpg").write_bytes(r.read())


def save_mask_and_overlay(port: int, name: str, out_dir: Path) -> None:
    """Тянет маску `/api/cv/mask/<name>` (только прямые камеры, см. supervisor_main.py)
    и строит overlay.jpg (полупрозрачная маска поверх уже сохранённого кадра
    `<name>.jpg` — вызывать после save_camera)."""
    from PIL import Image

    with urllib.request.urlopen(f"http://localhost:{port}/api/cv/mask/{name}", timeout=10) as r:
        mask_bytes = r.read()
    mask_path = out_dir / f"{name}_mask.jpg"
    mask_path.write_bytes(mask_bytes)

    frame = Image.open(out_dir / f"{name}.jpg").convert("RGB")
    mask = Image.open(mask_path).convert("L").resize(frame.size)
    red = Image.new("RGB", frame.size, (255, 0, 0))
    overlay = Image.composite(Image.blend(frame, red, 0.5), frame, mask)
    overlay.save(out_dir / f"{name}_overlay.jpg", quality=90)


def save_overview(port: int, out_dir: Path, filename: str = "overview.jpg") -> None:
    with urllib.request.urlopen(f"http://localhost:{port}/api/screenshot", timeout=10) as r:
        (out_dir / filename).write_bytes(r.read())


def spawn_and_wait(port: int, obj_type: str, tolerance: float = 0.05,
                    rotation: str | None = None, spawn_timeout: float = 15.0,
                    travel_timeout: float = 30.0) -> int:
    """Заспавнить объект и дождаться, пока он дойдёт до точки наблюдения
    `cv_rig.x` (в допуске `tolerance`, м). Возвращает uid объекта.
    `rotation="identity"` — детерминированная (не случайная) ориентация спавна,
    см. supervisor_main.py::try_spawn (нужно для численной сверки масок с
    истинными габаритами объекта, см. tools/report_cv_masks.py)."""
    layout = load_layout()
    cv_x = layout["cv_rig"]["x"]

    before = {o["uid"] for o in api_status(port)["objects"]}
    body = {"type": obj_type}
    if rotation is not None:
        body["rotation"] = rotation
    api_post(port, "spawn", body)

    uid = None
    deadline = time.time() + spawn_timeout
    while uid is None and time.time() < deadline:
        for o in api_status(port)["objects"]:
            if o["uid"] not in before:
                uid = o["uid"]
        time.sleep(0.2)
    if uid is None:
        raise RuntimeError(f"объект {obj_type} не заспавнился за {spawn_timeout}с")

    deadline = time.time() + travel_timeout
    while time.time() < deadline:
        for o in api_status(port)["objects"]:
            if o["uid"] == uid and o["x"] is not None and abs(o["x"] - cv_x) <= tolerance:
                return uid
        time.sleep(0.03)
    raise RuntimeError(f"объект {obj_type} (uid={uid}) не дошёл до cv_rig.x={cv_x} за {travel_timeout}с")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8008)
    ap.add_argument("--type", default="box_small", help="тип объекта для спавна, см. config/objects.yaml")
    ap.add_argument("--cameras", default=None, help="камеры через запятую (по умолчанию все из cv_rig.cameras)")
    ap.add_argument("--out", required=True, help="папка для сохранения JPEG")
    ap.add_argument("--overview-only", action="store_true", help="только /api/screenshot, без спавна/CV-камер")
    ap.add_argument("--no-overview", action="store_true", help="не снимать обзорный кадр вместе с CV-камерами")
    ap.add_argument("--tolerance", type=float, default=0.05, help="допуск по X до cv_rig.x, м")
    ap.add_argument("--reset", action="store_true", help="сделать /api/reset перед спавном")
    ap.add_argument("--with-mask", action="store_true",
                    help="дополнительно снять маску фоновой субтракции (только прямые камеры "
                         "top/diag/side, Фаза 2.1) + overlay.jpg для визуальной проверки")
    ap.add_argument("--rotation", choices=["identity"], default=None,
                    help="identity — детерминированная (не случайная) ориентация спавна")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.overview_only:
        save_overview(args.port, out_dir)
        print(f"overview -> {out_dir / 'overview.jpg'}")
        return 0

    layout = load_layout()
    cameras = args.cameras.split(",") if args.cameras else list(layout["cv_rig"]["cameras"])
    direct_cameras = set(layout["cv_rig"]["cameras"]) - {
        n for n, c in layout["cv_rig"]["cameras"].items() if c.get("mirror")}

    if args.reset:
        api_post(args.port, "reset")
        time.sleep(1)

    if not args.no_overview:
        save_overview(args.port, out_dir)

    try:
        spawn_and_wait(args.port, args.type, args.tolerance, args.rotation)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1

    # все камеры запрашиваются конкурентно — контроллер обрабатывает их одним
    # тиком, иначе объект успевает уехать между первой и последней камерой
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(cameras))) as ex:
        futs = {ex.submit(save_camera, args.port, name, out_dir): name for name in cameras}
        failed = []
        for fut in futs:
            try:
                fut.result()
            except Exception as exc:
                failed.append((futs[fut], exc))
        if failed:
            for name, exc in failed:
                print(f"камера {name}: {exc}", file=sys.stderr)
            return 1

    if args.with_mask:
        mask_cameras = [c for c in cameras if c in direct_cameras]
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(mask_cameras))) as ex:
            futs = {ex.submit(save_mask_and_overlay, args.port, name, out_dir): name
                    for name in mask_cameras}
            for fut in futs:
                try:
                    fut.result()
                except Exception as exc:
                    print(f"маска {futs[fut]}: {exc}", file=sys.stderr)
                    return 1
        skipped = set(cameras) - direct_cameras
        if skipped:
            print(f"маски пропущены (зеркальные, Фаза 2.1 их не включает): {sorted(skipped)}")

    print(f"снято {len(cameras)} камер -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
