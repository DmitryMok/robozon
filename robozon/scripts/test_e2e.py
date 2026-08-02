#!/usr/bin/env python3
"""Сквозной тест: каждый тип объекта доезжает до своей зоны.

Требует запущенной симуляции (scripts/run.sh). Спавнит объекты по одному,
ждёт доставки и сверяет фактическую зону с целевой.

    python3 scripts/test_e2e.py [--types bottle,box_small] [--port 8008]
"""
import argparse
import json
import sys
import time
import urllib.request

DELIVERY_TIMEOUT_S = 40


def api(port: int, path: str, body: dict | None = None) -> dict:
    url = f"http://localhost:{port}/api/{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method="POST" if data else "GET")
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def wait_final_state(port: int, uid: int) -> dict:
    deadline = time.time() + DELIVERY_TIMEOUT_S
    while time.time() < deadline:
        status = api(port, "status")
        for obj in status["objects"]:
            if obj["uid"] == uid and obj["state"] in ("delivered", "missed"):
                return obj
        time.sleep(1)
    raise TimeoutError(f"объект {uid} не достиг финального состояния")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8008)
    parser.add_argument("--types", help="список типов через запятую (по умолчанию все)")
    args = parser.parse_args()

    catalog = {o["type"]: o for o in api(args.port, "objects")}
    types = args.types.split(",") if args.types else list(catalog)

    api(args.port, "reset", {})
    time.sleep(1)

    failures = []
    for obj_type in types:
        target = catalog[obj_type]["zone"]
        before = {o["uid"] for o in api(args.port, "status")["objects"]}
        api(args.port, "spawn", {"type": obj_type})

        uid = None
        deadline = time.time() + 15
        while uid is None and time.time() < deadline:
            for o in api(args.port, "status")["objects"]:
                if o["uid"] not in before:
                    uid = o["uid"]
            time.sleep(0.5)
        if uid is None:
            failures.append((obj_type, "не заспавнился"))
            print(f"FAIL  {obj_type:<12} не заспавнился")
            continue

        try:
            final = wait_final_state(args.port, uid)
        except TimeoutError as exc:
            failures.append((obj_type, str(exc)))
            print(f"FAIL  {obj_type:<12} {exc}")
            continue

        ok = final["state"] == "delivered" and final["result"] == target
        mark = "ok  " if ok else "FAIL"
        print(f"{mark}  {obj_type:<12} цель={target} факт={final['result'] or '-'} "
              f"состояние={final['state']}")
        if not ok:
            failures.append((obj_type, f"{final['state']} -> {final['result']}"))

    print()
    if failures:
        print(f"ПРОВАЛЕНО: {len(failures)} из {len(types)}: {failures}")
        return 1
    print(f"ВСЕ {len(types)} ТИПОВ ДОСТАВЛЕНЫ ВЕРНО")
    return 0


if __name__ == "__main__":
    sys.exit(main())
