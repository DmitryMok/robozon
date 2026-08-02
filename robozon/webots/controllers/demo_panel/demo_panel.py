"""Веб-панель demo_roller_shiber: спавн тестовых объектов (клин/конус) по
HTTP-команде вместо фиксированного батча при загрузке мира — для живого
показа на хакатоне (нажал кнопку -> увидел конкретный случай).

Геометрия (wedge_node/cone_node) и константы (SPAWN_X, X_FINISH,
Y_SIDE_THRESHOLD, PRESETS) — из tools/gen_demo_roller_shiber.py (тот же
модуль, что генерирует мир), чтобы не дублировать формулы клина/конуса.
"""
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from controller import Supervisor

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tools.gen_demo_roller_shiber import (  # noqa: E402
    PRESETS, REAL_OBJECTS, SIMPLE_SHAPES, SPAWN_X, X_FINISH,
    Y_SIDE_THRESHOLD, cone_node, real_object_node, simple_shape_node,
    wedge_node,
)

PANEL_PORT = 8020

PAGE_HTML = """<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<title>Demo: роллер-шибер</title>
<style>
  body { font: 14px/1.4 system-ui, sans-serif; margin: 0; padding: 16px 20px;
         background: #14161a; color: #e6e8eb; }
  h1 { font-size: 17px; margin: 0 0 4px; }
  p.sub { color: #9aa1ab; margin: 0 0 16px; }
  fieldset { border: 1px solid #333944; border-radius: 8px; padding: 12px 14px;
             margin-bottom: 14px; }
  legend { color: #9aa1ab; padding: 0 6px; }
  label { display: inline-block; min-width: 90px; }
  input[type=range] { width: 220px; vertical-align: middle; }
  .row { margin: 6px 0; }
  button { background: #2a6df0; color: #fff; border: none; border-radius: 6px;
           padding: 7px 14px; cursor: pointer; font-size: 13px; }
  button.secondary { background: #333944; }
  button:hover { filter: brightness(1.1); }
  .presets button { margin: 3px 6px 3px 0; }
  table { border-collapse: collapse; width: 100%; margin-top: 8px; }
  td, th { padding: 4px 8px; text-align: left; border-bottom: 1px solid #2a2f38; }
  .PASS { color: #4caf50; font-weight: 600; }
  .FAIL { color: #e5534b; font-weight: 600; }
  .traveling { color: #9aa1ab; }
</style></head>
<body>
<h1>Demo: модифицированный шибер (вал с лопастями)</h1>
<p class="sub">Спавн клина/конуса или реального товара на ленте — проверка,
вытолкнет ли вращающийся вал объект вбок или он проедет прямо под ним.</p>

<fieldset>
  <legend>Свободный спавн</legend>
  <div class="row">
    <label>Тип</label>
    <select id="type"></select>
  </div>
  <div class="row">
    <label>Y (поперёк ленты)</label>
    <input type="range" id="y" min="-0.24" max="0.24" step="0.01" value="0">
    <span id="yVal">0.00</span> м
  </div>
  <div class="row">
    <label>Поворот</label>
    <input type="range" id="yaw" min="0" max="180" step="15" value="0">
    <span id="yawVal">0</span>°
  </div>
  <div class="row">
    <button onclick="spawn()">Заспавнить</button>
    <button class="secondary" onclick="clearAll()">Очистить всё</button>
  </div>
</fieldset>

<fieldset>
  <legend>Пресеты (случаи из первого прогона)</legend>
  <div class="presets" id="presets"></div>
</fieldset>

<fieldset>
  <legend>Статус</legend>
  <table>
    <thead><tr><th>Объект</th><th>x</th><th>y</th><th>Исход</th></tr></thead>
    <tbody id="statusBody"></tbody>
  </table>
</fieldset>

<script>
const PRESETS = __PRESETS_JSON__;
const REAL_OBJECTS = __REAL_OBJECTS_JSON__;
const SIMPLE_SHAPES = __SIMPLE_SHAPES_JSON__;

const typeEl = document.getElementById('type');
[["wedge", "Клин (тест)"], ["cone", "Конус (тест)"]].forEach(([v, l]) => {
  const o = document.createElement('option'); o.value = v; o.textContent = l;
  typeEl.appendChild(o);
});
const simpleGroup = document.createElement('optgroup');
simpleGroup.label = 'Простые формы (без STL, для теста скорости)';
SIMPLE_SHAPES.forEach(([type, label]) => {
  const o = document.createElement('option'); o.value = type; o.textContent = label;
  simpleGroup.appendChild(o);
});
typeEl.appendChild(simpleGroup);
const realGroup = document.createElement('optgroup');
realGroup.label = 'Реальные товары';
REAL_OBJECTS.forEach(([type, label]) => {
  const o = document.createElement('option'); o.value = type; o.textContent = label;
  realGroup.appendChild(o);
});
typeEl.appendChild(realGroup);

const yEl = document.getElementById('y'), yValEl = document.getElementById('yVal');
const yawEl = document.getElementById('yaw'), yawValEl = document.getElementById('yawVal');
yEl.oninput = () => yValEl.textContent = parseFloat(yEl.value).toFixed(2);
yawEl.oninput = () => yawValEl.textContent = yawEl.value;

function currentType() {
  return typeEl.value;
}

async function spawnRaw(type, y, yaw) {
  await fetch('/api/spawn', {
    method: 'POST',
    body: JSON.stringify({type, y, yaw}),
  });
}

function spawn() {
  spawnRaw(currentType(), parseFloat(yEl.value), parseFloat(yawEl.value));
}

async function clearAll() {
  await fetch('/api/clear', {method: 'POST'});
}

const presetsDiv = document.getElementById('presets');
PRESETS.forEach(p => {
  const b = document.createElement('button');
  b.textContent = p.label;
  b.onclick = () => { typeEl.value = p.type; yEl.value = p.y; yEl.dispatchEvent(new Event('input'));
                      yawEl.value = p.yaw; yawEl.dispatchEvent(new Event('input')); spawnRaw(p.type, p.y, p.yaw); };
  presetsDiv.appendChild(b);
});

async function poll() {
  try {
    const r = await fetch('/api/status');
    const items = await r.json();
    const body = document.getElementById('statusBody');
    body.innerHTML = items.map(it => {
      const cls = it.outcome === 'PASS' ? 'PASS' : (it.outcome === 'FAIL' ? 'FAIL' : 'traveling');
      return `<tr><td>${it.kind} #${it.uid}</td><td>${it.x.toFixed(2)}</td>` +
             `<td>${it.y.toFixed(2)}</td><td class="${cls}">${it.outcome}</td></tr>`;
    }).join('');
  } catch (e) { /* сервер ещё не готов — просто подождать следующего тика */ }
  setTimeout(poll, 400);
}
poll();
</script>
</body></html>
"""


class Api:
    def __init__(self):
        self.lock = threading.Lock()
        self.spawn_requests = []
        self.clear_requested = False
        self.status = []


def make_handler(api: Api):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code: int, body: bytes, ctype: str = "application/json; charset=utf-8"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                presets_json = json.dumps([
                    {"type": kind, "label": label, "y": y, "yaw": yaw}
                    for kind, label, y, yaw in PRESETS
                ])
                real_objects_json = json.dumps([
                    [obj_type, cfg["label"]] for obj_type, cfg in REAL_OBJECTS.items()
                ])
                simple_shapes_json = json.dumps(SIMPLE_SHAPES)
                page = (PAGE_HTML
                        .replace("__PRESETS_JSON__", presets_json)
                        .replace("__REAL_OBJECTS_JSON__", real_objects_json)
                        .replace("__SIMPLE_SHAPES_JSON__", simple_shapes_json))
                self._send(200, page.encode("utf-8"), "text/html; charset=utf-8")
            elif self.path == "/api/status":
                with api.lock:
                    body = json.dumps(api.status).encode("utf-8")
                self._send(200, body)
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0) or 0)
            raw = self.rfile.read(length) if length else b"{}"
            try:
                payload = json.loads(raw or b"{}")
            except json.JSONDecodeError:
                payload = {}
            if self.path == "/api/spawn":
                with api.lock:
                    api.spawn_requests.append(payload)
                self._send(200, b'{"ok":true}')
            elif self.path == "/api/clear":
                with api.lock:
                    api.clear_requested = True
                self._send(200, b'{"ok":true}')
            else:
                self._send(404, b"not found", "text/plain")

    return Handler


def main() -> None:
    sup = Supervisor()
    timestep = int(sup.getBasicTimeStep())

    api = Api()
    server = ThreadingHTTPServer(("0.0.0.0", PANEL_PORT), make_handler(api))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"[demo] панель: http://localhost:{PANEL_PORT}/", flush=True)

    root_children = sup.getRoot().getField("children")
    tracked: dict[int, dict] = {}
    uid_seq = 0

    while sup.step(timestep) != -1:
        with api.lock:
            spawn_requests = api.spawn_requests
            api.spawn_requests = []
            clear_requested = api.clear_requested
            api.clear_requested = False

        if clear_requested:
            for info in tracked.values():
                if info["node"] is not None:
                    info["node"].remove()
            tracked.clear()

        for payload in spawn_requests:
            kind = payload.get("type", "wedge")
            y = max(-0.24, min(0.24, float(payload.get("y", 0.0))))
            yaw = float(payload.get("yaw", 0.0))
            uid_seq += 1
            name = f"spawn_{uid_seq}"
            if kind == "cone":
                node_str = cone_node(name, SPAWN_X, y)
            elif kind.startswith(("box_", "sphere_")):
                node_str = simple_shape_node(name, kind, SPAWN_X, y)
            elif kind in REAL_OBJECTS:
                node_str = real_object_node(name, kind, SPAWN_X, y, yaw)
            else:
                kind = "wedge"
                node_str = wedge_node(name, SPAWN_X, y, yaw, 1.0)
            root_children.importMFNodeFromString(-1, node_str)
            tracked[uid_seq] = {
                "node": sup.getFromDef(name.upper()),
                "kind": kind,
                "start_y": y,
                "outcome": None,
            }

        status = []
        for uid, info in tracked.items():
            node = info["node"]
            if node is None:
                continue
            x, y, _z = node.getPosition()
            if info["outcome"] is None and x >= X_FINISH:
                dy = abs(y - info["start_y"])
                info["outcome"] = "PASS" if dy >= Y_SIDE_THRESHOLD else "FAIL"
            status.append({
                "uid": uid, "kind": info["kind"],
                "x": round(x, 3), "y": round(y, 3),
                "outcome": info["outcome"] or "едет...",
            })
        with api.lock:
            api.status = status


if __name__ == "__main__":
    main()
