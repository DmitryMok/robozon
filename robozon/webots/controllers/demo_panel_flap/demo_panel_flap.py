"""Веб-панель demo_flap_shiber: спавн тестовых объектов (клин/конус/реальный
товар) по HTTP-команде + РУЧНОЕ управление щитом (кнопки Открыть/Закрыть —
автоцикл убран по просьбе пользователя, см. tools/gen_demo_flap_shiber.py).

Этот Robot — ОДНОВРЕМЕННО supervisor (спавн/статус) И владелец мотора щита
(paddle_flap): geн-скрипт кладёт узел paddle_node прямо в children ЭТОГО
Robot'а (см. demo_panel_node в gen-скрипте), поэтому getDevice("paddle_flap")
работает напрямую, без отдельного контроллера/таймера.

Геометрия (wedge_node/cone_node/real_object_node) и константы (SPAWN_X,
X_FINISH, Y_SIDE_THRESHOLD, PRESETS, CLOSED_ANGLE, OPEN_ANGLE) — из
tools/gen_demo_flap_shiber.py, не дублируются.
"""
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from controller import Supervisor

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tools.gen_demo_flap_shiber import (  # noqa: E402
    BEAM_TRIGGER_DIST, CLOSED_ANGLE, OPEN_ANGLE, PRESETS, REAL_OBJECTS,
    SENSOR_X, SIMPLE_SHAPES, SPAWN_X, TARGET_Y_SIGN, X_FINISH,
    Y_SIDE_THRESHOLD, cone_node, real_object_node, simple_shape_node,
    wedge_node,
)

PANEL_PORT = 8021
ANGLE_TOL = 0.05  # рад — допуск, чтобы отличить "доехал" от "едет"
AUTO_CLOSE_HOLD_S = 0.25  # выдержка в открытом положении перед автозакрытием (pulse)

PAGE_HTML = """<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<title>Demo: шибер на входе в зону</title>
<style>
  body { font: 14px/1.4 system-ui, sans-serif; margin: 0; padding: 16px 20px;
         background: #14161a; color: #e6e8eb; }
  h1 { font-size: 17px; margin: 0 0 4px; }
  p.sub { color: #9aa1ab; margin: 0 0 16px; }
  fieldset { border: 1px solid #333944; border-radius: 8px; padding: 12px 14px;
             margin-bottom: 14px; }
  legend { color: #9aa1ab; padding: 0 6px; }
  label { display: inline-block; min-width: 110px; }
  input[type=range], input[type=number] { width: 220px; vertical-align: middle; }
  .row { margin: 6px 0; }
  button { background: #2a6df0; color: #fff; border: none; border-radius: 6px;
           padding: 7px 14px; cursor: pointer; font-size: 13px; }
  button.secondary { background: #333944; }
  button.stop { background: #7a2a2a; }
  button.open { background: #1f8a4c; }
  button.close { background: #a8531f; }
  button:hover { filter: brightness(1.1); }
  .presets button { margin: 3px 6px 3px 0; }
  table { border-collapse: collapse; width: 100%; margin-top: 8px; }
  td, th { padding: 4px 8px; text-align: left; border-bottom: 1px solid #2a2f38; }
  .PASS { color: #4caf50; font-weight: 600; }
  .FAIL { color: #e5534b; font-weight: 600; }
  .traveling { color: #9aa1ab; }
  #summary { margin-top: 8px; color: #cfd3d9; }
  #gateState { font-weight: 600; }
  #errBanner { display: none; background: #7a2a2a; color: #fff; padding: 8px 12px;
               border-radius: 6px; margin-bottom: 12px; }
</style></head>
<body>
<h1>Demo: шибер на входе в зону (закрыто=барьер, открыто=проём)</h1>
<p class="sub">Зелёная метка на ленте — точка спавна. КРАСНАЯ метка — точка
доставки (сторона __ZONE_SIDE__, совпадает со стороной анкера щита): в
закрытом положении полотно физически лежит НАД красной меткой (вход в зону
перекрыт), открытие тянет объект к ней же — см. tools/gen_demo_flap_shiber.py.
Если красная метка не там, где должна быть зона у вас — это один параметр
(ANCHOR_SIDE), скажите и поменяю. Щит теперь ТОЛЬКО вручную — кнопки ниже.</p>
<div id="errBanner"></div>

<fieldset>
  <legend>Щит</legend>
  <div class="row">
    Состояние: <span id="gateState">?</span> (угол датчика: <span id="gateAngle">?</span>)
  </div>
  <div class="row">
    Луч: <span id="beamState">?</span> (дистанция: <span id="beamDist">?</span> м)
  </div>
  <div class="row">
    <button id="gateToggle" class="open" onclick="toggleGate()">Открыть</button>
    <label style="min-width:auto; margin-left:12px;">
      <input type="checkbox" id="autoClose"> Автозакрытие (открыть и сразу закрыть одной кнопкой)
    </label>
  </div>
  <div class="row">
    <label style="min-width:auto;">
      <input type="checkbox" id="sensorAuto" onchange="sensorAutoChanged()"> Автосрабатывание по датчику (лазер перед щитом)
    </label>
  </div>
</fieldset>

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
  <legend>Поток (стресс-тест интервала)</legend>
  <div class="row">
    <label>Кол-во</label>
    <input type="number" id="streamCount" min="1" max="200" value="20">
  </div>
  <div class="row">
    <label>Интервал, мс</label>
    <input type="number" id="streamInterval" min="50" max="5000" step="50" value="800">
  </div>
  <div class="row">
    <label style="min-width:auto;">
      <input type="checkbox" id="randomMode"> Случайно: кубики и шары разного размера
      (вместо типа выше), Y ±100 мм от центра — щит должен реагировать только
      на шары (включите «Автосрабатывание по датчику»)
    </label>
  </div>
  <div class="row">
    <button onclick="startStream()">Запустить поток</button>
    <button class="secondary stop" onclick="stopStream()">Остановить</button>
  </div>
</fieldset>

<fieldset>
  <legend>Пресеты</legend>
  <div class="presets" id="presets"></div>
</fieldset>

<fieldset>
  <legend>Статус</legend>
  <div id="summary"></div>
  <table>
    <thead><tr><th>Объект</th><th>x</th><th>y</th><th>Исход</th></tr></thead>
    <tbody id="statusBody"></tbody>
  </table>
</fieldset>

<script>
const PRESETS = __PRESETS_JSON__;
const REAL_OBJECTS = __REAL_OBJECTS_JSON__;
const SIMPLE_SHAPES = __SIMPLE_SHAPES_JSON__;

function showError(msg) {
  const el = document.getElementById('errBanner');
  el.textContent = msg;
  el.style.display = 'block';
}
function clearError() {
  document.getElementById('errBanner').style.display = 'none';
}

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

function currentType() { return typeEl.value; }

async function postJson(path, body) {
  const r = await fetch(path, { method: 'POST', body: JSON.stringify(body || {}) });
  if (!r.ok) throw new Error(`${path} -> HTTP ${r.status}`);
  clearError();
  return r;
}

async function spawnRaw(type, y, yaw) {
  try {
    await postJson('/api/spawn', {type, y, yaw});
  } catch (e) { showError('Спавн не прошёл: ' + e.message); }
}

function spawn() {
  spawnRaw(currentType(), parseFloat(yEl.value), parseFloat(yawEl.value));
}

async function clearAll() {
  try { await postJson('/api/clear'); } catch (e) { showError('Очистка не прошла: ' + e.message); }
}

async function gateCmd(cmd) {
  try { await postJson('/api/gate/' + cmd); } catch (e) { showError('Команда щита не прошла: ' + e.message); }
}

async function sensorAutoChanged() {
  const enabled = document.getElementById('sensorAuto').checked;
  try { await postJson('/api/gate/sensor_auto', {enabled}); }
  catch (e) { showError('Не удалось переключить автосрабатывание: ' + e.message); }
}

// Локальное (не из poll) состояние кнопки — чтобы быстрые повторные клики не
// путались с задержкой опроса /api/status (мотор физически едет к цели не
// мгновенно, а poll раз в 400мс): кнопка переключает НАМЕРЕНИЕ сразу же.
let gateOpenIntent = false;  // старт — ЗАКРЫТО (см. demo_panel_flap.py::main)
const gateBtn = document.getElementById('gateToggle');
const autoCloseEl = document.getElementById('autoClose');
function renderGateBtn() {
  if (autoCloseEl.checked) {
    gateBtn.textContent = 'Открыть (авто-закрытие)';
    gateBtn.className = 'open';
    return;
  }
  gateBtn.textContent = gateOpenIntent ? 'Закрыть' : 'Открыть';
  gateBtn.className = gateOpenIntent ? 'close' : 'open';
}
function toggleGate() {
  if (autoCloseEl.checked) {
    // Автозакрытие: щит сам вернётся в закрытое положение, как только
    // доедет до открытого (см. awaiting_auto_close в demo_panel_flap.py) —
    // кнопка не хранит "намерение", каждый клик = независимый цикл.
    gateCmd('pulse');
    return;
  }
  gateOpenIntent = !gateOpenIntent;
  renderGateBtn();
  gateCmd(gateOpenIntent ? 'open' : 'close');
}
autoCloseEl.onchange = () => {
  gateOpenIntent = false;  // после любого pulse щит всегда заканчивает закрытым
  renderGateBtn();
};
renderGateBtn();

const randomModeEl = document.getElementById('randomMode');

function randomSpawnYMm() {
  // ±100 мм от центра ленты (не путать со слайдером Y свободного спавна).
  return Math.random() * 0.2 - 0.1;
}

let streamTimer = null;
function startStream() {
  stopStream();
  const count = parseInt(document.getElementById('streamCount').value, 10);
  const interval = parseInt(document.getElementById('streamInterval').value, 10);
  const random = randomModeEl.checked;
  let sent = 0;
  streamTimer = setInterval(() => {
    if (sent >= count) { stopStream(); return; }
    if (random) {
      const [type] = SIMPLE_SHAPES[Math.floor(Math.random() * SIMPLE_SHAPES.length)];
      spawnRaw(type, randomSpawnYMm(), 0);
    } else {
      spawnRaw(currentType(), parseFloat(yEl.value), parseFloat(yawEl.value));
    }
    sent++;
  }, interval);
}
function stopStream() {
  if (streamTimer) { clearInterval(streamTimer); streamTimer = null; }
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
    if (!r.ok) throw new Error('HTTP ' + r.status);
    const data = await r.json();
    clearError();
    document.getElementById('gateState').textContent = data.gate_state;
    document.getElementById('gateAngle').textContent = data.gate_angle.toFixed(3);
    document.getElementById('beamState').textContent = data.beam_blocked ? 'ПЕРЕКРЫТ' : 'свободен';
    document.getElementById('beamDist').textContent = data.beam_distance.toFixed(3);
    const items = data.items;
    const body = document.getElementById('statusBody');
    body.innerHTML = items.map(it => {
      const cls = it.outcome === 'PASS' ? 'PASS' : (it.outcome === 'FAIL' ? 'FAIL' : 'traveling');
      return `<tr><td>${it.kind} #${it.uid}</td><td>${it.x.toFixed(2)}</td>` +
             `<td>${it.y.toFixed(2)}</td><td class="${cls}">${it.outcome}</td></tr>`;
    }).join('');
    const done = items.filter(it => it.outcome === 'PASS' || it.outcome === 'FAIL');
    const passN = done.filter(it => it.outcome === 'PASS').length;
    document.getElementById('summary').textContent = done.length
      ? `${passN}/${done.length} PASS (${(100 * passN / done.length).toFixed(0)}%)`
      : 'нет завершённых объектов';
  } catch (e) {
    showError('Панель не отвечает (' + e.message + ') — проверьте, что мир загружен в Webots и вкладка controller demo_panel_flap запущена.');
  }
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
        self.gate_command = None  # "open" | "close" | "pulse" | None
        self.sensor_auto = False  # автосрабатывание pulse по лучу
        self.status = {"gate_state": "?", "gate_angle": 0.0,
                        "beam_blocked": False, "beam_distance": 0.0, "items": []}


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
                zone_side = "y>0" if TARGET_Y_SIGN > 0 else "y<0"
                page = (PAGE_HTML
                        .replace("__PRESETS_JSON__", presets_json)
                        .replace("__REAL_OBJECTS_JSON__", real_objects_json)
                        .replace("__SIMPLE_SHAPES_JSON__", simple_shapes_json)
                        .replace("__ZONE_SIDE__", zone_side))
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
                print(f"[demo] spawn request: {payload}", flush=True)
                with api.lock:
                    api.spawn_requests.append(payload)
                self._send(200, b'{"ok":true}')
            elif self.path == "/api/clear":
                with api.lock:
                    api.clear_requested = True
                self._send(200, b'{"ok":true}')
            elif self.path in ("/api/gate/open", "/api/gate/close", "/api/gate/pulse"):
                cmd = self.path.rsplit("/", 1)[-1]
                print(f"[demo] gate command: {cmd}", flush=True)
                with api.lock:
                    api.gate_command = cmd
                self._send(200, b'{"ok":true}')
            elif self.path == "/api/gate/sensor_auto":
                enabled = bool(payload.get("enabled", False))
                print(f"[demo] sensor_auto: {enabled}", flush=True)
                with api.lock:
                    api.sensor_auto = enabled
                self._send(200, b'{"ok":true}')
            else:
                self._send(404, b"not found", "text/plain")

    return Handler


def main() -> None:
    sup = Supervisor()
    timestep = int(sup.getBasicTimeStep())

    motor = sup.getDevice("paddle_flap")
    sensor = sup.getDevice("paddle_flap_sensor")
    sensor.enable(timestep)
    motor.setPosition(CLOSED_ANGLE)  # старт — ЗАКРЫТО (барьер, вход перекрыт)

    beam_sensor = sup.getDevice("beam_sensor")
    beam_sensor.enable(timestep)

    api = Api()
    server = ThreadingHTTPServer(("0.0.0.0", PANEL_PORT), make_handler(api))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"[demo] панель: http://localhost:{PANEL_PORT}/", flush=True)

    root_children = sup.getRoot().getField("children")
    tracked: dict[int, dict] = {}
    uid_seq = 0
    awaiting_auto_close = False  # "pulse": закрыть после выдержки на открытом
    auto_close_at = None         # sup.getTime(), когда пора закрыть (None — ещё едет к открытому)
    beam_was_blocked = False     # для срабатывания по фронту (не на каждом тике перекрытия)

    while sup.step(timestep) != -1:
        with api.lock:
            spawn_requests = api.spawn_requests
            api.spawn_requests = []
            clear_requested = api.clear_requested
            api.clear_requested = False
            gate_command = api.gate_command
            api.gate_command = None
            sensor_auto = api.sensor_auto

        beam_distance = beam_sensor.getValue()
        beam_blocked = beam_distance < BEAM_TRIGGER_DIST
        if sensor_auto and beam_blocked and not beam_was_blocked:
            # Избирательное автосрабатывание: кубики (box_*, режим "Случайно"
            # в потоке) должны проехать транзитом без реакции щита — шибер
            # реагирует на всё остальное (в первую очередь на шары). Ищем
            # объект, чей центр сейчас ближе всего к датчику (тот, что и
            # перекрыл луч), и смотрим его kind.
            nearest_kind, nearest_dist = None, None
            for info in tracked.values():
                if info["outcome"] is not None or info["node"] is None:
                    continue
                x, _y, _z = info["node"].getPosition()
                d = abs(x - SENSOR_X)
                if nearest_dist is None or d < nearest_dist:
                    nearest_dist, nearest_kind = d, info["kind"]
            if nearest_kind is None or not nearest_kind.startswith("box_"):
                gate_command = "pulse"  # луч только что перекрыт -> тот же цикл, что и кнопка
        beam_was_blocked = beam_blocked

        if gate_command == "open":
            motor.setPosition(OPEN_ANGLE)
            awaiting_auto_close = False
        elif gate_command == "close":
            motor.setPosition(CLOSED_ANGLE)
            awaiting_auto_close = False
        elif gate_command == "pulse":
            motor.setPosition(OPEN_ANGLE)
            awaiting_auto_close = True
            auto_close_at = None

        if awaiting_auto_close:
            if auto_close_at is None:
                if abs(sensor.getValue() - OPEN_ANGLE) <= ANGLE_TOL:
                    auto_close_at = sup.getTime() + AUTO_CLOSE_HOLD_S
            elif sup.getTime() >= auto_close_at:
                motor.setPosition(CLOSED_ANGLE)
                awaiting_auto_close = False
                auto_close_at = None

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

        items = []
        for uid, info in tracked.items():
            node = info["node"]
            if node is None:
                continue
            x, y, _z = node.getPosition()
            if info["outcome"] is None:
                # PASS проверяем СРАЗУ по Y, не дожидаясь X_FINISH: объект,
                # отклонённый за край ленты, падает на статичный пол (зона
                # абстрактная, без ската) и там останавливается — дальше по X
                # он может больше НИКОГДА не сдвинуться, а X_FINISH так и не
                # наступит (найдено 2026-07-29 живым прогоном: объект застыл
                # на x=3.759, "едет..." висело бесконечно, хотя уже ушёл в
                # y=0.56, явный PASS). FAIL, наоборот, ещё может оставаться
                # рано неверным (X_FINISH — момент, когда уже точно "проехал
                # щит насквозь", а не отклонился), поэтому FAIL проверяем как
                # раньше, только по X_FINISH.
                if y * TARGET_Y_SIGN >= Y_SIDE_THRESHOLD:
                    info["outcome"] = "PASS"
                elif x >= X_FINISH:
                    info["outcome"] = "FAIL"
            items.append({
                "uid": uid, "kind": info["kind"],
                "x": round(x, 3), "y": round(y, 3),
                "outcome": info["outcome"] or "едет...",
            })

        angle = sensor.getValue()
        if abs(angle - CLOSED_ANGLE) <= ANGLE_TOL:
            gate_state = "ЗАКРЫТО"
        elif abs(angle - OPEN_ANGLE) <= ANGLE_TOL:
            gate_state = "ОТКРЫТО"
        else:
            gate_state = "движется"

        with api.lock:
            api.status = {"gate_state": gate_state, "gate_angle": angle,
                          "beam_blocked": beam_blocked, "beam_distance": round(beam_distance, 3),
                          "items": items}


if __name__ == "__main__":
    main()
