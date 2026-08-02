"""Веб-панель demo_mechanism: спавн тестовых объектов (реальные товары/
простые формы) по HTTP-команде + ручное/автоматическое управление щитами
шиберов 1/2/3 (все три — "загребание", узел flap_node_upstream, см.
tools/gen_demo_mechanism.py — шаги 2-4/6: анкер каждого щита стоит на ТОЙ ЖЕ
стороне, что и его целевая зона — см. правку 2026-07-30 в докстринге
gen_demo_mechanism.py). Накопитель — в следующем шаге.

Геометрия/константы — из tools/gen_demo_mechanism.py (тот же модуль, что
генерирует мир), чтобы не дублировать формулы. Управление всеми щитами —
тот же паттерн (ручные кнопки Открыть/Закрыть + автосрабатывание по
лучевому датчику), что webots/controllers/demo_panel_flap/demo_panel_flap.py.
"""
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from controller import Supervisor

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tools.gen_demo_mechanism import (  # noqa: E402
    BEAM_TRIGGER_DIST, CLOSED1_ANGLE, CLOSED2_ANGLE, CLOSED3_ANGLE, CV_RIG,
    OPEN1_ANGLE, OPEN2_ANGLE, OPEN3_ANGLE, REAL_OBJECTS, SEG_HIGH_END_X,
    SEG_HIGH_X_START, SEG_LOW_END_X, SEG_LOW_X_START, SENSOR1_X, SENSOR2_X,
    SENSOR3_X, SHIBER1_CATEGORY, SHIBER1_X, SHIBER2_CATEGORY, SHIBER2_X,
    SHIBER3_CATEGORY, SHIBER3_X, SIMPLE_SHAPES, SPAWN_X, real_object_node,
    simple_shape_node,
)

PANEL_PORT = 8022
# Порт стрима Webots (--stream "--port=$STREAM_PORT" в run_demo_mechanism.ps1/.sh) — не 1234
# (занят sorting_line/run.sh) и не 8008/8010/8022.
STREAM_PORT = 1235
ANGLE_TOL = 0.05  # рад — допуск "доехал" от "едет" (как в demo_panel_flap.py)
AUTO_CLOSE_HOLD_S = 0.35  # выдержка в открытом положении перед автозакрытием (pulse)

WEB_DIR = ROOT / "web"
MESH_DIR = ROOT / "assets" / "meshes"
ASSETS_DIR = ROOT / "assets"
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
}
# Статика для 3D-стрима (см. web/index.html — тот же паттерн `<webots-view>`/`WebotsView.js`,
# только встроен в эту панель вместо тормозящего опроса /api/cv_top.jpg, см. ниже). /meshes/ —
# та же необходимость, что в supervisor_main.py: браузерный вьюер при --stream не может читать
# локальный диск, `gen_demo_mechanism.py::mesh_templates_node` теперь ссылается на STL по HTTP
# на этот же порт (правка 2026-07-31, до этого висело на "Downloading assets" — 404). /assets/ —
# то же самое для ImageTexture (ArUco-метки на зеркале/ленте + подписи C/D на кейджах,
# `gen_demo_mechanism.py::_aruco_marker_shape`): относительный `../../assets/...` при --stream
# разрешается против origin стрим-сервера (:1235) и даёт 404, вьюер висит на загрузке текстур.
STATIC_ROUTES = {
    "/wwi/": WEB_DIR / "wwi",
    "/meshes/": MESH_DIR,
    "/assets/": ASSETS_DIR,
}

PAGE_HTML = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Demo: исполнительный механизм</title>
<style>
  :root {
    --bg: #14171c; --panel: #1d2229; --line: #2c333d;
    --text: #dfe5ec; --dim: #8a94a3; --accent: #4da3ff;
    --ok: #43b581; --warn: #e6a23c; --err: #e05c5c;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--text);
    font: 14px/1.45 system-ui, "Segoe UI", sans-serif;
    display: grid; grid-template-columns: 340px 1fr; grid-template-rows: 48px 1fr;
    grid-template-areas: "hdr hdr" "side view"; height: 100vh;
  }
  header {
    grid-area: hdr; display: flex; align-items: center; gap: 12px;
    padding: 0 16px; background: var(--panel); border-bottom: 1px solid var(--line);
  }
  header h1 { font-size: 15px; margin: 0; font-weight: 600; }
  header .sub { color: var(--dim); font-size: 12px; }
  header .cv-report-link { color: var(--accent); font-size: 12px; text-decoration: none; }
  header .cv-report-link:hover { text-decoration: underline; }
  .view-toggle { display: flex; align-items: center; gap: 5px; color: var(--dim); font-size: 12px; }
  #sim-time { margin-left: auto; color: var(--dim); font-variant-numeric: tabular-nums; }
  aside {
    grid-area: side; overflow-y: auto; background: var(--panel);
    border-right: 1px solid var(--line); padding: 12px;
  }
  main { grid-area: view; position: relative; }
  webots-view { display: block; width: 100%; height: 100%; }
  h2 { font-size: 11px; text-transform: uppercase; letter-spacing: .08em;
       color: var(--dim); margin: 14px 0 6px; }
  .btns { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
  .btns.three { grid-template-columns: 1fr 1fr 1fr; }
  button {
    background: #262d37; color: var(--text); border: 1px solid var(--line);
    border-radius: 6px; padding: 7px 8px; cursor: pointer; text-align: left;
    font: inherit; font-size: 13px;
  }
  button:hover { border-color: var(--accent); }
  button .zone { float: right; font-size: 11px; color: var(--dim); }
  button.reset { width: 100%; margin-top: 8px; text-align: center; color: var(--warn); }
  button.open { color: var(--ok); }
  button.close { color: var(--err); }
  .row { margin: 6px 0; display: flex; align-items: center; gap: 6px; }
  .row label { min-width: 90px; color: var(--dim); font-size: 12px; }
  .row input[type=range] { flex: 1; }
  .row .val { font-variant-numeric: tabular-nums; font-size: 12px; min-width: 38px; text-align: right; }
  .chk { font-size: 12px; color: var(--dim); }
  .chk input { vertical-align: middle; margin-right: 4px; }
  .gate-box { border: 1px solid var(--line); border-radius: 6px; padding: 8px 10px; margin: 6px 0; }
  .gate-box .legend { color: var(--accent); font-size: 12px; margin-bottom: 4px; }
  .gate-box .info { font-size: 11px; color: var(--dim); margin: 2px 0; font-variant-numeric: tabular-nums; }
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  td, th { padding: 3px 4px; border-bottom: 1px solid var(--line); text-align: left; }
  th { color: var(--dim); font-weight: 500; }
  #conn { font-size: 12px; }
  .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%;
         background: var(--err); margin-right: 5px; }
  .dot.on { background: var(--ok); }
</style>
</head>
<body>
<header>
  <h1>Demo: исполнительный механизм</h1>
  <span class="sub">3 шибера + накопитель — лента A → кейдж 1 (round) / зона B (ok) / кейдж 2 (oversize)</span>
  <a class="cv-report-link" id="cv-report-link" target="_blank" rel="noopener">CV-отчёт по STL &#8599;</a>
  <label class="view-toggle">
    <input type="checkbox" id="view-toggle" onchange="toggleView()"> 3D-вид
  </label>
  <span id="sim-time"></span>
</header>

<aside>
  <div id="conn"><span class="dot" id="conn-dot"></span><span id="conn-text">подключение…</span></div>

  <h2>Положить на ленту</h2>
  <div class="btns" id="spawn-buttons"></div>
  <div class="row">
    <label>Y (поперёк)</label>
    <input type="range" id="y" min="-0.24" max="0.24" step="0.01" value="0">
    <span class="val" id="yVal">0.00</span>
  </div>
  <div class="row">
    <label>Поворот</label>
    <input type="range" id="yaw" min="0" max="180" step="15" value="0">
    <span class="val" id="yawVal">0&#176;</span>
  </div>
  <button class="reset" onclick="clearAll()">Очистить всё</button>

  <h2>Щиты</h2>
  <div class="gate-box" id="gate1-box">
    <div class="legend">Щит 1 &#8594; кейдж 1 (round)</div>
    <div class="info">Состояние: <span id="gate1State">?</span> (угол <span id="gate1Angle">?</span>)</div>
    <div class="info">Луч: <span id="beam1State">?</span> (<span id="beam1Dist">?</span> м)</div>
    <div class="btns">
      <button id="gate1Toggle" class="open" onclick="toggleGate1()">Открыть</button>
      <button id="gate1Pulse" onclick="gateCmd(1,'pulse')">Открыть+закрыть</button>
    </div>
    <label class="chk"><input type="checkbox" id="sensorAuto1" onchange="sensorAutoChanged(1)" checked> Автосрабатывание по датчику</label>
  </div>
  <div class="gate-box" id="gate2-box">
    <div class="legend">Щит 2 &#8594; зона B (ok)</div>
    <div class="info">Состояние: <span id="gate2State">?</span> (угол <span id="gate2Angle">?</span>)</div>
    <div class="info">Луч: <span id="beam2State">?</span> (<span id="beam2Dist">?</span> м)</div>
    <div class="btns">
      <button id="gate2Toggle" class="open" onclick="toggleGate2()">Открыть</button>
      <button id="gate2Pulse" onclick="gateCmd(2,'pulse')">Открыть+закрыть</button>
    </div>
    <label class="chk"><input type="checkbox" id="sensorAuto2" onchange="sensorAutoChanged(2)" checked> Автосрабатывание по датчику</label>
  </div>
  <div class="gate-box" id="gate3-box">
    <div class="legend">Щит 3 &#8594; кейдж 2 (oversize)</div>
    <div class="info">Состояние: <span id="gate3State">?</span> (угол <span id="gate3Angle">?</span>)</div>
    <div class="info">Луч: <span id="beam3State">?</span> (<span id="beam3Dist">?</span> м)</div>
    <div class="btns">
      <button id="gate3Toggle" class="open" onclick="toggleGate3()">Открыть</button>
      <button id="gate3Pulse" onclick="gateCmd(3,'pulse')">Открыть+закрыть</button>
    </div>
    <label class="chk"><input type="checkbox" id="sensorAuto3" onchange="sensorAutoChanged(3)" checked> Автосрабатывание по датчику</label>
  </div>

  <h2>Объекты на ленте</h2>
  <table>
    <thead><tr><th>Объект</th><th>Кат.</th><th>x</th><th>y</th><th>Сегм.</th></tr></thead>
    <tbody id="statusBody"></tbody>
  </table>
</aside>

<main>
  <webots-view></webots-view>
</main>

<script type="module" src="/wwi/WebotsView.js"></script>
<script>
const STREAM_PORT = __STREAM_PORT__;
// CV-отчёт — независимый процесс на :8010 (см. run_demo_mechanism.sh).
document.getElementById('cv-report-link').href =
  `${location.protocol}//${location.hostname}:8010/`;

const yEl = document.getElementById('y'), yValEl = document.getElementById('yVal');
const yawEl = document.getElementById('yaw'), yawValEl = document.getElementById('yawVal');
yEl.oninput = () => yValEl.textContent = parseFloat(yEl.value).toFixed(2);
yawEl.oninput = () => yawValEl.textContent = yawEl.value + '\u00b0';

async function api(path, body) {
  const opts = body !== undefined
    ? {method: 'POST', body: JSON.stringify(body)}
    : (['clear'].includes(path) ? {method: 'POST'} : {});
  const res = await fetch('/api/' + path, opts);
  return res.json();
}

async function clearAll() { await api('clear'); }

// Кнопки спавна — на каждый объект (как web/index.html панели sorting_line),
// один клик = спавн. Y/yaw берутся из ползунков выше (по умолчанию центр/0°).
async function initButtons() {
  const objects = await api('objects');
  const box = document.getElementById('spawn-buttons');
  for (const o of objects) {
    const btn = document.createElement('button');
    btn.innerHTML = `${o.label} <span class="zone">${o.category}</span>`;
    btn.onclick = () => {
      api('spawn', {type: o.type, y: parseFloat(yEl.value), yaw: parseFloat(yawEl.value)});
    };
    box.appendChild(btn);
  }
}

// Щиты: gateN_command open/close/pulse, sensor_auto {enabled}.
// Локальное состояние кнопки «Открыть/Закрыть» — щит едет не мгновенно, а poll
// раз в 400мс — кнопка переключает НАМЕРЕНИЕ сразу (как demo_panel_flap.py).
let gateIntent = [false, false, false];  // старт — все ЗАКРЫТО
const gateBtns = [document.getElementById('gate1Toggle'),
                  document.getElementById('gate2Toggle'),
                  document.getElementById('gate3Toggle')];

function renderGateBtn(n) {
  const btn = gateBtns[n - 1];
  btn.textContent = gateIntent[n - 1] ? 'Закрыть' : 'Открыть';
  btn.className = gateIntent[n - 1] ? 'close' : 'open';
}
function toggleGate(n) {
  gateIntent[n - 1] = !gateIntent[n - 1];
  renderGateBtn(n);
  gateCmd(n, gateIntent[n - 1] ? 'open' : 'close');
}
function toggleGate1() { toggleGate(1); }
function toggleGate2() { toggleGate(2); }
function toggleGate3() { toggleGate(3); }

async function gateCmd(n, cmd) {
  try { await fetch(`/api/gate${n}/${cmd}`, {method: 'POST'}); } catch (e) { /* ignore */ }
}
async function sensorAutoChanged(n) {
  const enabled = document.getElementById(`sensorAuto${n}`).checked;
  try { await fetch(`/api/gate${n}/sensor_auto`, {method: 'POST', body: JSON.stringify({enabled})}); }
  catch (e) { /* ignore */ }
}
for (let n = 1; n <= 3; n++) renderGateBtn(n);

let lastStatus = null;
function renderStatus(s) {
  lastStatus = s;
  document.getElementById('sim-time').textContent = '';
  // Решение о 3D-виде принимается ОДИН раз по первому /api/status (поле headless),
  // дальше — только вручную чекбоксом (сохраняется в localStorage). Тот же приём,
  // что web/index.html панели sorting_line: при оконном запуске (headless=false) у
  // Webots уже есть нативное окно, повторный рендер в браузере незачем по умолчанию.
  if (viewEnabled === null) {
    const saved = localStorage.getItem('robozon_demo_view');
    applyView(saved !== null ? saved === '1' : !!s.headless);
  }
  for (let n = 1; n <= 3; n++) {
    // Состояние щита из статуса синхронизирует локальное намерение, когда щит
    // доехал (например, после автосрабатывания по датчику — он сам открылся/
    // закрылся, кнопка должна за ним следовать).
    const st = (s[`gate${n}_state`] || '').toLowerCase();
    if (st === 'open' || st === 'открыт') gateIntent[n - 1] = true;
    else if (st === 'closed' || st === 'закрыт') gateIntent[n - 1] = false;
    renderGateBtn(n);
    document.getElementById(`gate${n}State`).textContent = s[`gate${n}_state`];
    document.getElementById(`gate${n}Angle`).textContent = (s[`gate${n}_angle`] || 0).toFixed(3);
    document.getElementById(`beam${n}State`).textContent =
      s[`beam${n}_blocked`] ? 'ПЕРЕКРЫТ' : 'свободен';
    document.getElementById(`beam${n}Dist`).textContent = (s[`beam${n}_distance`] || 0).toFixed(3);
  }
  const body = document.getElementById('statusBody');
  const items = s.items || [];
  body.innerHTML = items.map(it =>
    `<tr><td>${it.kind} #${it.uid}</td><td>${it.category}</td>` +
    `<td>${(it.x || 0).toFixed(2)}</td><td>${(it.y || 0).toFixed(2)}</td>` +
    `<td>${it.segment}</td></tr>`
  ).join('') || '<tr><td colspan="5" style="color:var(--dim)">—</td></tr>';
}

async function poll() {
  try {
    const r = await fetch('/api/status');
    renderStatus(await r.json());
    setConn(true);
  } catch (e) {
    setConn(false);
  }
  setTimeout(poll, 400);
}

function setConn(ok) {
  document.getElementById('conn-dot').className = 'dot' + (ok ? ' on' : '');
  document.getElementById('conn-text').textContent =
    ok ? 'симуляция на связи' : 'нет связи с симуляцией';
}

function connectView() {
  const view = document.querySelector('webots-view');
  if (typeof view.connect !== 'function') {  // модуль ещё грузится
    setTimeout(connectView, 300);
    return;
  }
  view.ondisconnect = () => { if (viewEnabled) setTimeout(connectView, 2000); };
  view.connect(`ws://${location.hostname}:${STREAM_PORT}`, 'w3d', false, false, -1);
}

// При оконном запуске Webots (нативное окно уже есть) 3D-вид в браузере по
// умолчанию выключен — иначе сцена рендерится дважды впустую. Решение — один
// раз по первому /api/status (поле headless), дальше вручную (localStorage).
let viewEnabled = null;

function applyView(enabled) {
  viewEnabled = enabled;
  document.getElementById('view-toggle').checked = enabled;
  const view = document.querySelector('webots-view');
  if (enabled) connectView();
  else if (typeof view.close === 'function') view.close();
}

function toggleView() {
  const enabled = document.getElementById('view-toggle').checked;
  localStorage.setItem('robozon_demo_view', enabled ? '1' : '0');
  applyView(enabled);
}

initButtons();
poll();
</script>
</body>
</html>
"""


CV_TOP_SCREENSHOT_PATH = Path("/tmp/robozon_demo_mechanism_cv_top.jpg")


class Api:
    def __init__(self):
        self.lock = threading.Lock()
        self.spawn_requests = []
        self.clear_requested = False
        # "headless" — тот же сигнал, что supervisor_main.py (через env
        # ROBOZON_HEADLESS из run_demo_mechanism.sh): 0 при оконном запуске (у
        # Webots уже есть нативное окно, 3D-вид в браузере по умолчанию незачем —
        # чекбокс для ручного включения остаётся, как в web/index.html), 1/нет —
        # headless, стрим нужен. JS читает это поле в renderStatus (один раз).
        self.status = {"items": [], "gate1_state": "?", "gate1_angle": 0.0,
                        "beam1_blocked": False, "beam1_distance": 0.0,
                        "gate2_state": "?", "gate2_angle": 0.0,
                        "beam2_blocked": False, "beam2_distance": 0.0,
                        "gate3_state": "?", "gate3_angle": 0.0,
                        "beam3_blocked": False, "beam3_distance": 0.0,
                        "headless": os.environ.get("ROBOZON_HEADLESS") != "0"}
        # Вид камеры "top" по запросу — тот же паттерн enable/settle/
        # saveImage/disable, что /api/screenshot в supervisor_main.py:
        # камера по умолчанию выключена (постоянный рендер 2592x1944 не
        # нужен, пока панель никто не смотрит), включается только пока
        # приходят запросы с панели.
        self.cv_top_requested = threading.Event()
        self.cv_top_done = threading.Event()
        # Щит 1 — команда open/close/pulse (как gate_command в demo_panel_flap.py)
        # + флаг автосрабатывания по лучевому датчику. Автосрабатывание включено
        # по умолчанию у всех трёх щитов (логично для демо: щит должен сам
        # реагировать на товар по своей категории, не ждать ручного клика) —
        # чекбокс в HTML тоже стоит checked, чтобы не было рассинхрона.
        self.gate1_command = None
        self.sensor1_auto = True
        # Щит 2 — команда open/close/pulse (та же схема, что щит 1 — оба
        # используют flap_node_upstream, см. докстринг модуля).
        self.gate2_command = None
        self.sensor2_auto = True
        # Щит 3 — та же схема.
        self.gate3_command = None
        self.sensor3_auto = True


def segment_of(x: float) -> str:
    if x < SEG_LOW_END_X:
        return "low"
    if x < SEG_HIGH_X_START:
        return "ramp"
    if x < SEG_HIGH_END_X:
        return "high"
    return "конец ленты"


def category_of(kind: str) -> str:
    """Категория товара (как sim/router.py::CATEGORY_TO_ZONE — ok/oversize/
    round) — для реальных товаров берётся из config/objects.yaml, для
    простых форм (без STL, только для теста) назначается эвристически: шар
    визуально круглый (round, как шибер 1), куб — как ok (шибер 2). Нужна
    для избирательного автосрабатывания (см. _nearest_category ниже) — без
    неё оба щита при автосрабатывании реагировали бы на ЛЮБОЙ товар."""
    if kind in REAL_OBJECTS:
        return REAL_OBJECTS[kind]["category"]
    if kind.startswith("ball_"):
        return "round"
    return "ok"


def _nearest_category(tracked: dict, x_ref: float) -> str | None:
    """Категория товара, чей центр сейчас ближе всего к датчику x_ref — тот
    же приём выбора "кто перекрыл луч", что nearest_kind в
    demo_panel_flap.py."""
    nearest_cat, nearest_dist = None, None
    for info in tracked.values():
        node = info["node"]
        if node is None:
            continue
        x, _y, _z = node.getPosition()
        d = abs(x - x_ref)
        if nearest_dist is None or d < nearest_dist:
            nearest_dist, nearest_cat = d, info["category"]
    return nearest_cat


def make_handler(api: Api):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code: int, body: bytes, ctype: str = "application/json; charset=utf-8"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            # CORS: браузерный webotsJS (ImageLoader.js#loadImage) ставит crossOrigin='' на
            # все текстуры, а MeshLoader.js использует fetch — оба требуют CORS-заголовок,
            # если origin запроса (127.0.0.1:8022, см. gen_demo_mechanism.py::mesh_templates_node
            # и _aruco_marker_shape) отличается от origin страницы (localhost:8022). Без этого
            # заголовка браузер блокирует текстуры/меши (net::ERR_FAILED 200, "No 'Access-Control-
            # Allow-Origin' header"), вьюер виснет на "Downloading assets" / "Parsing" —
            # тот же фикс, что в supervisor_main.py::_send (там стрим sorting_line работает).
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
                page = PAGE_HTML.replace("__STREAM_PORT__", str(STREAM_PORT))
                self._send(200, page.encode("utf-8"), "text/html; charset=utf-8")
            elif self.path == "/api/objects":
                # Список объектов для кнопок спавна (как supervisor_main.py:515) —
                # реальные товары (REAL_OBJECTS) + простые формы (SIMPLE_SHAPES),
                # JS строит кнопку на каждый (один клик = спавн, как web/index.html).
                items = [{"type": t, "label": c["label"], "category": c["category"]}
                         for t, c in REAL_OBJECTS.items()]
                for stype, slabel in SIMPLE_SHAPES:
                    items.append({"type": stype, "label": slabel,
                                  "category": category_of(stype)})
                self._send(200, json.dumps(items, ensure_ascii=False).encode())
            elif self.path == "/api/status":
                with api.lock:
                    body = json.dumps(api.status).encode("utf-8")
                self._send(200, body)
            elif self.path.startswith("/api/cv_top.jpg"):
                api.cv_top_done.clear()
                api.cv_top_requested.set()
                if api.cv_top_done.wait(timeout=3) and CV_TOP_SCREENSHOT_PATH.exists():
                    self._send(200, CV_TOP_SCREENSHOT_PATH.read_bytes(), "image/jpeg")
                else:
                    self._send(503, b'{"error": "cv_top screenshot failed"}')
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
            elif self.path in ("/api/gate1/open", "/api/gate1/close", "/api/gate1/pulse"):
                cmd = self.path.rsplit("/", 1)[-1]
                with api.lock:
                    api.gate1_command = cmd
                self._send(200, b'{"ok":true}')
            elif self.path == "/api/gate1/sensor_auto":
                with api.lock:
                    api.sensor1_auto = bool(payload.get("enabled", False))
                self._send(200, b'{"ok":true}')
            elif self.path in ("/api/gate2/open", "/api/gate2/close", "/api/gate2/pulse"):
                cmd = self.path.rsplit("/", 1)[-1]
                with api.lock:
                    api.gate2_command = cmd
                self._send(200, b'{"ok":true}')
            elif self.path == "/api/gate2/sensor_auto":
                with api.lock:
                    api.sensor2_auto = bool(payload.get("enabled", False))
                self._send(200, b'{"ok":true}')
            elif self.path in ("/api/gate3/open", "/api/gate3/close", "/api/gate3/pulse"):
                cmd = self.path.rsplit("/", 1)[-1]
                with api.lock:
                    api.gate3_command = cmd
                self._send(200, b'{"ok":true}')
            elif self.path == "/api/gate3/sensor_auto":
                with api.lock:
                    api.sensor3_auto = bool(payload.get("enabled", False))
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

    cv_top_camera = sup.getDevice("cv_top")
    cv_top_countdown = -1  # -1 = не включена; см. supervisor_main.py /api/screenshot

    motor1 = sup.getDevice("paddle_shiber1")
    sensor1 = sup.getDevice("paddle_shiber1_sensor")
    sensor1.enable(timestep)
    motor1.setPosition(CLOSED1_ANGLE)  # старт — ЗАКРЫТО (барьер, вход перекрыт)

    beam1 = sup.getDevice("beam_sensor_1")
    beam1.enable(timestep)
    awaiting_auto_close1 = False   # "pulse": закрыть после выдержки на открытом
    auto_close1_at = None          # sup.getTime(), когда пора закрыть
    beam1_was_blocked = False      # для срабатывания по фронту

    motor2 = sup.getDevice("paddle_shiber2")
    sensor2 = sup.getDevice("paddle_shiber2_sensor")
    sensor2.enable(timestep)
    motor2.setPosition(CLOSED2_ANGLE)  # старт — ЗАКРЫТО (барьер, вход перекрыт)

    beam2 = sup.getDevice("beam_sensor_2")
    beam2.enable(timestep)
    awaiting_auto_close2 = False   # "pulse": закрыть после выдержки на открытом
    auto_close2_at = None          # sup.getTime(), когда пора закрыть
    beam2_was_blocked = False      # для срабатывания по фронту

    motor3 = sup.getDevice("paddle_shiber3")
    sensor3 = sup.getDevice("paddle_shiber3_sensor")
    sensor3.enable(timestep)
    motor3.setPosition(CLOSED3_ANGLE)  # старт — ЗАКРЫТО (барьер, вход перекрыт)

    beam3 = sup.getDevice("beam_sensor_3")
    beam3.enable(timestep)
    awaiting_auto_close3 = False   # "pulse": закрыть после выдержки на открытом
    auto_close3_at = None          # sup.getTime(), когда пора закрыть
    beam3_was_blocked = False      # для срабатывания по фронту

    root_children = sup.getRoot().getField("children")
    tracked: dict[int, dict] = {}
    uid_seq = 0

    while sup.step(timestep) != -1:
        if api.cv_top_requested.is_set():
            if cv_top_countdown < 0:
                cv_top_camera.enable(timestep)
                cv_top_countdown = 3   # дать камере отрисовать кадр
            elif cv_top_countdown > 0:
                cv_top_countdown -= 1
            else:
                cv_top_camera.saveImage(str(CV_TOP_SCREENSHOT_PATH), 90)
                cv_top_camera.disable()
                cv_top_countdown = -1
                api.cv_top_requested.clear()
                api.cv_top_done.set()

        with api.lock:
            spawn_requests = api.spawn_requests
            api.spawn_requests = []
            clear_requested = api.clear_requested
            api.clear_requested = False
            gate1_command = api.gate1_command
            api.gate1_command = None
            sensor1_auto = api.sensor1_auto
            gate2_command = api.gate2_command
            api.gate2_command = None
            sensor2_auto = api.sensor2_auto
            gate3_command = api.gate3_command
            api.gate3_command = None
            sensor3_auto = api.sensor3_auto

        # Если включён РОВНО ОДИН чекбокс автосрабатывания — этому щиту не с
        # кем делить ленту, реагирует на любой товар без разбора категории
        # (удобно для одиночного теста щита на чём угодно). Если включено
        # два+ — разбор по категории обязателен (см. фидбек 2026-07-30 про
        # "оба щита реагируют на всё подряд").
        _auto_active_count = sum([sensor1_auto, sensor2_auto, sensor3_auto])

        beam1_distance = beam1.getValue()
        beam1_blocked = beam1_distance < BEAM_TRIGGER_DIST
        if sensor1_auto and beam1_blocked and not beam1_was_blocked:
            if _auto_active_count == 1 or _nearest_category(tracked, SENSOR1_X) == SHIBER1_CATEGORY:
                gate1_command = "pulse"   # луч только что перекрыт -> тот же цикл, что и кнопка
        beam1_was_blocked = beam1_blocked

        if gate1_command == "open":
            motor1.setPosition(OPEN1_ANGLE)
            awaiting_auto_close1 = False
        elif gate1_command == "close":
            motor1.setPosition(CLOSED1_ANGLE)
            awaiting_auto_close1 = False
        elif gate1_command == "pulse":
            motor1.setPosition(OPEN1_ANGLE)
            awaiting_auto_close1 = True
            auto_close1_at = None

        if awaiting_auto_close1:
            if auto_close1_at is None:
                if abs(sensor1.getValue() - OPEN1_ANGLE) <= ANGLE_TOL:
                    auto_close1_at = sup.getTime() + AUTO_CLOSE_HOLD_S
            elif sup.getTime() >= auto_close1_at:
                motor1.setPosition(CLOSED1_ANGLE)
                awaiting_auto_close1 = False
                auto_close1_at = None

        beam2_distance = beam2.getValue()
        beam2_blocked = beam2_distance < BEAM_TRIGGER_DIST
        if sensor2_auto and beam2_blocked and not beam2_was_blocked:
            if _auto_active_count == 1 or _nearest_category(tracked, SENSOR2_X) == SHIBER2_CATEGORY:
                gate2_command = "pulse"
        beam2_was_blocked = beam2_blocked

        if gate2_command == "open":
            motor2.setPosition(OPEN2_ANGLE)
            awaiting_auto_close2 = False
        elif gate2_command == "close":
            motor2.setPosition(CLOSED2_ANGLE)
            awaiting_auto_close2 = False
        elif gate2_command == "pulse":
            motor2.setPosition(OPEN2_ANGLE)
            awaiting_auto_close2 = True
            auto_close2_at = None

        if awaiting_auto_close2:
            if auto_close2_at is None:
                if abs(sensor2.getValue() - OPEN2_ANGLE) <= ANGLE_TOL:
                    auto_close2_at = sup.getTime() + AUTO_CLOSE_HOLD_S
            elif sup.getTime() >= auto_close2_at:
                motor2.setPosition(CLOSED2_ANGLE)
                awaiting_auto_close2 = False
                auto_close2_at = None

        beam3_distance = beam3.getValue()
        beam3_blocked = beam3_distance < BEAM_TRIGGER_DIST
        if sensor3_auto and beam3_blocked and not beam3_was_blocked:
            if _auto_active_count == 1 or _nearest_category(tracked, SENSOR3_X) == SHIBER3_CATEGORY:
                gate3_command = "pulse"
        beam3_was_blocked = beam3_blocked

        if gate3_command == "open":
            motor3.setPosition(OPEN3_ANGLE)
            awaiting_auto_close3 = False
        elif gate3_command == "close":
            motor3.setPosition(CLOSED3_ANGLE)
            awaiting_auto_close3 = False
        elif gate3_command == "pulse":
            motor3.setPosition(OPEN3_ANGLE)
            awaiting_auto_close3 = True
            auto_close3_at = None

        if awaiting_auto_close3:
            if auto_close3_at is None:
                if abs(sensor3.getValue() - OPEN3_ANGLE) <= ANGLE_TOL:
                    auto_close3_at = sup.getTime() + AUTO_CLOSE_HOLD_S
            elif sup.getTime() >= auto_close3_at:
                motor3.setPosition(CLOSED3_ANGLE)
                awaiting_auto_close3 = False
                auto_close3_at = None

        if clear_requested:
            for info in tracked.values():
                if info["node"] is not None:
                    info["node"].remove()
            tracked.clear()

        for payload in spawn_requests:
            kind = payload.get("type", "cube_medium")
            y = max(-0.24, min(0.24, float(payload.get("y", 0.0))))
            yaw = float(payload.get("yaw", 0.0))
            uid_seq += 1
            name = f"spawn_{uid_seq}"
            if kind.startswith(("cube_", "ball_")):
                node_str = simple_shape_node(name, kind, SPAWN_X, y)
            elif kind in REAL_OBJECTS:
                node_str = real_object_node(name, kind, SPAWN_X, y, yaw)
            else:
                kind = "cube_medium"
                node_str = simple_shape_node(name, kind, SPAWN_X, y)
            root_children.importMFNodeFromString(-1, node_str)
            tracked[uid_seq] = {"node": sup.getFromDef(name.upper()), "kind": kind,
                                "category": category_of(kind)}

        items = []
        for uid, info in tracked.items():
            node = info["node"]
            if node is None:
                continue
            x, y, _z = node.getPosition()
            items.append({
                "uid": uid, "kind": info["kind"], "category": info["category"],
                "x": round(x, 3), "y": round(y, 3),
                "segment": segment_of(x),
            })

        angle1 = sensor1.getValue()
        if abs(angle1 - CLOSED1_ANGLE) <= ANGLE_TOL:
            gate1_state = "ЗАКРЫТО"
        elif abs(angle1 - OPEN1_ANGLE) <= ANGLE_TOL:
            gate1_state = "ОТКРЫТО"
        else:
            gate1_state = "движется"

        angle2 = sensor2.getValue()
        if abs(angle2 - CLOSED2_ANGLE) <= ANGLE_TOL:
            gate2_state = "ЗАКРЫТО"
        elif abs(angle2 - OPEN2_ANGLE) <= ANGLE_TOL:
            gate2_state = "ОТКРЫТО"
        else:
            gate2_state = "движется"

        angle3 = sensor3.getValue()
        if abs(angle3 - CLOSED3_ANGLE) <= ANGLE_TOL:
            gate3_state = "ЗАКРЫТО"
        elif abs(angle3 - OPEN3_ANGLE) <= ANGLE_TOL:
            gate3_state = "ОТКРЫТО"
        else:
            gate3_state = "движется"

        with api.lock:
            api.status = {"items": items, "gate1_state": gate1_state, "gate1_angle": angle1,
                          "beam1_blocked": beam1_blocked, "beam1_distance": round(beam1_distance, 3),
                          "gate2_state": gate2_state, "gate2_angle": angle2,
                          "beam2_blocked": beam2_blocked, "beam2_distance": round(beam2_distance, 3),
                          "gate3_state": gate3_state, "gate3_angle": angle3,
                          "beam3_blocked": beam3_blocked, "beam3_distance": round(beam3_distance, 3),
                          "headless": os.environ.get("ROBOZON_HEADLESS") != "0"}


if __name__ == "__main__":
    main()
