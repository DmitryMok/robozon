# Run the sorting-line simulation natively on Windows (perf comparison vs WSL).
# Mirrors scripts/run.sh: same world generation, same supervisor controller,
# just native Windows Webots + venv (real GPU render instead of WSLg,
# ODE physics without the hypervisor layer).
#   scripts\run.ps1              — обычный запуск, окно Webots
#   scripts\run.ps1 -Headless    — без окна/рендера (--minimize --no-rendering),
#                                  для автоматических тестов через API (нет
#                                  xvfb-эквивалента на Windows, --no-rendering
#                                  это ближайший аналог run.sh --headless)
param(
    [switch]$Headless
)
$ErrorActionPreference = "Stop"

$Root = (Resolve-Path "$PSScriptRoot\..").Path
$WebotsHome = "C:\Program Files\Webots"
$VenvPython = "C:\Projects\.venvs\venv-cv-pytorch\Scripts\python.exe"

$env:WEBOTS_HOME = $WebotsHome
$env:PYTHONPATH = "$WebotsHome\lib\controller\python;$Root"
$env:ROBOZON_HEADLESS = if ($Headless) { "1" } else { "0" }

# Webots launches .py controllers via "python" on PATH -> put the venv first
# so controller/numpy/yaml/PIL resolve from venv-cv-pytorch.
$env:Path = "C:\Projects\.venvs\venv-cv-pytorch\Scripts;" + $env:Path

$PanelPort  = & $VenvPython -c "import sys; sys.path.insert(0, r'$Root'); from sim.config import load_layout; print(load_layout()['http']['port'])"

Write-Output "== Generating world from config/layout.yaml =="
& $VenvPython "$Root\tools\gen_world.py"

Write-Output "== Starting Webots =="
Write-Output "== Control panel: http://localhost:$PanelPort/ =="

# --stream снят: подозревается в зависании диалога "Opening world file" на
# Windows (сама симуляция под диалогом работала нормально - см. переписку).
# Панели/API это не касалось, стрим 3D-вида в браузер тут не нужен, у Webots
# и так нативное окно.
$WebotsArgs = @("--batch", "--mode=realtime", "--stdout", "--stderr")
if ($Headless) {
    # --no-rendering отключает рендер камер/3D-вида целиком (не только окно) —
    # быстрее realtime, для тестов через HTTP API (позиции/статус) рендер не
    # нужен вовсе. --minimize на случай, если --no-rendering всё же откроет окно.
    $WebotsArgs = @("--batch", "--minimize", "--no-rendering", "--mode=fast", "--stdout", "--stderr")
}

& "$WebotsHome\msys64\mingw64\bin\webots.exe" $WebotsArgs "$Root\webots\worlds\sorting_line.wbt"
