# Запуск автономной демо-сцены demo_roller_shiber (модифицированный шибер
# с вращающимся валом) — отдельно от основной линии, БЕЗ gen_world.py и
# без HTTP-панели (сцена не читает config/layout.yaml).
#   scripts\run_demo_roller_shiber.ps1              — обычный запуск, окно Webots
#   scripts\run_demo_roller_shiber.ps1 -Headless    — без окна/рендера, для
#                                                      быстрой проверки по логу
#                                                      консоли (PASS/FAIL по
#                                                      каждому тестовому объекту)
#
# Правка геометрии/скорости вала — tools/gen_demo_roller_shiber.py, затем:
#   C:\Projects\.venvs\venv-cv-pytorch\Scripts\python.exe tools\gen_demo_roller_shiber.py
param(
    [switch]$Headless
)
$ErrorActionPreference = "Stop"

$Root = (Resolve-Path "$PSScriptRoot\..").Path
$WebotsHome = "C:\Program Files\Webots"
$VenvPython = "C:\Projects\.venvs\venv-cv-pytorch\Scripts\python.exe"

$env:WEBOTS_HOME = $WebotsHome
$env:PYTHONPATH = "$WebotsHome\lib\controller\python;$Root"

# Webots запускает .py-контроллеры через "python" из PATH.
$env:Path = "C:\Projects\.venvs\venv-cv-pytorch\Scripts;" + $env:Path

Write-Output "== Генерация demo_roller_shiber.wbt из tools/gen_demo_roller_shiber.py =="
& $VenvPython "$Root\tools\gen_demo_roller_shiber.py"

Write-Output "== Запуск Webots =="
Write-Output "== Исход по каждому объекту (PASS/FAIL) печатается в эту консоль (controller demo_outcome_logger) =="

$WebotsArgs = @("--batch", "--mode=realtime", "--stdout", "--stderr")
if ($Headless) {
    $WebotsArgs = @("--batch", "--minimize", "--no-rendering", "--mode=fast", "--stdout", "--stderr")
}

& "$WebotsHome\msys64\mingw64\bin\webots.exe" $WebotsArgs "$Root\webots\worlds\demo_roller_shiber.wbt"
