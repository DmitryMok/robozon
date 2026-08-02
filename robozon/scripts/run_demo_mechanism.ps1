# Запуск автономной демо-сцены demo_mechanism (исполнительный механизм —
# отдельная сцена для финальной сдачи, см. вики "План финализации сдачи
# (раздельные демо CV и исполнительного механизма)") — отдельно от основной
# линии, БЕЗ gen_world.py и без основной HTTP-панели.
#   scripts\run_demo_mechanism.ps1              — обычный запуск, окно Webots
#   scripts\run_demo_mechanism.ps1 -Headless    — без окна/рендера, для
#                                                  быстрой проверки по логу
#
# Веб-панель (спавн) — http://localhost:8022/ после запуска Webots.
# Независимый CV-отчёт — http://localhost:8010/ (поднимается этим скриптом).
#
# Правка геометрии — tools/gen_demo_mechanism.py, затем:
#   C:\Projects\.venvs\venv-cv-pytorch\Scripts\python.exe tools\gen_demo_mechanism.py
param(
    [switch]$Headless
)
$ErrorActionPreference = "Stop"

$Root = (Resolve-Path "$PSScriptRoot\..").Path
$WebotsHome = "C:\Program Files\Webots"
$VenvPython = "C:\Projects\.venvs\venv-cv-pytorch\Scripts\python.exe"
$CvReportPython = "C:\Projects\.venvs\venv-roboson-tools\Scripts\python.exe"
$CvReportPort = 8010

$env:WEBOTS_HOME = $WebotsHome
$env:PYTHONPATH = "$WebotsHome\lib\controller\python;$Root"

# Webots запускает .py-контроллеры через "python" из PATH.
$env:Path = "C:\Projects\.venvs\venv-cv-pytorch\Scripts;" + $env:Path

Write-Output "== Генерация demo_mechanism.wbt из tools/gen_demo_mechanism.py =="
& $VenvPython "$Root\tools\gen_demo_mechanism.py"

if (-not (Test-Path $CvReportPython)) {
    throw "Не найден Python для CV-отчёта: $CvReportPython"
}

Write-Output "== Запуск независимого CV-отчёта =="
$CvReportProcess = Start-Process -FilePath $CvReportPython `
    -ArgumentList @("-B", "$Root\tools\cv_report_server.py", "--host", "127.0.0.1", "--port", "$CvReportPort") `
    -WorkingDirectory $Root -PassThru

Write-Output "== Запуск Webots =="
Write-Output "== Панель: http://localhost:8022/ =="
Write-Output "== CV-отчёт: http://localhost:$CvReportPort/ =="

# --stream НЕ добавляем (как и в run.ps1, см. тот же комментарий там) — на Windows
# подозревается в зависании диалога "Opening world file"/загрузки ассетов. На Windows
# у Webots и так нативное окно, стрим не нужен — встроенный в панель 8022 <webots-view>
# реально нужен только для WSL/headless-запуска (scripts/run_demo_mechanism.sh), где
# нативного окна нет вообще. Если открыть панель 8022 на Windows-запуске — блок
# "3D-стрим" в ней просто не подключится (не критично, есть нативное окно Webots).
$WebotsArgs = @("--batch", "--mode=realtime", "--stdout", "--stderr")
if ($Headless) {
    $WebotsArgs = @("--batch", "--minimize", "--no-rendering", "--mode=fast", "--stdout", "--stderr")
}

try {
    & "$WebotsHome\msys64\mingw64\bin\webots.exe" $WebotsArgs "$Root\webots\worlds\demo_mechanism.wbt"
}
finally {
    # CV-отчёт не остаётся фоновым процессом после завершения демо.
    if ($CvReportProcess -and -not $CvReportProcess.HasExited) {
        Stop-Process -Id $CvReportProcess.Id -Force
    }
}
