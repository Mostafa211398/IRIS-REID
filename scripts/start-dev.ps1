param()
$ErrorActionPreference = 'Stop'
$pipelineRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$backendPython = Join-Path $pipelineRoot '.runtime/backend/Scripts/python.exe'
$env:PYTHONUTF8 = '1'
$env:PYTHONPATH = Join-Path $pipelineRoot 'backend'
$env:YOLO_CONFIG_DIR = Join-Path $pipelineRoot '.runtime/ultralytics'
& $backendPython (Join-Path $PSScriptRoot 'launch-check.py')
if ($LASTEXITCODE) { throw 'Pipeline startup check failed; no new server was launched.' }
$process = Start-Process -FilePath $backendPython -ArgumentList @('-m','iris_pipeline') -WorkingDirectory $pipelineRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $pipelineRoot '.runtime/backend.out.log') -RedirectStandardError (Join-Path $pipelineRoot '.runtime/backend.err.log')
Push-Location (Join-Path $pipelineRoot 'frontend')
try { npm.cmd run dev } finally {
    Pop-Location
    if (!$process.HasExited) { & (Join-Path $PSScriptRoot 'stop.ps1') -LauncherProcessId $process.Id }
}
