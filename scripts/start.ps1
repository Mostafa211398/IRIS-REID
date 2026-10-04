param([switch]$Background)
$ErrorActionPreference = 'Stop'
$pipelineRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$backendPython = Join-Path $pipelineRoot '.runtime/backend/Scripts/python.exe'
if (!(Test-Path -LiteralPath $backendPython)) { throw 'Run scripts/setup.ps1 first' }
if (!(Test-Path -LiteralPath (Join-Path $pipelineRoot 'frontend/dist/index.html'))) { throw 'Build the frontend with npm.cmd run build in frontend/' }
$env:PYTHONUTF8 = '1'
$env:PYTHONPATH = Join-Path $pipelineRoot 'backend'
$env:YOLO_CONFIG_DIR = Join-Path $pipelineRoot '.runtime/ultralytics'
Push-Location $pipelineRoot
try {
    & $backendPython (Join-Path $PSScriptRoot 'launch-check.py')
    if ($LASTEXITCODE) { throw 'Pipeline startup check failed; no new server was launched.' }
    if ($Background) {
        $process = Start-Process -FilePath $backendPython -ArgumentList @('-m','iris_pipeline') -WorkingDirectory $pipelineRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $pipelineRoot '.runtime/server.out.log') -RedirectStandardError (Join-Path $pipelineRoot '.runtime/server.err.log')
        $process.Id | Set-Content -LiteralPath (Join-Path $pipelineRoot '.runtime/server.pid')
        Write-Host "Started in background. Run scripts/stop.ps1 to stop it. PID: $($process.Id)"
    } else {
        & $backendPython -m iris_pipeline
    }
} finally { Pop-Location }
