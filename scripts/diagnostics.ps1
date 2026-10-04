param()
$ErrorActionPreference = 'Stop'
$pipelineRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$env:PYTHONUTF8 = '1'
$env:PYTHONPATH = Join-Path $pipelineRoot 'backend'
$env:YOLO_CONFIG_DIR = Join-Path $pipelineRoot '.runtime/ultralytics'
& (Join-Path $pipelineRoot '.runtime/backend/Scripts/python.exe') (Join-Path $PSScriptRoot 'diagnostics.py')
if ($LASTEXITCODE) { throw 'At least one inference stage failed verification' }
