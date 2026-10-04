param()
$ErrorActionPreference = 'Stop'
$pipelineRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$env:PYTHONUTF8 = '1'
$env:PYTHONPATH = Join-Path $pipelineRoot 'backend'
Push-Location $pipelineRoot
try {
    & (Join-Path $pipelineRoot '.runtime/backend/Scripts/python.exe') -m pytest backend/tests -q
    if ($LASTEXITCODE) { throw 'Backend verification failed' }
    Push-Location frontend
    try { npm.cmd run build; if ($LASTEXITCODE) { throw 'Frontend build failed' } } finally { Pop-Location }
} finally { Pop-Location }
