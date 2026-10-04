param([string]$Python = '', [ValidateSet('auto','cuda','cpu')][string]$Device = 'auto')
$ErrorActionPreference = 'Stop'
$pipelineRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$runtimeRoot = Join-Path $pipelineRoot '.runtime'
New-Item -ItemType Directory -Force -Path $runtimeRoot | Out-Null
if ($Python) {
    $pythonCommand = (Resolve-Path -LiteralPath $Python).Path
    $pythonArguments = @()
} else {
    $pythonCommand = 'py.exe'
    $pythonArguments = @('-3.11')
}
& $pythonCommand @pythonArguments -c 'import sys; assert sys.version_info[:2] == (3,11), "Python 3.11 required"'
if ($LASTEXITCODE) { throw 'Install Python 3.11 or provide -Python C:/path/to/python.exe' }
foreach ($name in @('backend','paddle')) {
    $environment = Join-Path $runtimeRoot $name
    & $pythonCommand @pythonArguments -m venv $environment
    if ($LASTEXITCODE) { throw "Could not create $name environment" }
}
$backendPython = Join-Path $runtimeRoot 'backend/Scripts/python.exe'
$paddlePython = Join-Path $runtimeRoot 'paddle/Scripts/python.exe'
if ($Device -eq 'auto') {
    $Device = 'cpu'
    if (Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue) {
        $gpuNames = & nvidia-smi.exe --query-gpu=name --format=csv,noheader
        if ($LASTEXITCODE -eq 0 -and $gpuNames) { $Device = 'cuda' }
    }
}
$env:PYTHONUTF8 = '1'
& $backendPython -m pip install --upgrade pip
if ($LASTEXITCODE) { throw 'Backend pip upgrade failed' }
& $paddlePython -m pip install --upgrade pip
if ($LASTEXITCODE) { throw 'Paddle pip upgrade failed' }
$torchIndex = if ($Device -eq 'cuda') { 'https://download.pytorch.org/whl/cu118' } else { 'https://download.pytorch.org/whl/cpu' }
& $backendPython -m pip install --no-cache-dir --timeout 180 torch==2.6.0 torchvision==0.21.0 --index-url $torchIndex
if ($LASTEXITCODE) { throw 'PyTorch installation failed' }
& $backendPython -m pip install --no-cache-dir -r (Join-Path $pipelineRoot 'backend/requirements.txt')
if ($LASTEXITCODE) { throw 'Backend dependency installation failed' }
if ($Device -eq 'cuda') {
    & $paddlePython -m pip install --timeout 180 --retries 5 paddlepaddle-gpu==3.2.0 --index-url https://www.paddlepaddle.org.cn/packages/stable/cu118/
} else {
    & $paddlePython -m pip install paddlepaddle==3.2.0
}
if ($LASTEXITCODE) { throw 'Paddle installation failed' }
& $paddlePython -m pip install --no-cache-dir numpy==1.26.4 opencv-python==4.10.0.84 PyYAML==6.0.2
if ($LASTEXITCODE) { throw 'OCR dependency installation failed' }
Push-Location (Join-Path $pipelineRoot 'frontend')
try {
    npm.cmd install
    if ($LASTEXITCODE) { throw 'Frontend install failed' }
    npm.cmd run build
    if ($LASTEXITCODE) { throw 'Frontend build failed' }
} finally { Pop-Location }
if (!(Test-Path -LiteralPath (Join-Path $pipelineRoot '.env'))) {
    Copy-Item -LiteralPath (Join-Path $pipelineRoot '.env.example') -Destination (Join-Path $pipelineRoot '.env')
}
& (Join-Path $PSScriptRoot 'diagnostics.ps1')
if ($LASTEXITCODE) { throw 'Model verification failed; see diagnostics output' }
Write-Host 'IRIS Pipeline setup complete. Run scripts/start.ps1.' -ForegroundColor Green
