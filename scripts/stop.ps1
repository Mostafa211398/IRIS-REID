param([int]$LauncherProcessId = 0)
$ErrorActionPreference = 'Stop'
$pipelineRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$pidFile = Join-Path $pipelineRoot '.runtime/server.pid'
if ($LauncherProcessId) {
    $launcherId = $LauncherProcessId
} else {
    if (!(Test-Path -LiteralPath $pidFile)) { Write-Host 'No background launch record.'; return }
    $launcherId = [int](Get-Content -LiteralPath $pidFile)
}
$processes = @(Get-CimInstance Win32_Process)
$launcher = $processes | Where-Object ProcessId -eq $launcherId
if (!$launcher) { Write-Host 'Background server already stopped.'; return }
if (!$launcher.CommandLine -or !$launcher.CommandLine.Contains($pipelineRoot) -or !$launcher.CommandLine.Contains('iris_pipeline')) {
    throw 'Launch record does not belong to this IRIS Pipeline process. No process was stopped.'
}
$ownedIds = [System.Collections.Generic.List[int]]::new()
$ownedIds.Add($launcherId)
do {
    $added = $false
    foreach ($process in $processes) {
        if ($ownedIds.Contains([int]$process.ParentProcessId) -and !$ownedIds.Contains([int]$process.ProcessId)) {
            $ownedIds.Add([int]$process.ProcessId)
            $added = $true
        }
    }
} while ($added)
for ($i=$ownedIds.Count-1; $i -ge 0; $i--) { Stop-Process -Id $ownedIds[$i] -ErrorAction SilentlyContinue }
Write-Host 'IRIS Pipeline background server stopped. Interrupted jobs can be retried after restart.'
