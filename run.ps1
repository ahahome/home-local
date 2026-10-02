param([int]$Port = 8000)
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw 'Please create .venv and install requirements.txt first. See README.md.'
}
if (-not (Test-Path -LiteralPath (Join-Path $projectRoot '.env'))) {
    throw 'Copy .env.example to .env and configure HOME_LLM_API_KEY first.'
}
$taskLogDir = Join-Path $projectRoot 'data/runtime'
New-Item -ItemType Directory -Path $taskLogDir -Force | Out-Null
$appUrl = "http://127.0.0.1:$Port"
try {
    $existing = Invoke-RestMethod "$appUrl/api/status" -TimeoutSec 2
    if ($null -ne $existing.models -and $null -ne $existing.configured) {
        Write-Output "home is already running: $appUrl"
        return
    }
} catch { }
$runningApp = Start-Process -FilePath $pythonPath -ArgumentList '-m','uvicorn','app.api.main:app','--host','127.0.0.1','--port',"$Port" -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $taskLogDir 'server.out.log') -RedirectStandardError (Join-Path $taskLogDir 'server.err.log') -PassThru
$runningApp.Id | Set-Content (Join-Path $taskLogDir 'server.pid')
for ($attempt = 0; $attempt -lt 30; $attempt++) {
    Start-Sleep -Milliseconds 500
    try {
        $ready = Invoke-RestMethod "$appUrl/api/status" -TimeoutSec 2
        if ($null -ne $ready.models) {
            Write-Output "home is ready: $appUrl"
            return
        }
    } catch { }
    if ($runningApp.HasExited) { break }
}
throw "Startup failed. See $taskLogDir/server.err.log"
