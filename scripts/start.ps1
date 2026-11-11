<#
.SYNOPSIS
  Start the TerraVision AI backend (FastAPI, port 8000) and frontend (Vite, port 5173) on Windows.
.EXAMPLE
  .\scripts\start.ps1            # start both, open the browser
  .\scripts\start.ps1 -NoBrowser
  Stop with Ctrl+C: both processes are shut down.
#>
param([switch]$NoBrowser)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) { throw "Virtual environment not found. See docs/Implementation_Guide.md (Setup)." }
if (-not (Test-Path (Join-Path $root 'frontend\node_modules'))) { throw "Run 'npm install' inside frontend/ first." }
foreach ($ckpt in 'swin_siamese_levir.pt', 'fc_siam_diff_levir.pt') {
    if (-not (Test-Path (Join-Path $root "checkpoints\$ckpt"))) { Write-Warning "checkpoints\$ckpt is missing - that model will be unavailable." }
}

$backend = Start-Process -FilePath $python -WorkingDirectory $root -PassThru -NoNewWindow `
    -ArgumentList '-m', 'uvicorn', 'backend.app.main:app', '--host', '127.0.0.1', '--port', '8000'
$frontend = Start-Process -FilePath 'npm.cmd' -WorkingDirectory (Join-Path $root 'frontend') -PassThru -NoNewWindow `
    -ArgumentList 'run', 'dev'

try {
    $ready = $false
    foreach ($i in 1..60) {
        try { if ((Invoke-RestMethod 'http://127.0.0.1:8000/health' -TimeoutSec 2).status) { $ready = $true; break } } catch { Start-Sleep -Milliseconds 500 }
    }
    if (-not $ready) { throw 'Backend did not become healthy within 30 s.' }
    Write-Host "`nBackend : http://127.0.0.1:8000  (API docs: /docs)" -ForegroundColor Green
    Write-Host "Frontend: http://localhost:5173`n" -ForegroundColor Green
    if (-not $NoBrowser) { Start-Process 'http://localhost:5173' }
    Wait-Process -Id $backend.Id, $frontend.Id
}
finally {
    foreach ($proc in $backend, $frontend) {
        if ($proc -and -not $proc.HasExited) { taskkill /PID $proc.Id /T /F | Out-Null }
    }
    Write-Host 'TerraVision AI stopped.'
}
