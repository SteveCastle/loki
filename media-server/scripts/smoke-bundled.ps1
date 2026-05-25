# Boots a freshly-built server in the release layout, polls /api/deps/status,
# asserts every bundled dep is "ready" or "broken" (NOT "missing"), then
# stops the server. Used as a post-build CI check on Windows runners.
#
# Usage:
#   powershell -File media-server\scripts\smoke-bundled.ps1
#   powershell -File media-server\scripts\smoke-bundled.ps1 path\to\install_dir
#
# install_dir must contain lowkeymediaserver.exe and a bin\ subdirectory.
param(
  [string]$InstallDir = ""
)
$ErrorActionPreference = "Stop"

if ([string]::IsNullOrEmpty($InstallDir)) {
  $InstallDir = (Resolve-Path "$PSScriptRoot\..").Path
}
if (-not (Test-Path (Join-Path $InstallDir "lowkeymediaserver.exe"))) {
  Write-Error "lowkeymediaserver.exe not found in $InstallDir"
}

$port = 18762
$statusFile = Join-Path $env:TEMP "lms-smoke-status.json"
if (Test-Path $statusFile) { Remove-Item $statusFile -Force }

$env:LOWKEY_PORT = "$port"
$proc = Start-Process -FilePath (Join-Path $InstallDir "lowkeymediaserver.exe") `
  -WorkingDirectory $InstallDir `
  -PassThru -WindowStyle Hidden

try {
  # Poll up to ~10s for the listener.
  $ready = $false
  for ($i = 0; $i -lt 20; $i++) {
    try {
      $resp = Invoke-WebRequest -Uri "http://127.0.0.1:$port/api/deps/status" `
        -UseBasicParsing -TimeoutSec 1 -ErrorAction Stop
      $resp.Content | Out-File -FilePath $statusFile -Encoding utf8
      $ready = $true
      break
    } catch {
      Start-Sleep -Milliseconds 500
    }
  }
  if (-not $ready -or -not (Test-Path $statusFile) -or (Get-Item $statusFile).Length -eq 0) {
    Write-Error "server did not serve /api/deps/status within 10s"
  }

  $items = Get-Content $statusFile -Raw | ConvertFrom-Json
  $missing = @($items | Where-Object { $_.category -eq "bundled" -and $_.state -eq "missing" })

  if ($missing.Count -ne 0) {
    Write-Host "smoke failed: $($missing.Count) bundled deps missing" -ForegroundColor Red
    $missing | ConvertTo-Json -Depth 5 | Write-Host
    exit 1
  }
  Write-Host "smoke OK" -ForegroundColor Green
}
finally {
  if ($proc -and -not $proc.HasExited) {
    Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
  }
  if (Test-Path $statusFile) { Remove-Item $statusFile -Force -ErrorAction SilentlyContinue }
  Remove-Item Env:\LOWKEY_PORT -ErrorAction SilentlyContinue
}
