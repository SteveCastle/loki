# Reads media-server/deps/models/manifest.json, downloads each file to a temp
# dir, computes SHA-256, and prints "<id> <rel> <sha256>" lines to stdout.
# After verifying, edit manifest.json by hand to replace the UNVERIFIED
# placeholders with the printed hashes.
#
# Usage:
#   powershell -File media-server\scripts\hash-manifest.ps1
#   powershell -File media-server\scripts\hash-manifest.ps1 path\to\manifest.json
param(
  [string]$ManifestPath = "media-server\deps\models\manifest.json"
)
$ErrorActionPreference = "Stop"

# Invoke-WebRequest's Write-Progress overhead can slow large model
# downloads 10-100x on PowerShell 5.1. Silence it.
$ProgressPreference = 'SilentlyContinue'

if (-not (Test-Path $ManifestPath)) {
  Write-Error "manifest not found: $ManifestPath"
}

$tmp = New-Item -ItemType Directory -Path ([System.IO.Path]::GetTempPath() + [System.Guid]::NewGuid().ToString())
try {
  $data = Get-Content $ManifestPath -Raw | ConvertFrom-Json
  foreach ($model in $data.models) {
    foreach ($file in $model.files) {
      Write-Host "fetching $($model.id)/$($file.rel_path) ..." -ForegroundColor DarkGray
      # Hash the URL to get a stable temp filename so re-runs don't collide.
      $urlHash = [System.BitConverter]::ToString(
        [System.Security.Cryptography.SHA1]::Create().ComputeHash(
          [System.Text.Encoding]::UTF8.GetBytes($file.url))
      ).Replace("-", "").Substring(0, 12).ToLower()
      $out = Join-Path $tmp $urlHash
      Invoke-WebRequest -Uri $file.url -OutFile $out -UseBasicParsing
      $sum = (Get-FileHash -Algorithm SHA256 $out).Hash.ToLower()
      Write-Output "$($model.id) $($file.rel_path) $sum"
    }
  }
} finally {
  Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
}
