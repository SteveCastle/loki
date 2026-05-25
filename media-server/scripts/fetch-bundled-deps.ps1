# Windows equivalent of fetch-bundled-deps.sh. PowerShell 5.1 compatible.
param(
  [Parameter(Mandatory=$true)][string]$Target,
  [ValidateSet("verify","update")][string]$Mode = "verify"
)
$ErrorActionPreference = "Stop"

$root  = (Resolve-Path "$PSScriptRoot\..\..").Path
$conf  = Join-Path $root "media-server\scripts\bundled-versions.json"
$outDir = Join-Path $root "media-server\bin\$Target"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$data = Get-Content $conf -Raw | ConvertFrom-Json
$tmp  = New-Item -ItemType Directory -Path ([System.IO.Path]::GetTempPath() + [System.Guid]::NewGuid().ToString())

try {
  foreach ($bin in $data.binaries.PSObject.Properties.Name) {
    $entry = $data.binaries.$bin.$Target
    if ($null -eq $entry) { continue }

    $archivePath = Join-Path $tmp "$bin.archive"
    Write-Host "fetching $bin ($($entry.url)) ..."
    Invoke-WebRequest -Uri $entry.url -OutFile $archivePath
    $gotSum = (Get-FileHash -Algorithm SHA256 $archivePath).Hash.ToLower()

    if ($Mode -eq "update") {
      Write-Host "SHA256 $bin $Target $gotSum"
    } else {
      if ($entry.sha256 -ne "TO_FILL" -and $entry.sha256 -ne $gotSum) {
        Write-Error "SHA256 mismatch for $bin $Target`n  want: $($entry.sha256)`n  got:  $gotSum"
      }
    }

    $extractDir = Join-Path $tmp $bin
    New-Item -ItemType Directory -Force -Path $extractDir | Out-Null
    switch ($entry.archive) {
      "zip"    { Expand-Archive -Path $archivePath -DestinationPath $extractDir -Force }
      "tar.gz" { tar -xzf $archivePath -C $extractDir }
      "tar.xz" { tar -xJf $archivePath -C $extractDir }
      "none"   { Copy-Item $archivePath (Join-Path $extractDir ([IO.Path]::GetFileName($entry.url))) }
      default  { Write-Error "unknown archive type $($entry.archive)" }
    }

    foreach ($ex in $entry.extract) {
      $type = if ($ex.type) { $ex.type } else { "file" }
      $matches = Get-ChildItem -Path $extractDir -Recurse -Filter ([IO.Path]::GetFileName($ex.from)) -ErrorAction SilentlyContinue
      if (-not $matches) { Write-Error "no match for $($ex.from) in $bin" }
      $src = $matches[0].FullName
      $dst = Join-Path $outDir $ex.to
      if ($type -eq "dir") {
        if (Test-Path $dst) { Remove-Item -Recurse -Force $dst }
        Copy-Item -Recurse $src $dst
      } else {
        Copy-Item -Force $src $dst
      }
    }
  }
  Write-Host "Bundled binaries for $Target written to $outDir"
} finally {
  Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
}
