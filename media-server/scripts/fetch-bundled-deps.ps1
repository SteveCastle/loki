# Windows equivalent of fetch-bundled-deps.sh. PowerShell 5.1 compatible.
param(
  [Parameter(Mandatory=$true)][string]$Target,
  [ValidateSet("verify","update")][string]$Mode = "verify"
)
$ErrorActionPreference = "Stop"

# Invoke-WebRequest is notoriously slow on PowerShell 5.1 because it writes
# a Write-Progress event on every chunk. For multi-hundred-MB downloads
# (ffmpeg, onnxruntime) this can be 10-100x slower than a browser. Disabling
# the progress UI fixes it.
$ProgressPreference = 'SilentlyContinue'

$root  = (Resolve-Path "$PSScriptRoot\..\..").Path
$conf  = Join-Path $root "media-server\scripts\bundled-versions.json"
$outDir = Join-Path $root "media-server\bin\$Target"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$data = Get-Content $conf -Raw | ConvertFrom-Json
$tmp  = New-Item -ItemType Directory -Path ([System.IO.Path]::GetTempPath() + [System.Guid]::NewGuid().ToString())

$targetGoos, $targetGoarch = $Target -split '-', 2

try {
  foreach ($bin in $data.binaries.PSObject.Properties.Name) {
    $entry = $data.binaries.$bin.$Target
    if ($null -eq $entry) { continue }

    $extractDir = Join-Path $tmp $bin
    New-Item -ItemType Directory -Force -Path $extractDir | Out-Null

    # ---- "build" entries compile from a local Go package, no download ----
    if ($entry.archive -eq "build") {
      $source = $entry.source
      $outName = $entry.extract[0].to  # use the destination filename as the binary name
      $outPath = Join-Path $extractDir $outName
      Write-Host "building $bin from $source ($targetGoos/$targetGoarch) ..."
      $env:GOOS = $targetGoos
      $env:GOARCH = $targetGoarch
      $env:CGO_ENABLED = "0"
      try {
        Push-Location (Join-Path $root "media-server")
        & go build -ldflags="-s -w" -o $outPath $source
        if ($LASTEXITCODE -ne 0) { Write-Error "go build failed for $bin" }
      } finally {
        Pop-Location
        Remove-Item Env:GOOS, Env:GOARCH, Env:CGO_ENABLED -ErrorAction SilentlyContinue
      }
      if ($Mode -eq "update") {
        $gotSum = (Get-FileHash -Algorithm SHA256 $outPath).Hash.ToLower()
        Write-Host "SHA256 $bin $Target $gotSum (built)"
      }
      # Fall through to the extract loop with the built binary already in place.
    } else {
      # Expand-Archive demands a .zip extension, and `tar` is happier when the
      # extension matches the content — so name the temp file accordingly.
      $archiveExt = switch ($entry.archive) {
        "zip"    { ".zip" }
        "tar.gz" { ".tar.gz" }
        "tar.xz" { ".tar.xz" }
        "none"   { [IO.Path]::GetExtension([IO.Path]::GetFileName($entry.url)) }
        default  { ".bin" }
      }
      $archivePath = Join-Path $tmp ("$bin$archiveExt")
      Write-Host "fetching $bin ($($entry.url)) ..."
      Invoke-WebRequest -Uri $entry.url -OutFile $archivePath -UseBasicParsing
      $gotSum = (Get-FileHash -Algorithm SHA256 $archivePath).Hash.ToLower()

      if ($Mode -eq "update") {
        Write-Host "SHA256 $bin $Target $gotSum"
      } else {
        if ($entry.sha256 -ne "TO_FILL" -and $entry.sha256 -ne $gotSum) {
          Write-Error "SHA256 mismatch for $bin $Target`n  want: $($entry.sha256)`n  got:  $gotSum"
        }
      }

      switch ($entry.archive) {
        "zip"    { Expand-Archive -Path $archivePath -DestinationPath $extractDir -Force }
        "tar.gz" { tar -xzf $archivePath -C $extractDir }
        "tar.xz" { tar -xJf $archivePath -C $extractDir }
        "none"   { Copy-Item $archivePath (Join-Path $extractDir ([IO.Path]::GetFileName($entry.url))) }
        default  { Write-Error "unknown archive type $($entry.archive)" }
      }
    }

    foreach ($ex in $entry.extract) {
      $type = if ($ex.type) { $ex.type } else { "file" }
      # $matches is an automatic variable in PowerShell — avoid shadowing it.
      $found = Get-ChildItem -Path $extractDir -Recurse -Filter ([IO.Path]::GetFileName($ex.from)) -ErrorAction SilentlyContinue
      if (-not $found) { Write-Error "no match for $($ex.from) in $bin" }
      $src = $found[0].FullName
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
