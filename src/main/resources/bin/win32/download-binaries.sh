#!/bin/bash
# Script to download Windows binaries (FFmpeg + UnRAR)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FFMPEG_URL="https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
# RARLAB ships UnRAR for Windows as a self-extracting console SFX. The
# unversioned URL always points at the latest build; rarlab does not
# publish stable versioned URLs for this particular SFX. We run it with
# -s2 (fully silent) -d<dir> -y to drop UnRAR.exe into our bin/ without
# any GUI.
UNRAR_URL="https://www.rarlab.com/rar/unrarw64.exe"

echo "Downloading binaries for Windows..."

# Create temp directory
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

# --- FFmpeg ---
echo "Downloading FFmpeg from: $FFMPEG_URL"
curl -L "$FFMPEG_URL" -o "$TEMP_DIR/ffmpeg.zip"
echo "Extracting FFmpeg..."
unzip -q "$TEMP_DIR/ffmpeg.zip" -d "$TEMP_DIR"
cp "$TEMP_DIR"/ffmpeg-*/bin/ffmpeg.exe "$SCRIPT_DIR/"
cp "$TEMP_DIR"/ffmpeg-*/bin/ffprobe.exe "$SCRIPT_DIR/"

# --- UnRAR ---
echo "Downloading UnRAR from: $UNRAR_URL"
curl -L "$UNRAR_URL" -o "$TEMP_DIR/unrar-installer.exe"
echo "Extracting UnRAR.exe from installer..."
# Convert to a Windows path for the SFX so it can find the destination.
WIN_TARGET="$(cygpath -w "$TEMP_DIR/unrar-out" 2>/dev/null || echo "$TEMP_DIR/unrar-out")"
mkdir -p "$TEMP_DIR/unrar-out"
# -s2 = silent (no UI), -d<dir> = destination, -y = yes to all prompts.
"$TEMP_DIR/unrar-installer.exe" -s2 -d"$WIN_TARGET" -y || true
# The SFX exits non-zero in some headless environments even when extraction
# succeeds, so verify the binary landed instead of trusting the exit code.
if [ ! -f "$TEMP_DIR/unrar-out/UnRAR.exe" ]; then
  echo "Error: UnRAR.exe not found after running installer." >&2
  echo "If the SFX requires interactive consent in this environment," >&2
  echo "extract it manually and copy UnRAR.exe to $SCRIPT_DIR/unrar.exe" >&2
  exit 1
fi
cp "$TEMP_DIR/unrar-out/UnRAR.exe" "$SCRIPT_DIR/unrar.exe"

echo "✓ Windows binaries installed successfully!"
echo "  - ffmpeg.exe:  $(file "$SCRIPT_DIR/ffmpeg.exe" | cut -d: -f2-)"
echo "  - ffprobe.exe: $(file "$SCRIPT_DIR/ffprobe.exe" | cut -d: -f2-)"
echo "  - unrar.exe:   $(file "$SCRIPT_DIR/unrar.exe" | cut -d: -f2-)"
