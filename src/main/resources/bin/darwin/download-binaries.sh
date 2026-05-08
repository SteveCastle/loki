#!/bin/bash
# Script to download macOS binaries (FFmpeg + UnRAR)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# RARLAB pins by version; bump when ready to roll forward. Universal binary
# (handles both x86_64 and arm64).
UNRAR_VERSION="710"
UNRAR_URL="https://www.rarlab.com/rar/rarmacos-x64-${UNRAR_VERSION}.tar.gz"

echo "Downloading binaries for macOS..."

# Create temp directory
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

# Detect architecture
ARCH=$(uname -m)
if [ "$ARCH" = "arm64" ]; then
    FFMPEG_URL="https://evermeet.cx/ffmpeg/getrelease/ffmpeg/zip"
    FFPROBE_URL="https://evermeet.cx/ffmpeg/getrelease/ffprobe/zip"
else
    FFMPEG_URL="https://evermeet.cx/ffmpeg/getrelease/ffmpeg/zip"
    FFPROBE_URL="https://evermeet.cx/ffmpeg/getrelease/ffprobe/zip"
fi

# --- FFmpeg ---
echo "Downloading ffmpeg..."
curl -L "$FFMPEG_URL" -o "$TEMP_DIR/ffmpeg.zip"
unzip -q "$TEMP_DIR/ffmpeg.zip" -d "$TEMP_DIR"
cp "$TEMP_DIR/ffmpeg" "$SCRIPT_DIR/"
chmod +x "$SCRIPT_DIR/ffmpeg"

echo "Downloading ffprobe..."
curl -L "$FFPROBE_URL" -o "$TEMP_DIR/ffprobe.zip"
unzip -q "$TEMP_DIR/ffprobe.zip" -d "$TEMP_DIR"
cp "$TEMP_DIR/ffprobe" "$SCRIPT_DIR/"
chmod +x "$SCRIPT_DIR/ffprobe"

# --- UnRAR ---
echo "Downloading UnRAR from: $UNRAR_URL"
curl -L "$UNRAR_URL" -o "$TEMP_DIR/rar.tar.gz"
echo "Extracting UnRAR..."
tar -xzf "$TEMP_DIR/rar.tar.gz" -C "$TEMP_DIR"
cp "$TEMP_DIR/rar/unrar" "$SCRIPT_DIR/"
chmod +x "$SCRIPT_DIR/unrar"

echo "✓ macOS binaries installed successfully!"
echo "  - ffmpeg:  $(file "$SCRIPT_DIR/ffmpeg" | cut -d: -f2-)"
echo "  - ffprobe: $(file "$SCRIPT_DIR/ffprobe" | cut -d: -f2-)"
echo "  - unrar:   $(file "$SCRIPT_DIR/unrar" | cut -d: -f2-)"
