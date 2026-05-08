#!/bin/bash
# Script to download Linux binaries (FFmpeg + UnRAR)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FFMPEG_VERSION="master-latest"
FFMPEG_URL="https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-${FFMPEG_VERSION}-linux64-gpl.tar.xz"
# RARLAB pins by version; bump when ready to roll forward.
UNRAR_VERSION="710"
UNRAR_URL="https://www.rarlab.com/rar/rarlinux-x64-${UNRAR_VERSION}.tar.gz"

echo "Downloading binaries for Linux..."

# Create temp directory
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

# --- FFmpeg ---
echo "Downloading FFmpeg from: $FFMPEG_URL"
curl -L "$FFMPEG_URL" -o "$TEMP_DIR/ffmpeg.tar.xz"
echo "Extracting FFmpeg..."
tar -xf "$TEMP_DIR/ffmpeg.tar.xz" -C "$TEMP_DIR"
cp "$TEMP_DIR"/ffmpeg-*/bin/ffmpeg "$SCRIPT_DIR/"
cp "$TEMP_DIR"/ffmpeg-*/bin/ffprobe "$SCRIPT_DIR/"
chmod +x "$SCRIPT_DIR/ffmpeg" "$SCRIPT_DIR/ffprobe"

# --- UnRAR ---
echo "Downloading UnRAR from: $UNRAR_URL"
curl -L "$UNRAR_URL" -o "$TEMP_DIR/rar.tar.gz"
echo "Extracting UnRAR..."
tar -xzf "$TEMP_DIR/rar.tar.gz" -C "$TEMP_DIR"
cp "$TEMP_DIR/rar/unrar" "$SCRIPT_DIR/"
chmod +x "$SCRIPT_DIR/unrar"

echo "✓ Linux binaries installed successfully!"
echo "  - ffmpeg:  $(file "$SCRIPT_DIR/ffmpeg" | cut -d: -f2-)"
echo "  - ffprobe: $(file "$SCRIPT_DIR/ffprobe" | cut -d: -f2-)"
echo "  - unrar:   $(file "$SCRIPT_DIR/unrar" | cut -d: -f2-)"
