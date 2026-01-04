#!/bin/bash
# Script to download macOS FFmpeg binaries

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Downloading FFmpeg binaries for macOS..."
echo "This will download approximately 60MB of binaries."

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

# Download ffmpeg
echo "Downloading ffmpeg..."
curl -L "$FFMPEG_URL" -o "$TEMP_DIR/ffmpeg.zip"
unzip -q "$TEMP_DIR/ffmpeg.zip" -d "$TEMP_DIR"
cp "$TEMP_DIR/ffmpeg" "$SCRIPT_DIR/"
chmod +x "$SCRIPT_DIR/ffmpeg"

# Download ffprobe
echo "Downloading ffprobe..."
curl -L "$FFPROBE_URL" -o "$TEMP_DIR/ffprobe.zip"
unzip -q "$TEMP_DIR/ffprobe.zip" -d "$TEMP_DIR"
cp "$TEMP_DIR/ffprobe" "$SCRIPT_DIR/"
chmod +x "$SCRIPT_DIR/ffprobe"

echo "✓ FFmpeg binaries for macOS installed successfully!"
echo "  - ffmpeg: $(file "$SCRIPT_DIR/ffmpeg" | cut -d: -f2-)"
echo "  - ffprobe: $(file "$SCRIPT_DIR/ffprobe" | cut -d: -f2-)"
