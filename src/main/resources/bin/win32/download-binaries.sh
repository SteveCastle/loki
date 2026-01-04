#!/bin/bash
# Script to download Windows FFmpeg binaries

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FFMPEG_URL="https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"

echo "Downloading FFmpeg binaries for Windows..."
echo "This will download approximately 80MB and extract binaries."

# Create temp directory
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

# Download
echo "Downloading from: $FFMPEG_URL"
curl -L "$FFMPEG_URL" -o "$TEMP_DIR/ffmpeg.zip"

# Extract
echo "Extracting..."
unzip -q "$TEMP_DIR/ffmpeg.zip" -d "$TEMP_DIR"

# Copy binaries
echo "Installing binaries..."
cp "$TEMP_DIR"/ffmpeg-*/bin/ffmpeg.exe "$SCRIPT_DIR/"
cp "$TEMP_DIR"/ffmpeg-*/bin/ffprobe.exe "$SCRIPT_DIR/"

echo "✓ FFmpeg binaries for Windows installed successfully!"
echo "  - ffmpeg.exe: $(file "$SCRIPT_DIR/ffmpeg.exe" | cut -d: -f2-)"
echo "  - ffprobe.exe: $(file "$SCRIPT_DIR/ffprobe.exe" | cut -d: -f2-)"
