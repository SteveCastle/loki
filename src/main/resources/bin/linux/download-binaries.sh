#!/bin/bash
# Script to download Linux FFmpeg binaries

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FFMPEG_VERSION="master-latest"
FFMPEG_URL="https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-${FFMPEG_VERSION}-linux64-gpl.tar.xz"

echo "Downloading FFmpeg binaries for Linux..."
echo "This will download approximately 130MB and extract ~380MB of binaries."

# Create temp directory
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

# Download
echo "Downloading from: $FFMPEG_URL"
curl -L "$FFMPEG_URL" -o "$TEMP_DIR/ffmpeg.tar.xz"

# Extract
echo "Extracting..."
tar -xf "$TEMP_DIR/ffmpeg.tar.xz" -C "$TEMP_DIR"

# Copy binaries
echo "Installing binaries..."
cp "$TEMP_DIR"/ffmpeg-*/bin/ffmpeg "$SCRIPT_DIR/"
cp "$TEMP_DIR"/ffmpeg-*/bin/ffprobe "$SCRIPT_DIR/"

# Make executable
chmod +x "$SCRIPT_DIR/ffmpeg" "$SCRIPT_DIR/ffprobe"

echo "✓ FFmpeg binaries installed successfully!"
echo "  - ffmpeg: $(file "$SCRIPT_DIR/ffmpeg" | cut -d: -f2-)"
echo "  - ffprobe: $(file "$SCRIPT_DIR/ffprobe" | cut -d: -f2-)"
