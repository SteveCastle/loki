#!/bin/bash
# Script to download macOS binaries (FFmpeg + UnRAR)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# RARLAB pins by version; bump when ready to roll forward. Universal binary
# (handles both x86_64 and arm64).
UNRAR_VERSION="710"
UNRAR_URL="https://www.rarlab.com/rar/rarmacos-x64-${UNRAR_VERSION}.tar.gz"

# FFmpeg: Martin Riedl's static release builds, pinned to one immutable
# build directory per architecture (the site keeps history) and checked
# against the published SHA-256. evermeet.cx used to serve these, but it
# rate-limits CI — the second download in a row comes back HTTP 429 and
# the error page landed on disk as "ffmpeg.zip". Same pins as
# media-server/scripts/bundled-versions.json; bump both together.
FFMPEG_VERSION="9.0.2"
ARCH=$(uname -m)
if [ "$ARCH" = "arm64" ]; then
    FFMPEG_BASE="https://ffmpeg.martin-riedl.de/download/macos/arm64/1789931890_${FFMPEG_VERSION}"
    FFMPEG_SHA256="c8ed4c4e6978a03c485edbfe4e0a5dc2380f8a30bba5150531b31b094492d924"
    FFPROBE_SHA256="fcbe839537485eaee7a7a8bc5cbc0f90d53617e80943e8a5b2e31cb851197ea6"
else
    FFMPEG_BASE="https://ffmpeg.martin-riedl.de/download/macos/amd64/1789931006_${FFMPEG_VERSION}"
    FFMPEG_SHA256="7c6b4125b191cbf773832dc51f424cf2b6bb7da43007d1e066f95909e47cacd4"
    FFPROBE_SHA256="2322438ed2f6319a691291b247d09c69dcaa3a982460d1f269a7e1af335cfdfd"
fi

echo "Downloading binaries for macOS ($ARCH)..."

# Create temp directory
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

# fetch <url> <dest> [sha256] — fails on HTTP errors instead of saving the
# error page, retries transient failures, and verifies the pin when given.
fetch() {
    local url="$1" dest="$2" want="${3:-}"
    curl -fL --retry 5 --retry-delay 15 --retry-all-errors "$url" -o "$dest"
    if [ -n "$want" ]; then
        local got
        got=$(shasum -a 256 "$dest" | awk '{print $1}')
        if [ "$got" != "$want" ]; then
            echo "SHA-256 mismatch for $url" >&2
            echo "  want: $want" >&2
            echo "  got:  $got" >&2
            exit 1
        fi
    fi
}

# --- FFmpeg ---
echo "Downloading ffmpeg ${FFMPEG_VERSION}..."
fetch "$FFMPEG_BASE/ffmpeg.zip" "$TEMP_DIR/ffmpeg.zip" "$FFMPEG_SHA256"
unzip -q "$TEMP_DIR/ffmpeg.zip" -d "$TEMP_DIR"
cp "$TEMP_DIR/ffmpeg" "$SCRIPT_DIR/"
chmod +x "$SCRIPT_DIR/ffmpeg"

echo "Downloading ffprobe ${FFMPEG_VERSION}..."
fetch "$FFMPEG_BASE/ffprobe.zip" "$TEMP_DIR/ffprobe.zip" "$FFPROBE_SHA256"
unzip -q "$TEMP_DIR/ffprobe.zip" -d "$TEMP_DIR"
cp "$TEMP_DIR/ffprobe" "$SCRIPT_DIR/"
chmod +x "$SCRIPT_DIR/ffprobe"

# --- UnRAR ---
echo "Downloading UnRAR from: $UNRAR_URL"
fetch "$UNRAR_URL" "$TEMP_DIR/rar.tar.gz"
echo "Extracting UnRAR..."
tar -xzf "$TEMP_DIR/rar.tar.gz" -C "$TEMP_DIR"
cp "$TEMP_DIR/rar/unrar" "$SCRIPT_DIR/"
chmod +x "$SCRIPT_DIR/unrar"

echo "✓ macOS binaries installed successfully!"
echo "  - ffmpeg:  $(file "$SCRIPT_DIR/ffmpeg" | cut -d: -f2-)"
echo "  - ffprobe: $(file "$SCRIPT_DIR/ffprobe" | cut -d: -f2-)"
echo "  - unrar:   $(file "$SCRIPT_DIR/unrar" | cut -d: -f2-)"
