#!/bin/bash
# Script to download macOS binaries (FFmpeg + UnRAR)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# RARLAB pins by version; bump when ready to roll forward. RARLAB ships
# separate x64 and arm64 builds; both are fetched and lipo'd together.
UNRAR_VERSION="710"
UNRAR_URL_X64="https://www.rarlab.com/rar/rarmacos-x64-${UNRAR_VERSION}.tar.gz"
UNRAR_URL_ARM64="https://www.rarlab.com/rar/rarmacos-arm-${UNRAR_VERSION}.tar.gz"

# FFmpeg: Martin Riedl's static release builds, pinned to one immutable
# build directory per architecture (the site keeps history) and checked
# against the published SHA-256. evermeet.cx used to serve these, but it
# rate-limits CI — the second download in a row comes back HTTP 429 and
# the error page landed on disk as "ffmpeg.zip". Same pins as
# media-server/scripts/bundled-versions.json; bump both together.
#
# Every binary here is made UNIVERSAL (lipo of the arm64 and x86_64
# builds): CI packages the arm64 and x64 DMGs from this one folder on an
# Apple Silicon runner, so a host-arch download shipped arm64 ffmpeg to
# Intel Macs.
FFMPEG_VERSION="9.0.2"
FFMPEG_BASE_ARM64="https://ffmpeg.martin-riedl.de/download/macos/arm64/1789931890_${FFMPEG_VERSION}"
FFMPEG_SHA256_ARM64="c8ed4c4e6978a03c485edbfe4e0a5dc2380f8a30bba5150531b31b094492d924"
FFPROBE_SHA256_ARM64="fcbe839537485eaee7a7a8bc5cbc0f90d53617e80943e8a5b2e31cb851197ea6"
FFMPEG_BASE_AMD64="https://ffmpeg.martin-riedl.de/download/macos/amd64/1789931006_${FFMPEG_VERSION}"
FFMPEG_SHA256_AMD64="7c6b4125b191cbf773832dc51f424cf2b6bb7da43007d1e066f95909e47cacd4"
FFPROBE_SHA256_AMD64="2322438ed2f6319a691291b247d09c69dcaa3a982460d1f269a7e1af335cfdfd"

echo "Downloading universal binaries for macOS..."

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

# fetch_zip_bin <url> <sha256> <name> <arch>: one zipped binary into
# $TEMP_DIR/<arch>/<name>.
fetch_zip_bin() {
    local url="$1" want="$2" name="$3" arch="$4"
    mkdir -p "$TEMP_DIR/$arch"
    fetch "$url" "$TEMP_DIR/$arch/$name.zip" "$want"
    unzip -q -o "$TEMP_DIR/$arch/$name.zip" -d "$TEMP_DIR/$arch"
}

# --- FFmpeg ---
echo "Downloading ffmpeg + ffprobe ${FFMPEG_VERSION} (arm64, x86_64)..."
fetch_zip_bin "$FFMPEG_BASE_ARM64/ffmpeg.zip" "$FFMPEG_SHA256_ARM64" ffmpeg arm64
fetch_zip_bin "$FFMPEG_BASE_ARM64/ffprobe.zip" "$FFPROBE_SHA256_ARM64" ffprobe arm64
fetch_zip_bin "$FFMPEG_BASE_AMD64/ffmpeg.zip" "$FFMPEG_SHA256_AMD64" ffmpeg x86_64
fetch_zip_bin "$FFMPEG_BASE_AMD64/ffprobe.zip" "$FFPROBE_SHA256_AMD64" ffprobe x86_64

# --- UnRAR ---
echo "Downloading UnRAR ${UNRAR_VERSION} (arm64, x86_64)..."
for pair in "arm64 $UNRAR_URL_ARM64" "x86_64 $UNRAR_URL_X64"; do
    set -- $pair
    mkdir -p "$TEMP_DIR/$1/rar-pkg"
    fetch "$2" "$TEMP_DIR/$1/rar.tar.gz"
    tar -xzf "$TEMP_DIR/$1/rar.tar.gz" -C "$TEMP_DIR/$1/rar-pkg"
    cp "$TEMP_DIR/$1/rar-pkg/rar/unrar" "$TEMP_DIR/$1/unrar"
done

# --- Universal binaries ---
for name in ffmpeg ffprobe unrar; do
    lipo -create "$TEMP_DIR/arm64/$name" "$TEMP_DIR/x86_64/$name" -output "$SCRIPT_DIR/$name"
    chmod +x "$SCRIPT_DIR/$name"
    lipo "$SCRIPT_DIR/$name" -verify_arch arm64 x86_64         || { echo "$name is not universal" >&2; exit 1; }
done

echo "✓ macOS binaries installed successfully!"
echo "  - ffmpeg:  $(file "$SCRIPT_DIR/ffmpeg" | cut -d: -f2-)"
echo "  - ffprobe: $(file "$SCRIPT_DIR/ffprobe" | cut -d: -f2-)"
echo "  - unrar:   $(file "$SCRIPT_DIR/unrar" | cut -d: -f2-)"
