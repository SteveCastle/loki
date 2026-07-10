#!/usr/bin/env bash
# Assemble "Lowkey Media Server.app" from a staged release payload.
#
#   make-macos-app.sh <payload-dir> <out-dir> <version>
#
# payload-dir is a staged release dir (lowkeymediaserver, lokictl, bin/,
# licenses/). Bundled deps live under Contents/Resources/bin with a
# Contents/MacOS/bin symlink: the server resolves deps relative to its
# executable (so the symlink keeps that working unchanged), but codesign
# treats everything under Contents/MacOS as nested code — exiftool's Perl
# modules there make bundle signing fail with "code object is not signed
# at all". Signing/notarization happen in the workflow, after this.
set -euo pipefail

PAYLOAD="${1:?payload dir}"
OUT="${2:?out dir}"
VERSION="${3:?version}"

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
APP="$OUT/Lowkey Media Server.app"

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

cp "$PAYLOAD/lowkeymediaserver" "$APP/Contents/MacOS/"
cp "$PAYLOAD/lokictl"           "$APP/Contents/MacOS/"
cp -R "$PAYLOAD/bin"            "$APP/Contents/Resources/bin"
ln -s ../Resources/bin          "$APP/Contents/MacOS/bin"
cp -R "$PAYLOAD/licenses"       "$APP/Contents/Resources/licenses"

sed "s/APP_VERSION/$VERSION/g" \
  "$ROOT/media-server/packaging/macos/Info.plist" > "$APP/Contents/Info.plist"

# Reuse the Electron app icon (repo root assets/).
cp "$ROOT/assets/icon.icns" "$APP/Contents/Resources/icon.icns"

echo "built: $APP"
