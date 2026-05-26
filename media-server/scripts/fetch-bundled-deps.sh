#!/usr/bin/env bash
# Fetch bundled binaries for one GOOS-GOARCH target into media-server/bin/<target>/.
# Reads media-server/scripts/bundled-versions.json. Verifies SHA-256 if not
# "TO_FILL"; with --update prints discovered SHA-256s to stdout instead.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CONF="$ROOT/media-server/scripts/bundled-versions.json"
TARGET="${1:-}"
MODE="${2:-verify}"   # verify | update
if [ -z "$TARGET" ]; then
  echo "usage: $0 <goos-goarch> [verify|update]" >&2
  exit 2
fi

OUTDIR="$ROOT/media-server/bin/$TARGET"
mkdir -p "$OUTDIR"

tmpdir="$(mktemp -d)"
trap "rm -rf $tmpdir" EXIT

target_goos="${TARGET%-*}"
target_goarch="${TARGET#*-}"

# List binaries that have an entry for $TARGET.
bins=$(jq -r --arg t "$TARGET" '.binaries | to_entries[] | select(.value[$t] != null) | .key' "$CONF")
for bin in $bins; do
  archive=$(jq -r --arg t "$TARGET" --arg b "$bin" '.binaries[$b][$t].archive' "$CONF")
  extract_dir="$tmpdir/$bin"
  mkdir -p "$extract_dir"

  if [ "$archive" = "build" ]; then
    # Compile the named Go package, no download.
    source=$(jq -r --arg t "$TARGET" --arg b "$bin" '.binaries[$b][$t].source' "$CONF")
    out_name=$(jq -r --arg t "$TARGET" --arg b "$bin" '.binaries[$b][$t].extract[0].to' "$CONF")
    echo "building $bin from $source ($target_goos/$target_goarch) ..."
    (
      cd "$ROOT/media-server"
      GOOS="$target_goos" GOARCH="$target_goarch" CGO_ENABLED=0 \
        go build -ldflags="-s -w" -o "$extract_dir/$out_name" "$source"
    )
    if [ "$MODE" = "update" ]; then
      got_sum=$(shasum -a 256 "$extract_dir/$out_name" | awk '{print $1}')
      echo "SHA256 $bin $TARGET $got_sum (built)"
    fi
  else
    url=$(jq -r --arg t "$TARGET" --arg b "$bin" '.binaries[$b][$t].url' "$CONF")
    want_sum=$(jq -r --arg t "$TARGET" --arg b "$bin" '.binaries[$b][$t].sha256' "$CONF")

    archive_path="$tmpdir/$bin.archive"
    echo "fetching $bin ($url) ..."
    curl -fsSL "$url" -o "$archive_path"
    got_sum=$(shasum -a 256 "$archive_path" | awk '{print $1}')

    if [ "$MODE" = "update" ]; then
      echo "SHA256 $bin $TARGET $got_sum"
    else
      if [ "$want_sum" != "TO_FILL" ] && [ "$want_sum" != "$got_sum" ]; then
        echo "SHA256 mismatch for $bin $TARGET" >&2
        echo "  want: $want_sum" >&2
        echo "  got:  $got_sum" >&2
        exit 1
      fi
    fi

    case "$archive" in
      zip)    unzip -q "$archive_path" -d "$extract_dir" ;;
      tar.gz) tar -xzf "$archive_path" -C "$extract_dir" ;;
      tar.xz) tar -xJf "$archive_path" -C "$extract_dir" ;;
      none)   cp "$archive_path" "$extract_dir/$(basename "$url")" ;;
      *) echo "unknown archive type $archive" >&2; exit 1 ;;
    esac
  fi

  # For each extract entry, glob and copy.
  jq -r --arg t "$TARGET" --arg b "$bin" '.binaries[$b][$t].extract[] | [.from, .to, (.type // "file")] | @tsv' "$CONF" |
  while IFS=$'\t' read -r from to type; do
    matches=( $(cd "$extract_dir" && ls -1 $from 2>/dev/null || true) )
    if [ ${#matches[@]} -eq 0 ]; then
      echo "no match for $from in $bin" >&2
      exit 1
    fi
    src="$extract_dir/${matches[0]}"
    dst="$OUTDIR/$to"
    if [ "$type" = "dir" ]; then
      rm -rf "$dst"
      cp -R "$src" "$dst"
    else
      cp "$src" "$dst"
      chmod +x "$dst" || true
    fi
  done
done

echo "Bundled binaries for $TARGET written to $OUTDIR"
