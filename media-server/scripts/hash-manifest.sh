#!/usr/bin/env bash
# Reads media-server/deps/models/manifest.json, downloads each file to a temp
# dir, computes SHA-256, and prints a patched manifest to stdout. Pipe to
# `tee media-server/deps/models/manifest.json` once verified.
set -euo pipefail
in="${1:-media-server/deps/models/manifest.json}"
tmpdir="$(mktemp -d)"
trap "rm -rf $tmpdir" EXIT
jq -c '.models[] | {id, files: .files[]}' "$in" | while read -r line; do
  id=$(echo "$line" | jq -r '.id')
  url=$(echo "$line" | jq -r '.files.url')
  rel=$(echo "$line" | jq -r '.files.rel_path')
  echo "fetching $id/$rel ..." 1>&2
  out="$tmpdir/$(echo "$url" | shasum | cut -c1-12)"
  curl -fsSL "$url" -o "$out"
  sum=$(shasum -a 256 "$out" | awk '{print $1}')
  echo "$id $rel $sum"
done
