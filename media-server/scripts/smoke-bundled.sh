#!/usr/bin/env bash
# Boots a freshly-built server in the release layout, polls /api/deps/status,
# asserts every bundled dep is "ready" or "broken" (NOT "missing"), then
# stops the server. Used as a post-build CI check.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
INSTALL_DIR="${1:-$ROOT/media-server}"

cd "$INSTALL_DIR"
PORT=18762
LOWKEY_PORT=$PORT ./lowkeymediaserver &
PID=$!
trap "kill $PID 2>/dev/null || true" EXIT

# Wait up to 10s for the listener.
for i in $(seq 1 20); do
  if curl -fsS "http://127.0.0.1:$PORT/api/deps/status" >/tmp/status.json 2>/dev/null; then
    break
  fi
  sleep 0.5
done

if ! [ -s /tmp/status.json ]; then
  echo "server did not serve /api/deps/status within 10s" >&2
  exit 1
fi

# Every bundled entry must NOT be "missing".
missing=$(jq -r '[.[] | select(.category=="bundled" and .state=="missing")] | length' /tmp/status.json)
if [ "$missing" -ne 0 ]; then
  echo "smoke failed: $missing bundled deps missing" >&2
  jq '[.[] | select(.category=="bundled" and .state=="missing")]' /tmp/status.json >&2
  exit 1
fi
echo "smoke OK"
