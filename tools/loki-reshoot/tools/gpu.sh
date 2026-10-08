#!/usr/bin/env bash
# Serialize GPU jobs across workers/agents: bash tools/gpu.sh <command...>
# Holds a mkdir-lock (<repo root>/.gpu.lock) while the command runs; stale locks older than 60 min are broken.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK="$ROOT/.gpu.lock"
while ! mkdir "$LOCK" 2>/dev/null; do
  if [ -d "$LOCK" ] && [ -n "$(find "$LOCK" -maxdepth 0 -mmin +60 2>/dev/null)" ]; then rmdir "$LOCK" 2>/dev/null; fi
  sleep 3
done
trap 'rmdir "$LOCK" 2>/dev/null' EXIT INT TERM
"$@"
