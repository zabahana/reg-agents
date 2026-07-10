#!/usr/bin/env bash
# Stop everything started by run_local.sh
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -f .run/pids ]]; then
  while read -r pid name; do
    if kill "$pid" 2>/dev/null; then
      echo "stopped $name ($pid)"
    fi
  done < .run/pids
  rm -f .run/pids
else
  echo "no .run/pids file; nothing to stop"
fi
