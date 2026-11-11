#!/usr/bin/env bash
# Run every talktrack in a folder (default talktracks/), one after another; keeps going if one fails and lists failures at the end.
#   CONFIRM_OWNER=yes ./run_batch.sh [folder] [pipeline.py options]
set -uo pipefail
cd "$(dirname "$0")"
DIR="${1:-talktracks}"; [ $# -ge 1 ] && shift
fail=()
for f in "$DIR"/*.txt; do
  echo "=================== $f"
  ./run_pipeline.sh "$f" "$@" || fail+=("$f")
done
[ ${#fail[@]} -eq 0 ] && echo "all talktracks finished" || { echo "failed: ${fail[*]}"; exit 1; }
