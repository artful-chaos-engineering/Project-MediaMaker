#!/usr/bin/env bash
# Run the whole pipeline for one talktrack.   ./run_pipeline.sh talktracks/example.txt [options]
# Options are passed to pipeline.py (--draft, --background gradient, --from render, --dry-run, ...). --i-am-the-subject is added for you
# only if you export CONFIRM_OWNER=yes, so the consent gate stays a deliberate act.
set -euo pipefail
cd "$(dirname "$0")"
[ $# -ge 1 ] || { echo "usage: $0 talktrack.txt [pipeline.py options]"; exit 2; }
TT="$1"; shift
extra=()
[ "${CONFIRM_OWNER:-}" = "yes" ] && extra+=(--i-am-the-subject)
python3 talktrack.py "$TT"
exec python3 pipeline.py "$TT" "${extra[@]}" "$@"
