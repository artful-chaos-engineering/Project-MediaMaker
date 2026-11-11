#!/usr/bin/env bash
# Show both studios' recorded state and the finished outputs.
cd "$(dirname "$0")/.."
for s in spark_voice_studio spark_avatar_studio; do echo "== $s"; cat "$s/state.json" 2>/dev/null || echo "(no state)"; echo; done
echo "== finished"; ls -1 pipeline/output 2>/dev/null || echo "(none yet)"
