#!/usr/bin/env bash
# One-time preparation of both studios, in the order the READMEs give. Steps that need you (camera, microphone) are guided, so this
# stops and tells you when it reaches them. Safe to re-run: finished steps are quick checks.
set -euo pipefail
cd "$(dirname "$0")/.."
V=spark_voice_studio; A=spark_avatar_studio
[ "${CONFIRM_OWNER:-}" = "yes" ] || { echo "export CONFIRM_OWNER=yes to confirm the voice and face are your own"; exit 2; }
echo "== Voice Studio: connect, download"; (cd $V && python3 step1_connect_spark.py && python3 step2_download_model.py)
echo "== Avatar Studio: connect, download"; (cd $A && python3 step1_connect_spark.py && python3 step2_download_models.py)
[ -f $V/data/voice/ref.wav ] || { echo "Next: record your voice:  cd $V && python3 step4_capture_audio.py --i-am-the-speaker"; exit 0; }
[ -d $A/avatars/mark ] || { echo "Next: capture your face then build:  cd $A && python3 step4_capture_media.py --i-am-the-subject --video && python3 step5_build_avatar.py --i-am-the-subject"; exit 0; }
echo "Optional, for the fine-tuned voice:  cd $V && python3 step5_train_voice.py --i-am-the-speaker"
echo "Setup complete. Try:  CONFIRM_OWNER=yes pipeline/run_pipeline.sh pipeline/talktracks/example.txt --draft"
