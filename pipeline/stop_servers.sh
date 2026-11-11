#!/usr/bin/env bash
# Free the Spark's memory: stop the voice model server.
cd "$(dirname "$0")/../spark_voice_studio" && python3 step3_load_model.py --stop
