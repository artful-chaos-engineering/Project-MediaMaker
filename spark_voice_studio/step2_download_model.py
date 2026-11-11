#!/usr/bin/env python3
"""Step 2 - Download the models to the Spark.

Pulls the Qwen3-TTS Base model (the voice-cloning and fine-tuning base), its 12 Hz tokenizer (needed to
prepare training data), and Whisper (for transcripts of your recordings) into <remote_home>/models.
Also clones the Qwen3-TTS repository, whose finetuning/ folder holds the training scripts.

    python step2_download_model.py                 # models from studio.yaml
    python step2_download_model.py --tts Qwen/Qwen3-TTS-12Hz-0.6B-Base
"""
import argparse
from studio.common import Spark, cfg, say, set_state


def main():
    c = cfg(); ap = argparse.ArgumentParser()
    ap.add_argument("--tts", default=c["models"]["tts"]); ap.add_argument("--asr", default=c["models"]["asr"])
    args = ap.parse_args(); sp = Spark()
    for repo in (args.tts, c["models"]["tokenizer"], args.asr):
        local = "models/" + repo.replace("/", "__")
        say(f"downloading {repo}")
        sp.run(f"HF_HUB_ENABLE_HF_TRANSFER=0 huggingface-cli download {repo} --local-dir {local} | tail -1")
    qr = c["training"]["qwen_repo"]
    say("fetching the Qwen3-TTS repository (fine-tuning scripts)")
    sp.run(f"[ -d {qr} ] && (cd {qr} && git pull -q) || git clone -q --depth 1 https://github.com/QwenLM/Qwen3-TTS.git {qr}; ls {qr}/finetuning", venv=False)
    size = sp.run("du -sh models | cut -f1", stream=False).strip()
    set_state(tts_model=args.tts, asr_model=args.asr, models_downloaded=True)
    say(f"models on the Spark: {size}")


if __name__ == "__main__":
    main()
