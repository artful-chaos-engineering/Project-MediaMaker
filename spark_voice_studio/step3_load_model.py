#!/usr/bin/env python3
"""Step 3 - Load the model into the Spark's memory.

Starts the model server on the Spark if it is not running, then tells it which model to hold:
  --mode clone   (default) the Base model with a voice prompt from data/voice/ref.wav (works before any training)
  --mode custom  a fine-tuned checkpoint from step 5 (checkpoints/<run>/checkpoint-epoch-N)

    python step3_load_model.py
    python step3_load_model.py --mode custom --path checkpoints/run1/checkpoint-epoch-3
    python step3_load_model.py --stop
"""
import argparse, sys, time
import requests
from studio.common import Spark, cfg, say, set_state, state


def wait_health(url, tries=60):
    for _ in range(tries):
        try:
            r = requests.get(url + "/health", timeout=3)
            if r.ok: return r.json()
        except requests.RequestException:
            pass
        time.sleep(2)
    return None


def main():
    c = cfg(); ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["clone", "custom"], default="clone")
    ap.add_argument("--path", help="model path on the Spark; default = the downloaded Base model, or the last trained checkpoint for custom")
    ap.add_argument("--ref-audio", help="reference clip on the Spark for clone mode (default data/voice/ref.wav)")
    ap.add_argument("--ref-text", help="transcript of the reference clip (default data/voice/ref.txt)")
    ap.add_argument("--stop", action="store_true"); args = ap.parse_args()
    sp = Spark(); url = sp.url(); port = c["server"]["port"]
    if args.stop:
        sp.run(f"pkill -f studio/model_server.py || true", stream=False); set_state(model_loaded=None); say("server stopped"); return
    if wait_health(url, tries=1) is None:
        say("starting the model server on the Spark")
        sp.run("mkdir -p out", stream=False, check=False)
        sp.detach("python studio/model_server.py", log="out/model_server.log")
        if wait_health(url) is None:
            sys.exit("server did not come up; see out/model_server.log on the Spark")
    path = args.path
    if path is None:
        path = ("models/" + c["models"]["tts"].replace("/", "__")) if args.mode == "clone" else state().get("last_checkpoint")
        if path is None: sys.exit("no trained checkpoint recorded yet; run step 5 or pass --path")
    say(f"loading {path} as {args.mode} (bf16 on cuda:0)")
    payload = {"path": path, "mode": args.mode}
    if args.ref_audio: payload["ref_audio"] = args.ref_audio
    if args.ref_text: payload["ref_text"] = args.ref_text
    r = requests.post(url + "/load", json=payload, timeout=900); r.raise_for_status()
    h = r.json(); set_state(model_loaded=path, model_mode=args.mode)
    say(f"loaded in {h['seconds']} s; GPU free {h['gpu_free_gib']} GiB; server at {url}")


if __name__ == "__main__":
    main()
