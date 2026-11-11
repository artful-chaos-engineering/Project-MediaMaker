#!/usr/bin/env python3
"""studio.py - Spark Voice Studio master menu.

A text menu over the seven step scripts. Each step is a separate Python program with its own
command line, so the menu does nothing but ask what you want, build the arguments, and run it.
Run with no arguments for the menu, or `python studio.py 1 2 3` to run steps in order unattended.
"""
import subprocess, sys
from pathlib import Path
from studio.common import ROOT, cfg, state

STEPS = {
    "1": ("Connect to the DGX Spark (SSH, GPU check, environment, sync code)", "step1_connect_spark.py", []),
    "2": ("Download the models (Qwen3-TTS Base, tokenizer, Whisper, fine-tuning scripts)", "step2_download_model.py", []),
    "3": ("Load a model into the Spark's memory (start the model server)", "step3_load_model.py", ["mode", "path"]),
    "4": ("Capture your voice (guided recording or import, then upload)", "step4_capture_audio.py", ["speaker", "import"]),
    "5": ("Train: fine-tune the voice model on your recordings", "step5_train_voice.py", ["speaker", "epochs"]),
    "6": ("Play back a text file in the trained voice and score it", "step6_playback_text.py", ["textfile"]),
    "7": ("Fine-tune on keywords (record the hard words, resume training, or add lexicon fixes)", "step7_keyword_tune.py", ["speaker", "keywords"]),
    "8": ("Show status", None, []),
    "9": ("Stop the model server", "step3_load_model.py", ["stop"]),
    "0": ("Quit", None, []),
}


def ask(prompt, default=""):
    v = input(f"  {prompt}{f' [{default}]' if default else ''}: ").strip()
    return v or default


def build_args(opts):
    a = []
    if "speaker" in opts:
        if ask("Are these recordings of your own voice? (yes/no)", "yes").lower().startswith("y"): a.append("--i-am-the-speaker")
        else: print("  the studio clones the owner's voice only; nothing run"); return None
    if "mode" in opts:
        mode = ask("Mode: clone (Base model + reference clip) or custom (trained checkpoint)", "clone"); a += ["--mode", mode]
        if mode == "custom":
            p = ask("Checkpoint path on the Spark", state().get("last_checkpoint", ""))
            if p: a += ["--path", p]
    if "import" in opts:
        imp = ask("Import existing recordings? Paths separated by spaces, or Enter for a guided session")
        if imp: a += ["--import", *imp.split()]
    if "epochs" in opts: a += ["--epochs", ask("Epochs", str(cfg()["training"]["epochs"]))]
    if "textfile" in opts: a.append(ask("Text file to speak", "sample.txt"))
    if "keywords" in opts:
        mode = ask("Record keyword sentences (r), lexicon fix only (l), or train on existing keyword recordings (t)", "r")
        if mode == "l":
            a = ["--lexicon", *ask("Pairs WRITTEN SPOKEN ... (e.g. MBSE 'M B S E')").replace("'", "").split()]
        elif mode == "t": a.append("--train-only")
        else:
            k = ask("Keywords (space-separated; Enter = step 6's flagged words)")
            if k: a += ["--keywords", *k.split()]
    if "stop" in opts: a.append("--stop")
    return a


def status():
    s = state()
    if not s: print("  nothing recorded yet; start with 1"); return
    for k, v in s.items(): print(f"  {k:18} {v}")


def run_step(n, interactive=True):
    title, script, opts = STEPS[n]
    print(f"\n== {n}. {title}")
    if script is None:
        if n == "8": status()
        return 0
    args = build_args(opts) if interactive else ([] if "speaker" not in opts else ["--i-am-the-speaker"])
    if args is None: return 1
    if n == "6" and not interactive: args = ["sample.txt", "--no-play"]
    return subprocess.call([sys.executable, str(ROOT / script), *args], cwd=ROOT)


def main():
    if len(sys.argv) > 1:
        for n in sys.argv[1:]:
            if run_step(n, interactive=False): sys.exit(1)
        return
    c = cfg()["spark"]
    print(f"\nSpark Voice Studio  ·  {c['user']}@{c['host']}  ·  owner: {cfg()['owner']}")
    while True:
        print()
        for k, (t, _, _) in STEPS.items(): print(f"  {k}  {t}")
        try:
            n = input("\n> ").strip()
        except EOFError:
            break
        if n == "0": break
        if n in STEPS: run_step(n)
        else: print("  choose a number from the list")


if __name__ == "__main__":
    main()
