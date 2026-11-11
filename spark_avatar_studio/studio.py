#!/usr/bin/env python3
"""studio.py - Spark Avatar Studio master menu.

A text menu over the nine step scripts. Each step is its own Python program with its own command line, so the menu only
asks what you want, builds the arguments, and runs it. No arguments = menu; `python studio.py 1 2 3` runs steps in order.
"""
import subprocess, sys
from studio.common import ROOT, cfg, state

STEPS = {
    "1": ("Connect to the DGX Spark (SSH, GPU check, environment, LatentSync)", "step1_connect_spark.py", []),
    "2": ("Download the models (LatentSync 1.6, VAE, BiRefNet matting)", "step2_download_models.py", []),
    "3": ("Load the models into memory and test (add a smoke render after step 5)", "step3_load_models.py", ["smoke"]),
    "4": ("Capture photos / idle video of yourself, check them, upload", "step4_capture_media.py", ["subject", "capture"]),
    "5": ("Build the avatar (loop window, background preview)", "step5_build_avatar.py", ["subject", "bgpreview"]),
    "6": ("Make the speech: text file -> your cloned voice (or import audio)", "step6_make_audio.py", ["textorwav"]),
    "7": ("Render the talking avatar (lip-sync + background)", "step7_render_avatar.py", ["subject", "audio", "bg"]),
    "8": ("Change the background of a finished render (no new lip-sync)", "step8_background.py", ["name", "bgmode"]),
    "9": ("Assess and tune (checks, scoring, offset / quality sweeps)", "step9_assess_tune.py", ["assess"]),
    "s": ("Show status", None, []),
    "q": ("Quit", None, []),
}


def ask(prompt, default=""):
    v = input(f"  {prompt}{f' [{default}]' if default else ''}: ").strip()
    return v or default


def build_args(opts):
    a = []
    if "subject" in opts:
        if ask("Are these files of your own face, or of someone who gave you written consent? (yes/no)", "yes").lower().startswith("y"): a.append("--i-am-the-subject")
        else: print("  the studio animates the owner's likeness only; nothing run"); return None
    if "smoke" in opts and ask("Run the lip-sync smoke test too? (needs step 5 done) (yes/no)", "no").lower().startswith("y"): a.append("--smoke")
    if "capture" in opts:
        m = ask("Capture: photos (p), idle video (v), import files (i), re-check saved files (c)", "v")
        if m == "p": a.append("--photos")
        elif m == "v": a.append("--video")
        elif m == "c": a = ["--check-only"]
        else: a += ["--import", *ask("Paths separated by spaces").split()]
    if "bgpreview" in opts:
        b = ask("Preview a background (keep/color/gradient/image/video/blur/transparent) or Enter for studio.yaml"); a += ["--background", b] if b else []
    if "textorwav" in opts:
        t = ask("Text file to speak, or a WAV to import", "sample.txt")
        a += ["--audio", t] if t.lower().endswith(".wav") else [t]
    if "audio" in opts:
        t = ask("Audio file to lip-sync (step 6 writes data/audio/<name>.wav), or 'tone' for a smoke test", "")
        if t == "tone": a += ["--tone", "3", "--steps", "10", "--name", "smoke"]
        elif t: a += ["--audio", t]
        else: print("  an audio file is needed"); return None
    if "bg" in opts:
        b = ask("Background (Enter = studio.yaml; keep/color/gradient/image/video/blur/transparent)")
        if b: a += ["--background", b]
        if b == "image": a += ["--bg-image", ask("Image path")]
        if b == "video": a += ["--bg-video", ask("Video path")]
        s = ask("Steps / guidance (Enter = defaults, or step 9's tuned values)")
        if s: st = s.split(); a += ["--steps", st[0]] + (["--guidance", st[1]] if len(st) > 1 else [])
    if "name" in opts: a.append(ask("Render name (out/<name>/)", state().get("last_render", "")))
    if "bgmode" in opts:
        m = ask("Mode: color, gradient, image, video, blur, transparent, keep", "color"); a += ["--mode", m]
        if m == "color": a += ["--color", ask("Colour", "#0B1F3A")]
        if m == "image": a += ["--image", ask("Image path")]
        if m == "video": a += ["--video", ask("Video path")]
        if ask("Green/blue screen footage? use the chroma key (yes/no)", "no").lower().startswith("y"): a += ["--engine", "chroma"]
    if "assess" in opts:
        m = ask("auto <name> | play <name> | sweep offset | sweep quality | apply <variant> | report", "report").split()
        if m[0] in ("auto", "play"): a += [f"--{m[0]}", m[1]]
        elif m[0] == "sweep": a += ["--sweep", m[1], "--audio", ask("Audio file for the 6 s excerpt")]
        elif m[0] == "apply": a += ["--apply", m[1]]
        else: a.append("--report")
    return a


def status():
    s = state()
    if not s: print("  nothing recorded yet; start with 1"); return
    for k, v in s.items(): print(f"  {k:20} {v}")


def run_step(n, interactive=True):
    title, script, opts = STEPS[n]
    print(f"\n== {n}. {title}")
    if script is None:
        if n == "s": status()
        return 0
    if interactive: args = build_args(opts)
    else:
        args = ["--i-am-the-subject"] if "subject" in opts else []
        if n == "4": args = ["--check-only"]
        if n == "6": args = ["sample.txt"]
        if n == "7": args += ["--tone", "3", "--steps", "10", "--name", "smoke"]
        if n == "8": args = ["smoke"]
        if n == "9": args = ["--report"]
    if args is None: return 1
    return subprocess.call([sys.executable, str(ROOT / script), *args], cwd=ROOT)


def main():
    if len(sys.argv) > 1:
        for n in sys.argv[1:]:
            if run_step(n, interactive=False): sys.exit(1)
        return
    c = cfg()["spark"]
    print(f"\nSpark Avatar Studio  ·  {c['user']}@{c['host']}  ·  owner: {cfg()['owner']}")
    while True:
        print()
        for k, (t, _, _) in STEPS.items(): print(f"  {k}  {t}")
        try: n = input("\n> ").strip().lower()
        except EOFError: break
        if n == "q": break
        if n in STEPS: run_step(n)
        else: print("  choose a number from the list")


if __name__ == "__main__":
    main()
