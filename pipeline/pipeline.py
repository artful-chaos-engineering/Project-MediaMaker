#!/usr/bin/env python3
"""pipeline.py - talktrack in, finished talking-avatar video out.

Stages (each calls the existing studio scripts; nothing is reimplemented):
  talktrack  validate the script, write the cleaned text          (this folder)
  voice      make sure the Voice Studio model server holds a model (voice step 3, only if needed)
  audio      speak the talktrack in your voice -> WAV + SRT        (avatar step 6)
  render     lip-sync your avatar to the audio, apply background   (avatar step 7)
  assess     automatic checks on the render                        (avatar step 9 --auto)
  deliver    copy video, audio, captions, talktrack to output/<name>/

One-time setup (connect, download, capture, train, build avatar) is NOT repeated here; see setup_once.sh.

    python pipeline.py talktracks/example.txt --i-am-the-subject
    python pipeline.py talktracks/example.txt --i-am-the-subject --draft --background gradient
    python pipeline.py talktracks/example.txt --i-am-the-subject --from render      # reuse existing audio
    python pipeline.py talktracks/example.txt --dry-run
"""
import argparse, json, shutil, subprocess, sys, time
from pathlib import Path
import talktrack

HERE = Path(__file__).resolve().parent
VOICE = HERE.parent / "spark_voice_studio"
AVATAR = HERE.parent / "spark_avatar_studio"
STAGES = ["talktrack", "voice", "audio", "render", "assess", "deliver"]


def run(cmd, cwd, dry):
    print(f"\n$ (cd {cwd.relative_to(HERE.parent) if cwd != HERE else "pipeline"} && {' '.join(map(str, cmd))})", flush=True)
    if dry: return ""
    cmd = [str(c) for c in cmd]
    # Use the studio's own venv (torch, soundfile, ...) when it exists on this machine; it sits at the studio's remote_home.
    home = {VOICE: "voice-studio", AVATAR: "avatar-studio"}.get(cwd)
    venv_py = Path.home() / home / ".venv/bin/python" if home else None
    if venv_py and venv_py.exists() and cmd[0] == sys.executable: cmd[0] = str(venv_py)
    p = subprocess.run(cmd, cwd=cwd, text=True)
    if p.returncode: sys.exit(f"stage failed (exit {p.returncode}): {' '.join(map(str, cmd))}")


def voice_health():
    """Ask the voice server (through Voice Studio's own tunnel) what it holds; None when not reachable."""
    code = ("import json,requests\nfrom studio.common import Spark\n"
            "try: print(json.dumps(requests.get(Spark().url()+'/health',timeout=5).json()))\n"
            "except Exception: print('null')")
    p = subprocess.run([sys.executable, "-c", code], cwd=VOICE, capture_output=True, text=True)
    try: return json.loads(p.stdout.strip().splitlines()[-1])
    except Exception: return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("talktrack"); ap.add_argument("--i-am-the-subject", action="store_true", help="confirm the avatar and voice are your own")
    ap.add_argument("--from", dest="start", choices=STAGES, default="talktrack", help="start at this stage (reuses earlier outputs)")
    ap.add_argument("--only", choices=STAGES); ap.add_argument("--skip", nargs="*", default=[], choices=STAGES)
    ap.add_argument("--voice-mode", choices=["clone", "custom"]); ap.add_argument("--background"); ap.add_argument("--bg-image"); ap.add_argument("--bg-video")
    ap.add_argument("--draft", action="store_true", help="fast low-quality render (15 steps)"); ap.add_argument("--steps", type=int); ap.add_argument("--guidance", type=float)
    ap.add_argument("--takes", type=int, help="speak each paragraph N times and keep the best (default: @takes in the talktrack, else 1)")
    ap.add_argument("--metrics", action="store_true", help="also measure lip-sync confidence, word error rate and voice similarity (step 9 --metrics)")
    ap.add_argument("--engine", choices=["infinitetalk", "latentsync"], default="infinitetalk", help="infinitetalk = natural full-face video from your real footage (slow); latentsync = mouth-only on a looped photo/clip (fast)")
    ap.add_argument("--clip", default="s01", help="reference clip for infinitetalk: s01, s03, s10, s14 or a path")
    ap.add_argument("--size", choices=["infinitetalk-480", "infinitetalk-720"], default="infinitetalk-720"); ap.add_argument("--lora", default="auto", help="face LoRA: auto, none or a path")
    ap.add_argument("--name"); ap.add_argument("--dry-run", action="store_true"); a = ap.parse_args()

    try: st, paras, warns = talktrack.parse(a.talktrack)
    except (OSError, ValueError) as e: sys.exit(f"talktrack error: {e}")
    name = a.name or st["name"]; pause = st.get("pause", "0.5"); lang = st.get("language", "English")
    mode = a.voice_mode or st.get("voice_mode", "clone"); bg = a.background or st.get("background")
    steps = a.steps or (15 if a.draft else (int(st["steps"]) if "steps" in st else None)); guid = a.guidance or (float(st["guidance"]) if "guidance" in st else None)
    takes = a.takes or int(st.get("takes", 1))
    if not a.i_am_the_subject and not a.dry_run and a.only not in ("talktrack",):
        sys.exit("refused: add --i-am-the-subject to confirm the voice and face are your own")

    todo = [s for s in STAGES[STAGES.index(a.start):] if s not in a.skip and (not a.only or s == a.only)]
    clean = AVATAR / "data" / "talktracks" / f"{name}.txt"; wav = Path("data/audio") / f"{name}.wav"
    t0 = time.time(); print(f"talktrack '{name}': {len(paras)} paragraphs, about {talktrack.seconds(paras, float(pause)):.0f} s of speech; stages: {', '.join(todo)}")

    if "talktrack" in todo:
        for w in warns: print(f"  warning: {w}")
        if not a.dry_run: talktrack.write_clean(paras, clean)
        print(f"  clean text -> {clean.relative_to(HERE.parent)}")
    if "voice" in todo:
        h = None if a.dry_run else voice_health()
        if h and h.get("loaded") and h.get("mode") == mode: print(f"  voice server already holds a {mode} model; leaving it")
        else: run([sys.executable, "step3_load_model.py", "--mode", mode], VOICE, a.dry_run)
    if "audio" in todo:
        run([sys.executable, "step6_make_audio.py", f"data/talktracks/{name}.txt", "--name", name, "--pause", pause, "--language", lang, "--takes", takes], AVATAR, a.dry_run)
    if "render" in todo and a.engine == "infinitetalk":
        run([sys.executable, HERE / "render_infinitetalk.py", "--audio", AVATAR / wav, "--name", name, "--clip", a.clip, "--size", a.size, "--lora", a.lora], HERE, a.dry_run)
        if "assess" in todo: todo.remove("assess"); print("  (assess skipped: its checks are written for the latentsync render)")
    elif "render" in todo:
        cmd = [sys.executable, "step7_render_avatar.py", "--i-am-the-subject", "--audio", wav, "--name", name]
        if bg: cmd += ["--background", bg]
        if a.bg_image: cmd += ["--bg-image", a.bg_image]
        if a.bg_video: cmd += ["--bg-video", a.bg_video]
        if steps: cmd += ["--steps", steps]
        if guid: cmd += ["--guidance", guid]
        run(cmd, AVATAR, a.dry_run)
    if "assess" in todo:
        run([sys.executable, "step9_assess_tune.py", "--auto", name], AVATAR, a.dry_run)
        if a.metrics: run([sys.executable, "step9_assess_tune.py", "--metrics", name], AVATAR, a.dry_run)
    if "deliver" in todo:
        dest = HERE / "output" / name; src = AVATAR / "out" / name
        print(f"\n  delivering to {dest.relative_to(HERE.parent)}")
        if not a.dry_run:
            dest.mkdir(parents=True, exist_ok=True)
            pairs = [(src / f"{name}_final.mp4", dest), (src / f"{name}_raw.mp4", dest), (AVATAR / "data/audio" / f"{name}.wav", dest),
                     (AVATAR / "data/audio" / f"{name}.srt", dest), (Path(a.talktrack), dest)]
            got = [s.name for s, d in pairs if s.exists() and shutil.copy(s, d)]
            if f"{name}_final.mp4" not in got: sys.exit(f"no final video found at {src}; check the render stage output")
            print("  " + ", ".join(got))
    print(f"\ndone in {time.time() - t0:.0f} s" + ("  (dry run: nothing was executed)" if a.dry_run else ""))


if __name__ == "__main__":
    main()
