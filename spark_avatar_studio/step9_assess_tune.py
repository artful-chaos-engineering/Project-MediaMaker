#!/usr/bin/env python3
"""Step 9 - Assess the avatar and tune it.

A lip-sync avatar has no per-person training: LatentSync follows your footage and your audio. What you can tune are the
render settings and the footage, and this step finds the best settings by experiment on a short excerpt.

    python step9_assess_tune.py --auto lesson                 # automatic checks: length match, black frames, flicker spikes
    python step9_assess_tune.py --play lesson                 # watch it, score sync / likeness / artifacts / background 1-5
    python step9_assess_tune.py --sweep offset --audio data/audio/lesson.wav     # 5 renders of a 6 s excerpt, lips earlier/later
    python step9_assess_tune.py --sweep quality --audio data/audio/lesson.wav    # steps 15/25 x guidance 1.5/2.0
    python step9_assess_tune.py --metrics lesson              # numbers: lip-sync confidence (SyncNet), word error rate, voice similarity
    python step9_assess_tune.py --apply tune_offset_p04       # make that variant's settings the default for step 7
    python step9_assess_tune.py --report
If lips are consistently early or late, sweep offset. If the mouth looks mushy, sweep quality. If the head looks frozen or
the loop jumps, the fix is the footage (step 4: record 30 to 60 s, not a photo).
"""
import argparse, base64, datetime, json, re, shutil, subprocess, sys, tempfile
from pathlib import Path
import cv2, numpy as np, requests
from studio.common import ROOT, Spark, cfg, say, set_state, state
from studio.media import ff, probe
from studio.takes import score as score_audio

OUT = ROOT / "out"; ASSESS = OUT / "assessments.jsonl"; METRICS = OUT / "metrics.jsonl"
GRID = {"offset": [{"offset": o} for o in (-0.08, -0.04, 0.0, 0.04, 0.08)],
        "quality": [{"steps": s, "guidance": g} for s in (15, 25) for g in (1.5, 2.0)]}


def auto(name):
    v = OUT / name / f"{name}_final.mp4"
    if not v.exists(): sys.exit(f"{v} not found")
    i = probe(v); cap = cv2.VideoCapture(str(v)); prev = None; diffs = []; black = 0; n = 0
    while True:
        ok, f = cap.read()
        if not ok: break
        g = cv2.cvtColor(cv2.resize(f, (160, 90)), cv2.COLOR_BGR2GRAY).astype(np.float32); n += 1
        black += g.mean() < 8
        if prev is not None: diffs.append(float(np.abs(g - prev).mean()))
        prev = g
    d = np.array(diffs) if diffs else np.zeros(1); spikes = int((d > max(4 * np.median(d), 3.0)).sum())
    man = json.loads((OUT / name / "manifest.json").read_text()) if (OUT / name / "manifest.json").exists() else {}
    rows = [("video seconds", f"{i['seconds']:.2f}"), ("audio seconds (manifest)", man.get("seconds", "?")), ("fps", f"{i['fps']:.2f}"), ("resolution", f"{i['width']}x{i['height']}"),
            ("has audio", i["has_audio"]), ("black frames", f"{black}/{n}"), ("frame-to-frame jumps (cuts or flicker)", spikes)]
    for k, val in rows: print(f"  {k:42} {val}")
    flags = []
    if man.get("seconds") and abs(i["seconds"] - man["seconds"]) > 0.3: flags.append("video and audio lengths differ by more than 0.3 s")
    if black > 2: flags.append("black frames present")
    if spikes > 3: flags.append(f"{spikes} sudden jumps: a loop seam or matte flicker; check chunk joins and the loop window")
    print("  " + ("; ".join(flags) if flags else "no automatic problems found"))


def audio_metrics(name, video):
    """Whisper word error rate and voice similarity of the finished video's own soundtrack, through the voice server."""
    man = ROOT / "data" / "audio" / f"{name}.json"
    if not man.exists(): return {"audio": "skipped: no data/audio/%s.json (the text that was spoken)" % name}
    text = " ".join(p["text"] for p in json.loads(man.read_text())["paragraphs"])
    with tempfile.TemporaryDirectory() as d:
        wav = Path(d) / "a.wav"; ff("-i", video, "-vn", "-ac", "1", "-ar", "24000", wav)
        try: s = score_audio(Spark().voice_url(), wav.read_bytes(), text)
        except (requests.RequestException, RuntimeError) as e: return {"audio": f"skipped: voice server /score unreachable ({str(e)[:80]}); load a model and restart the server after updating it"}
    return {"wer": s["wer"], "speaker_sim": s["speaker_sim"], "chars_per_sec": s["chars_per_sec"]}


def sync_metrics(name, video):
    """Lip-sync confidence and distance from LatentSync's bundled SyncNet evaluation, run on the Spark.
    Higher confidence (LSE-C) and lower distance (LSE-D) are better; compare variants and runs against each other."""
    c = cfg(); m = c.get("metrics", {}); ls = c["models"]["latentsync_dir"]; sp = Spark()
    have = sp.run(f"test -f {ls}/{m['syncnet_weights']} && echo yes || echo no", check=False, stream=False).strip().splitlines()[-1]
    if have != "yes":
        return {"sync": f"skipped: SyncNet weights not found at {ls}/{m['syncnet_weights']}. Download LatentSync's auxiliary checkpoints "
                        "(see the 'Evaluation' section of its README) into that folder, or change metrics.syncnet_weights in studio.yaml"}
    rel = f"out/_metrics/{name}"; sp.run(f"rm -rf {rel} && mkdir -p {rel}", stream=False); sp.put(video, f"{rel}/{name}.mp4")
    out = sp.run(f"cd {ls} && " + m["syncnet_cmd"].format(dir=f"{sp.home}/{rel}"), check=False, stream=False)
    conf = re.findall(r"(?:sync[ _]?conf\w*|LSE-?C)\D{0,20}?(-?\d+\.\d+)", out, re.I); dist = re.findall(r"(?:min[ _]?dist\w*|LSE-?D)\D{0,20}?(-?\d+\.\d+)", out, re.I)
    if not conf: return {"sync": "ran but its output was not recognised; last lines: " + " | ".join(out.strip().splitlines()[-3:])}
    r = {"lse_c": float(conf[-1])}
    if dist: r["lse_d"] = float(dist[-1])
    return r


def metrics(name, quiet=False):
    v = OUT / name / f"{name}_final.mp4"
    if not v.exists(): sys.exit(f"{v} not found")
    res = {**audio_metrics(name, v), **sync_metrics(name, v)}
    notes = {k: val for k, val in res.items() if k in ("audio", "sync")}; nums = {k: val for k, val in res.items() if k not in notes}
    for k, val in nums.items(): print(f"  {k:42} {val}")
    for k, val in notes.items(): print(f"  {k:42} {val}")
    if nums:
        OUT.mkdir(exist_ok=True); man = json.loads((OUT / name / "manifest.json").read_text()) if (OUT / name / "manifest.json").exists() else {}
        with open(METRICS, "a") as o: o.write(json.dumps({"when": datetime.datetime.now().isoformat(timespec="seconds"), "name": name, "steps": man.get("steps"),
                                                           "guidance": man.get("guidance"), "offset": man.get("offset"), **nums}) + "\n")
    return res


def play(path):
    cmd = ["open", "-W", str(path)] if shutil.which("open") else ["ffplay", "-autoexit", "-loglevel", "quiet", str(path)]
    subprocess.run(cmd)


def score(name, extra=None):
    v = OUT / name / f"{name}_final.mp4"; man = json.loads((OUT / name / "manifest.json").read_text()) if (OUT / name / "manifest.json").exists() else {}
    while True:
        play(v)
        a = input("  scores sync/likeness/artifacts/background 1-5 (e.g. 4 5 4 3; 0 for no background), Enter to replay, s skip: ").strip()
        if a == "": continue
        if a.lower() == "s": return
        try: sy, li, ar, bg = [int(x) for x in a.split()[:4]]; break
        except ValueError: print("  four numbers, please")
    note = input("  note (what looked wrong; Enter for none): ").strip()
    OUT.mkdir(exist_ok=True)
    with open(ASSESS, "a") as o:
        o.write(json.dumps({"when": datetime.datetime.now().isoformat(timespec="seconds"), "name": name, "sync": sy, "likeness": li, "artifacts": ar, "background": bg,
                            "steps": man.get("steps"), "guidance": man.get("guidance"), "offset": man.get("offset"), "bg_mode": man.get("background"), "note": note}) + "\n")


def report():
    if METRICS.exists():
        print("measured (higher lse_c, lower lse_d and wer, higher speaker_sim are better):")
        print(f"{'name':24} {'steps':>5} {'guid':>5} {'offs':>6} {'lse_c':>6} {'lse_d':>6} {'wer':>6} {'sim':>6}")
        for r in (json.loads(l) for l in METRICS.read_text().splitlines()):
            f = lambda k: "-" if r.get(k) is None else f"{r[k]:.3f}"
            print(f"{r['name'][:24]:24} {str(r['steps']):>5} {str(r['guidance']):>5} {str(r['offset']):>6} {f('lse_c'):>6} {f('lse_d'):>6} {f('wer'):>6} {f('speaker_sim'):>6}")
        print()
    rows = [json.loads(l) for l in ASSESS.read_text().splitlines()] if ASSESS.exists() else []
    if not rows: print("no assessments yet"); return
    print(f"{'name':24} {'steps':>5} {'guid':>5} {'offs':>6} {'sync':>5} {'like':>5} {'artif':>5} {'bg':>3}")
    for r in rows: print(f"{r['name'][:24]:24} {str(r['steps']):>5} {str(r['guidance']):>5} {str(r['offset']):>6} {r['sync']:5d} {r['likeness']:5d} {r['artifacts']:5d} {r['background']:3d}")


def sweep(kind, audio):
    cut = ROOT / "data" / "audio" / "_excerpt.wav"; cut.parent.mkdir(parents=True, exist_ok=True)
    ff("-i", audio, "-t", 6, cut)
    variants = {}
    for k, kw in enumerate(GRID[kind], 1):
        name = f"tune_{kind}_p{k:02d}"; variants[name] = kw
        flags = sum(([f"--{a}", str(b)] for a, b in kw.items()), [])
        say(f"{name}: {kw}")
        subprocess.run([sys.executable, str(ROOT / "step7_render_avatar.py"), "--i-am-the-subject", "--audio", str(cut), "--name", name, "--background", "keep", *flags], cwd=ROOT, check=True)
    (OUT / f"tune_{kind}_variants.json").write_text(json.dumps(variants, indent=1))
    say("all variants rendered; measuring, then watch each and score it")
    for name in variants:
        print(f"\n== {name}  {variants[name]}")
        try: metrics(name)
        except (SystemExit, RuntimeError) as e: print(f"  metrics skipped: {e}")
        score(name)
    report(); say("pick the best row and run: python step9_assess_tune.py --apply <name>")


def apply(name):
    for f in OUT.glob("tune_*_variants.json"):
        v = json.loads(f.read_text())
        if name in v:
            t = state().get("tuned", {}); t.update(v[name]); set_state(tuned=t); say(f"step 7 will now default to {t}"); return
    sys.exit(f"{name} not found in any sweep")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--auto"); ap.add_argument("--play"); ap.add_argument("--sweep", choices=list(GRID)); ap.add_argument("--audio")
    ap.add_argument("--apply"); ap.add_argument("--metrics"); ap.add_argument("--report", action="store_true"); a = ap.parse_args()
    if a.metrics: metrics(a.metrics)
    elif a.auto: auto(a.auto)
    elif a.play: auto(a.play); score(a.play)
    elif a.sweep:
        if not a.audio: sys.exit("--sweep needs --audio FILE")
        sweep(a.sweep, a.audio)
    elif a.apply: apply(a.apply)
    else: report()


if __name__ == "__main__":
    main()
