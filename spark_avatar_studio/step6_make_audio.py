#!/usr/bin/env python3
"""Step 6 - Turn text into the speech the avatar will lip-sync.

Sends each paragraph of a text file to the Spark Voice Studio model server (your cloned or fine-tuned voice), joins the
clips with short pauses, and writes data/audio/<name>.wav plus <name>.srt (one caption per paragraph) and <name>.json.
Or bring your own recording: --audio file.wav copies it in unchanged.

    python step6_make_audio.py lesson.txt
    python step6_make_audio.py lesson.txt --pause 0.6 --language English
    python step6_make_audio.py --audio my_recording.wav --name intro
    python step6_make_audio.py lesson.txt --takes 4        # speak each paragraph 4 times, keep the best-scoring take
With --takes the winner is chosen by Whisper word error rate, similarity to your reference voice and speaking pace; every take and
its scores are kept in data/audio/<name>_takes/ so you can listen to the alternatives. Restart the voice server once after updating it.
Start the voice server first (Spark Voice Studio step 3).
"""
import argparse, json, re, shutil
from pathlib import Path
import numpy as np, requests, soundfile as sf
from studio.common import ROOT, Spark, cfg, say, set_state
from studio.takes import best_of

OUT = ROOT / "data" / "audio"


def srt_time(t):
    h, r = divmod(t, 3600); m, s = divmod(r, 60); return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int((s % 1) * 1000):03d}"


def main():
    c = cfg(); ap = argparse.ArgumentParser(); ap.add_argument("textfile", nargs="?"); ap.add_argument("--audio"); ap.add_argument("--name")
    ap.add_argument("--pause", type=float, default=0.5); ap.add_argument("--language", default="English")
    ap.add_argument("--takes", type=int, default=1, help="speak each paragraph this many times and keep the best"); args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.audio:
        name = args.name or Path(args.audio).stem; shutil.copy(args.audio, OUT / f"{name}.wav")
        i = sf.info(args.audio); say(f"imported {args.audio}: {i.duration:.1f} s at {i.samplerate} Hz -> {OUT / (name + '.wav')}")
        set_state(last_audio=name); return
    if not args.textfile: ap.error("give a text file or --audio")
    name = args.name or Path(args.textfile).stem; url = Spark().voice_url()
    try: h = requests.get(url + "/health", timeout=5).json(); assert h["loaded"]
    except Exception: raise SystemExit(f"voice server at {url} is not loaded; start Spark Voice Studio step 3 first (or use --audio)")
    paras = [p.strip() for p in re.split(r"\n\s*\n", Path(args.textfile).read_text()) if p.strip() and not p.lstrip().startswith("#")]
    say(f"{len(paras)} paragraphs through {h['model']} ({h['mode']})")
    parts, sr, t, caps, report = [], None, 0.0, [], []
    for i, p in enumerate(paras, 1):
        content, pick, sc, clips = best_of(url, p, args.takes, args.language)
        tmp = OUT / "_clip.wav"; tmp.write_bytes(content); w, sr = sf.read(tmp, dtype="float32"); tmp.unlink()
        if sc:
            tk = OUT / f"{name}_takes"; tk.mkdir(exist_ok=True)
            for k, cw in enumerate(clips, 1): (tk / f"p{i:02d}_take{k}.wav").write_bytes(cw)
            report.append({"paragraph": i, "chosen": pick, "takes": sc})
            say(f"p{i:02d}: take {pick + 1} of {len(sc)}  wer " + " ".join(f"{x['wer']:.2f}" for x in sc) + "  cost " + " ".join(f"{x['cost']:.2f}" for x in sc))
        if w.ndim > 1: w = w[:, 0]
        caps.append((t, t + len(w) / sr, p)); parts += [w, np.zeros(int(args.pause * sr), np.float32)]; t += len(w) / sr + args.pause
        say(f"p{i:02d}: {len(w)/sr:.1f} s")
    audio = np.concatenate(parts[:-1]); sf.write(OUT / f"{name}.wav", audio, sr, subtype="PCM_16")
    (OUT / f"{name}.srt").write_text("".join(f"{n}\n{srt_time(a)} --> {srt_time(b)}\n{txt}\n\n" for n, (a, b, txt) in enumerate(caps, 1)))
    (OUT / f"{name}.json").write_text(json.dumps({"voice_model": h["model"], "mode": h["mode"], "seconds": round(len(audio) / sr, 2),
                                                  "paragraphs": [{"start": round(a, 2), "end": round(b, 2), "text": x} for a, b, x in caps]}, indent=1))
    if report: (OUT / f"{name}_takes").mkdir(exist_ok=True); (OUT / f"{name}_takes" / "scores.json").write_text(json.dumps(report, indent=1))
    set_state(last_audio=name); say(f"wrote {OUT / (name + '.wav')} ({len(audio)/sr:.1f} s), captions {name}.srt")


if __name__ == "__main__":
    main()
