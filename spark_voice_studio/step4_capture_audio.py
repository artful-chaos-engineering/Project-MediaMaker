#!/usr/bin/env python3
"""Step 4 - Capture your voice for training.

Runs where the microphone is (your Mac). Shows one sentence at a time from prompts/capture_sentences.txt,
records it, trims silence, saves data/voice/uttNNNN.wav with its transcript uttNNNN.txt, and picks the
cleanest 10 to 15 s clip as data/voice/ref.wav (the reference every training sample and the zero-shot
clone use). Then uploads the set to the Spark.

    python step4_capture_audio.py --i-am-the-speaker                 # guided session, N takes from studio.yaml
    python step4_capture_audio.py --i-am-the-speaker --import take1.wav take2.wav   # use existing recordings
    python step4_capture_audio.py --i-am-the-speaker --takes 20 --sentences my_sentences.txt
    python step4_capture_audio.py --upload-only
"""
import argparse, json, subprocess, sys, time
from pathlib import Path
import numpy as np, soundfile as sf
from studio.common import ROOT, Spark, cfg, say, require_owner, set_state

DATA = ROOT / "data" / "voice"


def trim(wav, sr, db=-40):
    thr, win = 10 ** (db / 20), int(0.05 * sr)
    env = np.array([np.abs(wav[i:i + win]).max() for i in range(0, len(wav), win)])
    idx = np.where(env > thr)[0]
    if not len(idx): return wav
    a, b = max(0, idx[0] * win - int(0.15 * sr)), min(len(wav), (idx[-1] + 1) * win + int(0.25 * sr))
    return wav[a:b]


def record(seconds, sr, device):
    import sounddevice as sd
    say("recording... (speak now)")
    audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="float32", device=device)
    sd.wait()
    return audio[:, 0]


def next_index():
    n = [int(p.stem[3:]) for p in DATA.glob("utt*.wav") if p.stem[3:].isdigit()]
    return max(n, default=0) + 1


def save_utt(wav, sr, text, idx):
    wav = trim(wav, sr); wav = wav / max(np.abs(wav).max(), 1e-6) * 10 ** (-3 / 20)
    f = DATA / f"utt{idx:04d}.wav"; sf.write(f, wav, sr, subtype="PCM_16"); f.with_suffix(".txt").write_text(text.strip() + "\n")
    return f, len(wav) / sr


def choose_ref():
    """Reference clip: the utterance between 8 and 15 s with the lowest noise floor; else concatenate the first few."""
    best, best_score = None, 1e9
    for f in sorted(DATA.glob("utt*.wav")):
        w, sr = sf.read(f, dtype="float32"); d = len(w) / sr
        if 6 <= d <= 15:
            floor = np.percentile(np.abs(w), 10); score = floor + (0.1 if d < 8 else 0)
            if score < best_score: best, best_score = (f, w, sr), score
    if best is None:
        parts, txt, sr = [], [], None
        for f in sorted(DATA.glob("utt*.wav"))[:4]:
            w, sr = sf.read(f, dtype="float32"); parts += [w, np.zeros(int(0.4 * sr), dtype="float32")]; txt.append(f.with_suffix(".txt").read_text().strip())
        if not parts: return None
        w = np.concatenate(parts)[: 15 * sr]; sf.write(DATA / "ref.wav", w, sr, subtype="PCM_16"); (DATA / "ref.txt").write_text(" ".join(txt) + "\n")
    else:
        f, w, sr = best; sf.write(DATA / "ref.wav", w, sr, subtype="PCM_16"); (DATA / "ref.txt").write_text(f.with_suffix(".txt").read_text()); (DATA / "ref.source").write_text(f.name + "\n")
    return DATA / "ref.wav"


def main():
    c = cfg(); ap = argparse.ArgumentParser()
    ap.add_argument("--i-am-the-speaker", action="store_true"); ap.add_argument("--takes", type=int, default=c["capture"]["takes"])
    ap.add_argument("--sentences", default=str(ROOT / "prompts/capture_sentences.txt")); ap.add_argument("--import", dest="imports", nargs="*")
    ap.add_argument("--upload-only", action="store_true"); ap.add_argument("--max-seconds", type=float, default=12)
    args = ap.parse_args(); DATA.mkdir(parents=True, exist_ok=True); sr = c["capture"]["sample_rate"]
    if not args.upload_only:
        require_owner(args)
        manifest = json.loads((DATA / "manifest.json").read_text()) if (DATA / "manifest.json").exists() else {"owner": c["owner"], "utterances": []}
        if args.imports:
            for path in args.imports:
                say(f"importing {path}")
                tmp = DATA / "_import.wav"
                subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", path, "-ac", "1", "-ar", str(sr), str(tmp)], check=True)
                w, _ = sf.read(tmp, dtype="float32"); tmp.unlink()
                # split long recordings at pauses into 4 to 15 s pieces; transcripts come from Whisper in step 5's prepare pass
                win = int(0.3 * sr); env = np.array([np.abs(w[i:i + win]).max() for i in range(0, len(w), win)]); cuts, start = [], 0
                for i, e in enumerate(env):
                    if e < 10 ** (-40 / 20) and (i * win - start) >= 4 * sr: cuts.append((start, i * win)); start = i * win
                    elif (i * win - start) >= 15 * sr: cuts.append((start, i * win)); start = i * win
                cuts.append((start, len(w)))
                for a, b in cuts:
                    if b - a < 2 * sr: continue
                    idx = next_index(); f, d = save_utt(w[a:b], sr, "", idx)
                    manifest["utterances"].append({"file": f.name, "seconds": round(d, 2), "text": "", "needs_transcript": True})
                    say(f"  {f.name} {d:.1f} s (transcript pending)")
        else:
            sentences = [s.strip() for s in Path(args.sentences).read_text().splitlines() if s.strip()][: args.takes]
            print("\nRead each sentence the way you teach. Enter = record, r = redo last, s = skip, q = quit.\n")
            i = 0
            while i < len(sentences):
                text = sentences[i]; print(f"\n[{i+1}/{len(sentences)}]  {text}")
                k = input("  Enter to record > ").strip().lower()
                if k == "q": break
                if k == "s": i += 1; continue
                if k == "r" and manifest["utterances"]:
                    last = manifest["utterances"].pop(); (DATA / last["file"]).unlink(missing_ok=True); (DATA / last["file"]).with_suffix(".txt").unlink(missing_ok=True); i -= 1; continue
                w = record(args.max_seconds, sr, c["capture"]["device"]); idx = next_index(); f, d = save_utt(w, sr, text, idx)
                manifest["utterances"].append({"file": f.name, "seconds": round(d, 2), "text": text}); say(f"  saved {f.name} ({d:.1f} s)"); i += 1
        (DATA / "manifest.json").write_text(json.dumps(manifest, indent=1))
        ref = choose_ref(); total = sum(u["seconds"] for u in manifest["utterances"])
        say(f"{len(manifest['utterances'])} utterances, {total/60:.1f} min; reference clip {ref.name if ref else 'none'}")
    sp = Spark(); say("uploading to the Spark")
    for f in sorted(DATA.iterdir()):
        if f.suffix in (".wav", ".txt", ".json"): sp.put(f, f"data/voice/{f.name}")
    set_state(captured=True, utterances=len(list(DATA.glob("utt*.wav"))))
    say("upload done")


if __name__ == "__main__":
    main()
