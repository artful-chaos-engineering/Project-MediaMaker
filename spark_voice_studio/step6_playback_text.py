#!/usr/bin/env python3
"""Step 6 - Play back a text file in the trained voice and assess it.

Sends each paragraph of a text file to the model server, saves the audio under out/<name>/, plays each
clip (afplay on macOS, ffplay elsewhere), and asks you to score it 1 to 5 on three things: does it sound
like you, is it natural, and did it say every word right. Words you flag go to data/keywords/wordlist.txt,
which step 7 uses. Scores accumulate in out/assessments.jsonl so runs can be compared.

    python step6_playback_text.py sample.txt
    python step6_playback_text.py sample.txt --no-play          # synthesize only
    python step6_playback_text.py --report                      # mean scores per model/run
"""
import argparse, datetime, json, re, shutil, subprocess, sys
from pathlib import Path
import requests
from studio.common import ROOT, Spark, say, state

OUT = ROOT / "out"; ASSESS = OUT / "assessments.jsonl"; WORDS = ROOT / "data" / "keywords" / "wordlist.txt"; LEX = ROOT / "data" / "keywords" / "lexicon.json"


def apply_lexicon(text):
    """Written form -> spoken form for the words step 7 put in the lexicon (acronyms, names)."""
    lex = json.loads(LEX.read_text()) if LEX.exists() else {}
    for k in sorted(lex, key=len, reverse=True):
        text = re.sub(rf"(?<![\w-]){re.escape(k)}(?![\w-])", lex[k], text)
    return text


def split_for_tts(text, max_chars):
    """Keep each synthesis request short enough that the model reaches its end token reliably."""
    if max_chars <= 0 or len(text) <= max_chars: return [text]
    out, cur = [], ""
    for s in re.split(r"(?<=[.!?])\s+", text):
        if len(s) > max_chars:
            parts = re.split(r"(?<=[;:])\s+", s)
        else:
            parts = [s]
        for part in parts:
            while len(part) > max_chars:
                cut = part.rfind(" ", 0, max_chars)
                cut = cut if cut > 0 else max_chars
                if cur: out.append(cur); cur = ""
                out.append(part[:cut].strip()); part = part[cut:].strip()
            if not part: continue
            if cur and len(cur) + 1 + len(part) > max_chars:
                out.append(cur); cur = part
            else:
                cur = (cur + " " + part).strip()
    if cur: out.append(cur)
    return [p for p in out if p]


def play(path):
    cmd = ["afplay", str(path)] if shutil.which("afplay") else ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(path)]
    subprocess.run(cmd)


def report():
    rows = [json.loads(l) for l in ASSESS.read_text().splitlines()] if ASSESS.exists() else []
    if not rows: print("no assessments yet"); return
    by = {}
    for r in rows: by.setdefault(r["model"], []).append(r)
    print(f"{'model':48} {'n':>3} {'like-me':>7} {'natural':>7} {'words':>6} {'rtf':>5}")
    for m, rs in by.items():
        f = lambda k: sum(r[k] for r in rs) / len(rs)
        print(f"{m[-48:]:48} {len(rs):3d} {f('similarity'):7.2f} {f('naturalness'):7.2f} {f('accuracy'):6.2f} {f('rtf'):5.1f}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("textfile", nargs="?"); ap.add_argument("--no-play", action="store_true")
    ap.add_argument("--language", default="English"); ap.add_argument("--report", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, help="cap codec tokens per paragraph (default: model setting)")
    ap.add_argument("--max-chars", type=int, default=300, help="split paragraphs into chunks of at most this many characters")
    ap.add_argument("--start", type=int, default=1, help="start at this chunk number (resume a partial run)")
    ap.add_argument("--temperature", type=float, help="override sampling temperature (lower avoids repetition loops)")
    ap.add_argument("--repetition-penalty", type=float, help="override repetition penalty")
    args = ap.parse_args()
    if args.report or not args.textfile: report(); return
    sp = Spark(); url = sp.url(); st = state()
    try: h = requests.get(url + "/health", timeout=5).json(); assert h["loaded"]
    except Exception: sys.exit("model server not loaded; run step3_load_model.py first")
    model = f"{h['model']} ({h['mode']})"; name = Path(args.textfile).stem; d = OUT / name; d.mkdir(parents=True, exist_ok=True)
    paras = [p.strip() for p in re.split(r"\n\s*\n", Path(args.textfile).read_text()) if p.strip() and not p.lstrip().startswith("#")]
    paras = [c for p in paras for c in split_for_tts(p, args.max_chars)]
    say(f"{len(paras)} chunks through {model}")
    WORDS.parent.mkdir(parents=True, exist_ok=True); flagged = set(WORDS.read_text().split()) if WORDS.exists() else set()
    for i, p in enumerate(paras, 1):
        if i < args.start: continue
        body = {"text": apply_lexicon(p), "language": args.language}
        body["max_new_tokens"] = args.max_new_tokens or max(256, min(4096, int(len(body["text"]) * 2.0)))
        body["temperature"] = args.temperature if args.temperature is not None else 0.7
        body["repetition_penalty"] = args.repetition_penalty if args.repetition_penalty is not None else 1.2
        r = requests.post(url + "/speak", json=body, timeout=600); r.raise_for_status()
        f = d / f"p{i:02d}.wav"; f.write_bytes(r.content); rtf = float(r.headers.get("X-RTF", 0)); secs = float(r.headers.get("X-Seconds", 0))
        say(f"p{i:02d}: {secs:.1f} s of audio, real-time factor {rtf:.1f}x")
        if args.no_play: continue
        print(f"\n  {p[:200]}{'...' if len(p) > 200 else ''}")
        play(f)
        while True:
            ans = input("  scores like-me/natural/words 1-5 (e.g. 4 4 5), Enter to replay, s skip: ").strip()
            if ans == "": play(f); continue
            if ans.lower() == "s": break
            try: sim, nat, acc = [int(x) for x in ans.split()[:3]]; break
            except ValueError: print("  three numbers, please")
        else: continue
        if ans.lower() == "s": continue
        bad = input("  words it said wrong (space-separated, Enter for none): ").split()
        flagged.update(bad)
        with open(ASSESS, "a") as o:
            o.write(json.dumps({"when": datetime.datetime.now().isoformat(timespec="seconds"), "model": model, "text": name, "para": i,
                                "similarity": sim, "naturalness": nat, "accuracy": acc, "rtf": rtf, "bad_words": bad}) + "\n")
    WORDS.write_text("\n".join(sorted(flagged)) + ("\n" if flagged else ""))
    say(f"audio in {d}; {len(flagged)} flagged words in {WORDS}"); report()


if __name__ == "__main__":
    main()
