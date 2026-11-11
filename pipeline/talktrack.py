#!/usr/bin/env python3
"""talktrack.py - read, validate and clean a talktrack script.

A talktrack is a plain text file:
    # note                  ignored (never spoken)
    @key: value             settings for this run: name, language, pause, background, voice_mode, steps, guidance, takes
    (blank-line separated)  paragraphs of speech; each becomes one clip and one caption

    python talktrack.py talktracks/example.txt            # validate and print a summary
    python talktrack.py talktracks/example.txt -o clean.txt
"""
import argparse, re, sys
from pathlib import Path

WPM = 150                      # speaking-rate estimate used for duration
MAX_PARA_CHARS = 700           # long paragraphs make slow, drifting TTS clips
KNOWN = {"name", "language", "pause", "background", "voice_mode", "steps", "guidance", "takes"}


def parse(path):
    """Return (settings dict, [paragraphs], [warnings]). Raises ValueError when unusable."""
    text = Path(path).read_text(encoding="utf-8-sig")
    settings, warns, paras = {}, [], []
    for blk in re.split(r"\n\s*\n", text.replace("\r\n", "\n")):
        speech = []
        for line in blk.split("\n"):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            m = re.match(r"@(\w+)\s*:\s*(.*)$", s)
            if m:
                k = m.group(1).lower()
                if k not in KNOWN: warns.append(f"unknown setting @{k} ignored")
                else: settings[k] = m.group(2).strip()
                continue
            speech.append(s)
        if speech: paras.append(" ".join(speech))
    if not paras: raise ValueError("the talktrack has no speech")
    settings.setdefault("name", re.sub(r"\W+", "_", Path(path).stem).strip("_").lower() or "talktrack")
    for i, p in enumerate(paras, 1):
        if len(p) > MAX_PARA_CHARS: warns.append(f"paragraph {i} is {len(p)} characters; split it for a steadier voice")
        if re.search(r"[<>{}\[\]|\\]|https?://", p): warns.append(f"paragraph {i} has symbols or a URL the voice will read oddly; spell them out")
        if re.search(r"\d{4,}", p): warns.append(f"paragraph {i} has a long number; write it as words")
        if not re.search(r"[.!?…:]$", p): warns.append(f"paragraph {i} does not end in punctuation; the voice may trail off")
    return settings, paras, warns


def seconds(paras, pause=0.5):
    words = sum(len(p.split()) for p in paras)
    return words / WPM * 60 + pause * (len(paras) - 1)


def write_clean(paras, out):
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n\n".join(paras) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("file"); ap.add_argument("-o", "--out"); a = ap.parse_args()
    try: st, paras, warns = parse(a.file)
    except (OSError, ValueError) as e: sys.exit(f"talktrack error: {e}")
    est = seconds(paras, float(st.get("pause", 0.5)))
    print(f"{a.file}: name={st['name']}  {len(paras)} paragraphs  {sum(len(p.split()) for p in paras)} words  about {est:.0f} s of speech")
    for k, v in st.items(): print(f"  @{k}: {v}")
    for w in warns: print(f"  warning: {w}")
    if a.out: write_clean(paras, a.out); print(f"clean text -> {a.out}")


if __name__ == "__main__":
    main()
