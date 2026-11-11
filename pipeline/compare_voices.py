#!/usr/bin/env python3
"""compare_voices.py - clone vs fine-tuned (custom) voice on the same text, with numbers and a blind listening set.

For each mode it loads the model on the Spark's voice server, speaks every paragraph --takes times, and scores each take
(Whisper word error rate, similarity to your reference voice, pace). It prints a side-by-side table and writes
pipeline/compare/<name>/ with the clips and a blind A/B set: files A_*.wav and B_*.wav, with the answer key in key.json
(do not open it until you have chosen).

    python compare_voices.py talktracks/example.txt
    python compare_voices.py talktracks/example.txt --takes 4 --checkpoint checkpoints/run_20261105_1050/checkpoint-epoch-2
Leaves the server holding whichever mode you pass with --leave (default custom if it wins on mean cost, else clone).
"""
import argparse, json, random, statistics, sys
from pathlib import Path
import requests

HERE = Path(__file__).resolve().parent
AVATAR = HERE.parent / "spark_avatar_studio"; VOICE = HERE.parent / "spark_voice_studio"
sys.path.insert(0, str(AVATAR)); sys.path.insert(0, str(HERE))
import talktrack
from studio.common import Spark
from studio.takes import score, speak


def load(url, path, mode):
    requests.post(url + "/unload", timeout=60)
    r = requests.post(url + "/load", json={"path": path, "mode": mode}, timeout=900); r.raise_for_status()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("talktrack"); ap.add_argument("--takes", type=int, default=3); ap.add_argument("--language", default="English")
    ap.add_argument("--checkpoint", help="custom model path on the Spark; default = last_checkpoint in the voice studio's state.json")
    ap.add_argument("--leave", choices=["clone", "custom"]); ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    st, paras, _ = talktrack.parse(a.talktrack); name = st["name"]
    vcfg = json.loads((VOICE / "state.json").read_text())
    paths = {"clone": "models/" + vcfg["tts_model"].replace("/", "__"),
             "custom": a.checkpoint or vcfg.get("last_checkpoint")}
    if not paths["custom"]: sys.exit("no fine-tuned checkpoint recorded; train in Voice Studio step 5 or pass --checkpoint")
    url = Spark().voice_url(); out = HERE / "compare" / name; out.mkdir(parents=True, exist_ok=True)
    res = {}
    for mode in ("clone", "custom"):
        print(f"\n== {mode}: {paths[mode]}", flush=True); load(url, paths[mode], mode); res[mode] = []
        for i, p in enumerate(paras, 1):
            for k in range(1, a.takes + 1):
                wav = speak(url, p, a.language); s = score(url, wav, p, a.language); s.update(paragraph=i, take=k)
                f = out / f"{mode}_p{i:02d}_t{k}.wav"; f.write_bytes(wav); s["file"] = f.name; res[mode].append(s)
                print(f"  p{i:02d} t{k}: wer {s['wer']:.2f}  sim {s['speaker_sim']}  {s['chars_per_sec']} c/s  cost {s['cost']}", flush=True)
    def mean(m, key): 
        v = [x[key] for x in res[m] if x[key] is not None]; return statistics.mean(v) if v else float("nan")
    best = {m: [min((x for x in res[m] if x["paragraph"] == i), key=lambda x: x["cost"]) for i in range(1, len(paras) + 1)] for m in res}
    print(f"\n{'':24}{'clone':>10}{'custom':>10}")
    for label, key in (("mean WER (all takes)", "wer"), ("mean speaker similarity", "speaker_sim"), ("mean chars/second", "chars_per_sec"), ("mean cost (all takes)", "cost")):
        print(f"{label:24}{mean('clone', key):>10.3f}{mean('custom', key):>10.3f}")
    bc = {m: statistics.mean(x["cost"] for x in best[m]) for m in best}
    print(f"{'cost of best take':24}{bc['clone']:>10.3f}{bc['custom']:>10.3f}\nLower WER and cost are better; higher similarity is better.")
    rng = random.Random(a.seed); key = {}
    for i in range(1, len(paras) + 1):
        order = ["clone", "custom"]; rng.shuffle(order)
        for lab, m in zip("AB", order):
            (out / f"{lab}_p{i:02d}.wav").write_bytes((out / best[m][i - 1]["file"]).read_bytes()); key[f"{lab}_p{i:02d}"] = m
    (out / "key.json").write_text(json.dumps(key, indent=1)); (out / "scores.json").write_text(json.dumps(res, indent=1))
    print(f"\nblind set: {out}/A_*.wav vs B_*.wav ; which is the real you? pick before opening key.json")
    win = a.leave or ("custom" if bc["custom"] < bc["clone"] else "clone")
    load(url, paths[win], win); print(f"server now holds the {win} model")


if __name__ == "__main__":
    main()
