#!/usr/bin/env python3
"""Step 5 - Train (fine-tune) the voice model on your recordings.

Runs on the Spark through the connection. Three passes:
  1. transcripts: any utterance without a transcript (imported recordings) is transcribed with Whisper
  2. prepare:     writes data/train.jsonl ({"audio","text","ref_audio"}) and runs Qwen3-TTS finetuning/prepare_data.py,
                  which encodes the audio with the 12 Hz tokenizer
  3. train:       runs finetuning/sft_12hz.py (single-speaker SFT) -> checkpoints/<run>/checkpoint-epoch-N

The Qwen3-TTS scripts take their arguments by name; the names below follow the project's fine-tuning guide.
If the project changes them, `python finetuning/sft_12hz.py --help` on the Spark shows the current ones
and training.extra_args in studio.yaml passes anything through.

    python step5_train_voice.py --i-am-the-speaker                      # full run, epochs from studio.yaml
    python step5_train_voice.py --i-am-the-speaker --run run2 --epochs 5 --init checkpoints/run1/checkpoint-epoch-3
    python step5_train_voice.py --prepare-only
"""
import argparse, datetime, sys
from studio.common import Spark, cfg, say, set_state, require_owner

TRANSCRIBE = r'''
python - <<'PY'
import json, glob, os, soundfile as sf, torch
from transformers import pipeline
asr_dir = "models/" + "{asr}".replace("/", "__")
todo = [f for f in sorted(glob.glob("data/{sub}/utt*.wav")) if not os.path.exists(f[:-4] + ".txt") or not open(f[:-4] + ".txt").read().strip()]
print(len(todo), "utterances need transcripts")
if todo:
    asr = pipeline("automatic-speech-recognition", model=asr_dir, torch_dtype=torch.bfloat16, device="cuda:0")
    for f in todo:
        a, sr = sf.read(f, dtype="float32")
        t = asr({{"raw": a, "sampling_rate": sr}}, generate_kwargs={{"language": "en", "task": "transcribe"}})["text"].strip()
        open(f[:-4] + ".txt", "w").write(t + "\n"); print(" ", os.path.basename(f), t[:70])
# the reference clip is a copy of one utterance; give it that utterance's transcript
src = "data/{sub}/ref.source"
if os.path.exists(src) and not open("data/{sub}/ref.txt").read().strip():
    name = open(src).read().strip()
    txt = open("data/{sub}/" + name[:-4] + ".txt").read().strip() if os.path.exists("data/{sub}/" + name[:-4] + ".txt") else ""
    if txt: open("data/{sub}/ref.txt", "w").write(txt + "\n"); print(" ref.txt <-", name)
PY
'''

PREPARE = r'''
python - <<'PY'
import glob, json, os
rows = []
for sub in {subs}:
    for f in sorted(glob.glob(f"data/{{sub}}/utt*.wav")):
        t = open(f[:-4] + ".txt").read().strip() if os.path.exists(f[:-4] + ".txt") else ""
        if t: rows.append({{"audio": os.path.abspath(f), "text": t, "ref_audio": os.path.abspath("data/voice/ref.wav")}})
with open("data/train.jsonl", "w") as o:
    for r in rows: o.write(json.dumps(r, ensure_ascii=False) + "\n")
print(len(rows), "training rows -> data/train.jsonl")
PY
cd {qr}/finetuning && python prepare_data.py --input_jsonl {home}/data/train.jsonl --output_jsonl {home}/data/train_codes.jsonl \
    --tokenizer_model_path {home}/models/{tok} --device cuda:0
'''

TRAIN = r'''
set -o pipefail
python -c "import flash_attn" 2>/dev/null || sed -i 's/attn_implementation="flash_attention_2"/attn_implementation="sdpa"/' {qr}/finetuning/sft_12hz.py
mkdir -p checkpoints/{run}
cd {qr}/finetuning && python sft_12hz.py --init_model_path {init} --output_model_path {home}/checkpoints/{run} \
    --train_jsonl {home}/data/train_codes.jsonl --batch_size {bs} --lr {lr} --num_epochs {epochs} --speaker_name {spk} {extra} 2>&1 | tee {home}/out/train_{run}.log
ls -d {home}/checkpoints/{run}/checkpoint-epoch-* | tail -1
'''


def main():
    c = cfg(); t = c["training"]; ap = argparse.ArgumentParser()
    ap.add_argument("--i-am-the-speaker", action="store_true"); ap.add_argument("--run", default=datetime.datetime.now().strftime("run_%Y%m%d_%H%M"))
    ap.add_argument("--epochs", type=int, default=t["epochs"]); ap.add_argument("--batch-size", type=int, default=t["batch_size"]); ap.add_argument("--lr", type=float, default=t["lr"])
    ap.add_argument("--init", help="model or checkpoint to start from; default the downloaded Base model")
    ap.add_argument("--include-keywords", action="store_true", help="also train on data/keywords (step 7)")
    ap.add_argument("--prepare-only", action="store_true"); args = ap.parse_args()
    if not args.prepare_only: require_owner(args)
    sp = Spark(); home = sp.home; qr = t["qwen_repo"]; subs = ["voice"] + (["keywords"] if args.include_keywords else [])
    say("pass 1: transcripts for recordings that have none")
    for sub in subs: sp.run(TRANSCRIBE.format(asr=c["models"]["asr"], sub=sub))
    say("pass 2: training JSONL and 12 Hz audio codes")
    sp.run(PREPARE.format(subs=subs, qr=qr, home=home, tok=c["models"]["tokenizer"].replace("/", "__")))
    if args.prepare_only: return
    init = args.init or f"{home}/models/" + c["models"]["tts"].replace("/", "__")
    say(f"pass 3: fine-tuning {init} for {args.epochs} epochs (run {args.run}); watch out/train_{args.run}.log on the Spark")
    out = sp.run(TRAIN.format(run=args.run, qr=qr, home=home, init=init, bs=args.batch_size, lr=args.lr, epochs=args.epochs,
                              spk=c["models"]["speaker_name"], extra=t.get("extra_args", "")))
    ck = out.strip().splitlines()[-1].strip()
    rel = ck.replace(home + "/", "")
    set_state(last_checkpoint=rel, last_run=args.run)
    say(f"done: {rel}. Load it with: python step3_load_model.py --mode custom")


if __name__ == "__main__":
    main()
