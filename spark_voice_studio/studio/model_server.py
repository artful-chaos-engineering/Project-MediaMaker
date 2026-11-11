#!/usr/bin/env python3
"""model_server.py - holds a Qwen3-TTS model in the Spark's memory and speaks on request.

Runs ON the Spark (step 3 starts it). Two modes, switchable at runtime:
  * clone   - the Base model with a voice prompt built from your reference clip (zero-shot, no training)
  * custom  - a fine-tuned checkpoint from step 5, which answers to the speaker name in studio.yaml

Endpoints:
  GET  /health                    -> {"model": ..., "mode": ..., "loaded": true}
  POST /load  {"path": ..., "mode": "clone"|"custom", "ref_audio": ..., "ref_text": ...}
  POST /speak {"text": ..., "language": "English"} -> WAV bytes
  POST /score {"wav_b64": ..., "text": ...} -> {"wer", "heard", "speaker_sim", "seconds", "chars_per_sec"}
  POST /unload
"""
import base64, io, os, re, sys, time
from pathlib import Path
import numpy as np, soundfile as sf, torch, yaml
from fastapi import FastAPI, Response
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
CFG = yaml.safe_load(open(ROOT / "studio.yaml"))
app = FastAPI(title="Spark Voice Studio model server")
STATE = {"model": None, "path": None, "mode": None, "prompt": None, "loaded_at": None}


class LoadReq(BaseModel):
    path: str
    mode: str = "clone"
    ref_audio: str | None = None
    ref_text: str | None = None


class SpeakReq(BaseModel):
    text: str
    language: str = "English"
    max_new_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    repetition_penalty: float | None = None
    do_sample: bool | None = None


def _load(req: LoadReq):
    from qwen_tts import Qwen3TTSModel
    t0 = time.time()
    # bf16 on cuda:0; SDPA attention (no flash-attn wheel for sm_121); memory is unified, so no device_map="auto".
    model = Qwen3TTSModel.from_pretrained(req.path, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")
    prompt = None
    if req.mode == "clone":
        ref = req.ref_audio or str(ROOT / "data/voice/ref.wav")
        wav, sr = sf.read(ref, dtype="float32")
        txt = req.ref_text
        if txt is None and Path(ref).with_suffix(".txt").exists():
            txt = Path(ref).with_suffix(".txt").read_text().strip()
        prompt = model.create_voice_clone_prompt(ref_audio=(wav, sr), ref_text=txt, x_vector_only_mode=txt is None)
    STATE.update(model=model, path=req.path, mode=req.mode, prompt=prompt, loaded_at=time.time())
    return time.time() - t0


@app.get("/health")
def health():
    return {"loaded": STATE["model"] is not None, "model": STATE["path"], "mode": STATE["mode"],
            "gpu_free_gib": round(torch.cuda.mem_get_info()[0] / 2**30, 1) if torch.cuda.is_available() else None}


@app.post("/load")
def load(req: LoadReq):
    secs = _load(req)
    return {"ok": True, "seconds": round(secs, 1), **health()}


class ScoreReq(BaseModel):
    wav_b64: str
    text: str
    language: str = "en"


_AUX = {"asr": None, "spk": None, "ref_emb": None}


def _mono16k(wav, sr):
    import librosa
    if wav.ndim > 1: wav = wav.mean(axis=1)
    return (librosa.resample(wav, orig_sr=sr, target_sr=16000) if sr != 16000 else wav).astype("float32")


def _words(t):
    return re.sub(r"[^a-z0-9' ]+", " ", t.lower().replace("-", " ")).split()


def _wer(ref, hyp):
    r, h = _words(ref), _words(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1])); prev, d[j] = d[j], cur
    return d[-1] / max(len(r), 1)


def _embed(wav16):
    """Speaker embedding from WavLM x-vector; None when that model cannot be loaded (no internet, not cached)."""
    if _AUX["spk"] is None:
        try:
            from transformers import AutoFeatureExtractor, WavLMForXVector
            name = "microsoft/wavlm-base-plus-sv"
            _AUX["spk"] = (AutoFeatureExtractor.from_pretrained(name), WavLMForXVector.from_pretrained(name).to("cuda:0").eval())
        except Exception as e:
            print("speaker model unavailable:", e, file=sys.stderr); _AUX["spk"] = False
    if not _AUX["spk"]: return None
    fe, m = _AUX["spk"]
    with torch.no_grad():
        x = fe(wav16, sampling_rate=16000, return_tensors="pt").to("cuda:0")
        return torch.nn.functional.normalize(m(**x).embeddings, dim=-1)[0].float().cpu()


@app.post("/score")
def score(req: ScoreReq):
    """Objective checks on one clip: Whisper word error rate against the text, and similarity to the reference voice."""
    from transformers import pipeline
    wav, sr = sf.read(io.BytesIO(base64.b64decode(req.wav_b64)), dtype="float32"); w16 = _mono16k(wav, sr)
    if _AUX["asr"] is None:
        _AUX["asr"] = pipeline("automatic-speech-recognition", model=str(ROOT / "models" / CFG["models"]["asr"].replace("/", "__")),
                               torch_dtype=torch.bfloat16, device="cuda:0")
    heard = _AUX["asr"]({"raw": w16, "sampling_rate": 16000}, generate_kwargs={"language": "en" if req.language.lower().startswith("en") else req.language.lower(), "task": "transcribe"})["text"].strip()
    sim = None
    e = _embed(w16)
    if e is not None:
        if _AUX["ref_emb"] is None:
            rw, rs = sf.read(ROOT / "data/voice/ref.wav", dtype="float32"); _AUX["ref_emb"] = _embed(_mono16k(rw, rs))
        sim = float(torch.dot(e, _AUX["ref_emb"]))
    secs = len(w16) / 16000
    return {"wer": round(_wer(req.text, heard), 4), "heard": heard, "speaker_sim": None if sim is None else round(sim, 4),
            "seconds": round(secs, 2), "chars_per_sec": round(len(req.text) / max(secs, 1e-6), 2)}


@app.post("/unload")
def unload():
    STATE.update(model=None, path=None, mode=None, prompt=None); torch.cuda.empty_cache()
    return health()


@app.post("/speak")
def speak(req: SpeakReq):
    m = STATE["model"]
    if m is None:
        return Response(status_code=409, content="no model loaded; POST /load first")
    t0 = time.time()
    gen = {k: v for k, v in {
        "max_new_tokens": req.max_new_tokens, "temperature": req.temperature,
        "top_p": req.top_p, "top_k": req.top_k, "repetition_penalty": req.repetition_penalty,
        "do_sample": req.do_sample,
    }.items() if v is not None}
    if STATE["mode"] == "clone":
        wavs, sr = m.generate_voice_clone(text=req.text, language=req.language, voice_clone_prompt=STATE["prompt"], **gen)
    else:
        wavs, sr = m.generate_custom_voice(text=req.text, language=req.language, speaker=CFG["models"]["speaker_name"], **gen)
    buf = io.BytesIO(); sf.write(buf, np.asarray(wavs[0], dtype="float32"), sr, format="WAV", subtype="PCM_16")
    rtf = (len(wavs[0]) / sr) / max(time.time() - t0, 1e-6)
    return Response(content=buf.getvalue(), media_type="audio/wav", headers={"X-RTF": f"{rtf:.2f}", "X-Seconds": f"{len(wavs[0])/sr:.2f}"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(CFG["server"]["port"]), log_level="warning")
