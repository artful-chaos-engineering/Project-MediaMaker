"""takes.py - speak a paragraph several times and keep the best take.

The voice model samples, so every take differs. We score each with the voice server's /score (Whisper word error rate,
similarity to the reference voice, speaking pace) and keep the lowest-cost one. Lower cost is better.
"""
import base64
import requests

PACE = (9.0, 20.0)      # plausible characters per second of speech; outside this a clip is truncated, rushed or dragging


def cost(s):
    c = s["wer"] * 3.0                                   # misread words matter most
    if s.get("speaker_sim") is not None: c += (1.0 - s["speaker_sim"]) * 1.5
    cps = s["chars_per_sec"]
    if cps < PACE[0]: c += (PACE[0] - cps) / PACE[0]
    if cps > PACE[1]: c += (cps - PACE[1]) / PACE[1]
    return round(c, 4)


def score(url, wav_bytes, text, language="English"):
    r = requests.post(url + "/score", json={"wav_b64": base64.b64encode(wav_bytes).decode(), "text": text, "language": language}, timeout=300)
    r.raise_for_status(); s = r.json(); s["cost"] = cost(s); return s


def speak(url, text, language="English", **gen):
    r = requests.post(url + "/speak", json={"text": text, "language": language, **gen}, timeout=600); r.raise_for_status(); return r.content


def best_of(url, text, n, language="English"):
    """Return (wav_bytes, chosen_index, [score dict per take], [wav bytes per take]). With n == 1 or no /score endpoint, returns the first take."""
    clips, scores = [], []
    for k in range(max(n, 1)):
        wav = speak(url, text, language); clips.append(wav)
        if n <= 1: return wav, 0, [], clips
        try: scores.append(score(url, wav, text, language))
        except requests.RequestException as e:
            print(f"  scoring unavailable ({e}); using the first take"); return clips[0], 0, [], clips
    best = min(range(len(clips)), key=lambda i: scores[i]["cost"])
    return clips[best], best, scores, clips
