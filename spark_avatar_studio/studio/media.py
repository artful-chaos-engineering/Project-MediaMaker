"""media.py - ffmpeg/ffprobe helpers shared by the steps: probing, loop selection, ping-pong footage, audio chunking."""
from __future__ import annotations
import json, subprocess
from pathlib import Path
import numpy as np


def probe(path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)],
                         capture_output=True, text=True, check=True).stdout
    j = json.loads(out); v = next((s for s in j["streams"] if s["codec_type"] == "video"), None)
    a = next((s for s in j["streams"] if s["codec_type"] == "audio"), None)
    info = {"seconds": float(j["format"].get("duration", 0)), "has_audio": a is not None}
    if v:
        n, d = v.get("r_frame_rate", "25/1").split("/"); info.update(width=int(v["width"]), height=int(v["height"]), fps=float(n) / float(d or 1))
    return info


def ff(*args):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *map(str, args)], check=True)


def gray_frames(path, size=64, fps=5) -> np.ndarray:
    """Tiny grayscale frames for loop matching: (n, size, size) float32."""
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(path), "-vf", f"fps={fps},scale={size}:{size},format=gray",
                          "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, size, size).astype(np.float32)


def best_loop(path, min_len=6.0, fps=5):
    """Find (start, end) seconds where the first and last frame look most alike, for a seamless-ish loop."""
    fr = gray_frames(path, fps=fps); n = len(fr)
    if n < min_len * fps * 1.2: return 0.0, n / fps, 0.0
    starts = range(0, max(1, int(n * 0.3))); best = (1e9, 0, n - 1)
    for s in starts:
        for e in range(int(s + min_len * fps), n):
            d = float(np.mean(np.abs(fr[s] - fr[e])))
            if d < best[0]: best = (d, s, e)
    return best[1] / fps, best[2] / fps, best[0]


def pingpong_footage(src, start, end, cursor, need, out, workdir, fps=25):
    """Build `need` seconds of video from src[start:end], playing forward then backward (ping-pong) from `cursor`.
    Returns the new cursor. Output is constant-fps H.264, no audio."""
    workdir = Path(workdir); workdir.mkdir(parents=True, exist_ok=True)
    length = end - start; pos = cursor; fwd = True; parts = []; left = need
    while left > 0.001:
        take = min(left, (length - pos) if fwd else pos)
        if take <= 0.04:
            fwd = not fwd; pos = length if not fwd else 0.0; continue
        seg = workdir / f"seg{len(parts):03d}.mp4"
        if fwd:
            ff("-ss", f"{start + pos:.3f}", "-t", f"{take:.3f}", "-i", src, "-an", "-r", fps, "-c:v", "libx264", "-crf", 14, "-pix_fmt", "yuv420p", seg); pos += take
        else:
            ff("-ss", f"{start + pos - take:.3f}", "-t", f"{take:.3f}", "-i", src, "-an", "-vf", "reverse", "-r", fps, "-c:v", "libx264", "-crf", 14, "-pix_fmt", "yuv420p", seg); pos -= take
        parts.append(seg); left -= take
    lst = workdir / "list.txt"; lst.write_text("".join(f"file '{p}'\n" for p in parts))
    ff("-f", "concat", "-safe", 0, "-i", lst, "-c", "copy", out)
    return pos


def silence_points(wav, noise_db=-35, min_sil=0.35):
    out = subprocess.run(["ffmpeg", "-i", str(wav), "-af", f"silencedetect=noise={noise_db}dB:d={min_sil}", "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    pts = []; start = None
    for line in out.splitlines():
        if "silence_start:" in line: start = float(line.split("silence_start:")[1].split()[0])
        elif "silence_end:" in line and start is not None:
            end = float(line.split("silence_end:")[1].split()[0]); pts.append((start + end) / 2); start = None
    return pts


def chunk_plan(total, cuts, max_len):
    """Split [0,total] at the silence midpoints in `cuts` into pieces no longer than max_len (hard cut if no silence)."""
    bounds = [0.0]; cuts = sorted(c for c in cuts if 0 < c < total)
    while total - bounds[-1] > max_len:
        cand = [c for c in cuts if bounds[-1] + max_len * 0.4 < c <= bounds[-1] + max_len]
        bounds.append(cand[-1] if cand else bounds[-1] + max_len)
    bounds.append(total)
    return list(zip(bounds[:-1], bounds[1:]))
