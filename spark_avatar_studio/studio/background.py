#!/usr/bin/env python3
"""background.py - remove or replace the background of a video.

Runs on the Spark (GPU) for engine birefnet; engine chroma needs no model and no GPU (green or blue screen footage).
    python -m studio.background IN.mp4 OUT.mp4 --mode color --color "#0B1F3A"
Modes: keep (copy), transparent (ProRes 4444 .mov, or VP9 alpha .webm), color, gradient, image, video, blur.
The audio of IN is carried over untouched.
"""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path
import cv2, numpy as np
from studio.media import probe


def hex_rgb(h):
    h = h.lstrip("#"); return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


# -- matte engines -----------------------------------------------------------------------------
class BiRefNet:
    def __init__(self, repo, size=1024):
        import torch
        from transformers import AutoModelForImageSegmentation
        self.torch, self.size = torch, size
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.m = AutoModelForImageSegmentation.from_pretrained(repo, trust_remote_code=True).to(self.dev).eval()
        if self.dev == "cuda": self.m = self.m.half()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.dev).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.dev).view(1, 3, 1, 1)

    def __call__(self, frames):                      # list of HxWx3 uint8 RGB -> list of HxW float32 alpha
        t = self.torch
        h, w = frames[0].shape[:2]
        x = np.stack([cv2.resize(f, (self.size, self.size), interpolation=cv2.INTER_AREA) for f in frames])
        x = t.from_numpy(x).to(self.dev).permute(0, 3, 1, 2).float() / 255.0
        x = ((x - self.mean) / self.std).to(next(self.m.parameters()).dtype)
        with t.no_grad():
            p = self.m(x)[-1].sigmoid().float().cpu().numpy()[:, 0]
        return [cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR) for m in p]


class Chroma:
    """Key a green/blue screen: alpha falls off with colour distance from the key in HSV hue/saturation."""
    def __init__(self, key, tol=0.28):
        r, g, b = hex_rgb(key); self.hue = cv2.cvtColor(np.uint8([[[r, g, b]]]), cv2.COLOR_RGB2HSV)[0, 0, 0]; self.tol = tol

    def __call__(self, frames):
        out = []
        for f in frames:
            hsv = cv2.cvtColor(f, cv2.COLOR_RGB2HSV).astype(np.float32)
            dh = np.minimum(np.abs(hsv[..., 0] - self.hue), 180 - np.abs(hsv[..., 0] - self.hue)) / 90.0
            sat = hsv[..., 1] / 255.0
            dist = dh + (1 - sat) * 0.6                     # grey/white pixels are not screen
            out.append(np.clip((dist - self.tol * 0.6) / (self.tol * 0.8), 0, 1).astype(np.float32))
        return out


def despill(f, key):
    """Pull the screen colour's reflection off the subject: clamp the key channel to the mean of the other two."""
    r, g, b = hex_rgb(key); ch = 1 if g >= max(r, b) else 2 if b >= max(r, g) else 0
    o = [i for i in range(3) if i != ch]; x = f.astype(np.float32)
    x[..., ch] = np.minimum(x[..., ch], (x[..., o[0]] + x[..., o[1]]) / 2)
    return x.astype(np.uint8)


# -- background sources --------------------------------------------------------------------------
def cover(img, w, h):
    ih, iw = img.shape[:2]; s = max(w / iw, h / ih)
    r = cv2.resize(img, (int(iw * s + 0.5), int(ih * s + 0.5)), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    y, x = (r.shape[0] - h) // 2, (r.shape[1] - w) // 2
    return r[y:y + h, x:x + w]


def reader(path, w, h, loop=False):
    cmd = ["ffmpeg", "-loglevel", "error"] + (["-stream_loop", "-1"] if loop else []) + ["-i", str(path), "-vf",
           f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},format=rgb24", "-f", "rawvideo", "-"]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE)


def read_n(p, n, w, h):
    buf = p.stdout.read(w * h * 3 * n)
    k = len(buf) // (w * h * 3)
    return np.frombuffer(buf[:k * w * h * 3], np.uint8).reshape(k, h, w, 3)


def make_background(a, w, h):
    mode = a.mode
    if mode == "color": return lambda frames: [np.full((h, w, 3), hex_rgb(a.color), np.uint8)] * len(frames)
    if mode == "gradient":
        c1, c2 = np.array(hex_rgb(a.color), np.float32), np.array(hex_rgb(a.color2), np.float32)
        t = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
        g = np.broadcast_to((c1 * (1 - t) + c2 * t).astype(np.uint8), (h, w, 3)).copy()
        return lambda frames: [g] * len(frames)
    if mode == "image":
        img = cv2.cvtColor(cv2.imread(a.image), cv2.COLOR_BGR2RGB); img = cover(img, w, h)
        return lambda frames: [img] * len(frames)
    if mode == "video":
        p = reader(a.video, w, h, loop=True)
        return lambda frames: list(read_n(p, len(frames), w, h))
    if mode == "blur":
        return lambda frames: [(cv2.GaussianBlur(f, (0, 0), 28) * 0.8).astype(np.uint8) for f in frames]
    raise SystemExit(f"unknown background mode {mode}")


# -- main pass ------------------------------------------------------------------------------------
def run(a):
    src = Path(a.src); info = probe(src); w, h, fps = info["width"], info["height"], info["fps"]
    out = Path(a.out)
    if a.mode == "keep":
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-c", "copy", str(out)], check=True); return
    transparent = a.mode == "transparent"
    engine = Chroma(a.chroma_key) if a.engine == "chroma" else BiRefNet(a.model, a.size)
    bg = None if transparent else make_background(a, w, h)
    rd = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-i", str(src), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    pix_in = "rgba" if transparent else "rgb24"
    if transparent and out.suffix == ".webm": venc = ["-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p", "-b:v", "0", "-crf", "20", "-auto-alt-ref", "0"]
    elif transparent: venc = ["-c:v", "prores_ks", "-profile:v", "4444", "-pix_fmt", "yuva444p10le"]
    else: venc = ["-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", "-preset", "medium"]
    aenc = ["-c:a", "pcm_s16le"] if (transparent and out.suffix == ".mov") else (["-c:a", "libopus"] if out.suffix == ".webm" else ["-c:a", "aac", "-b:a", "192k"])
    wr = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", pix_in, "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-",
                           "-i", str(src), "-map", "0:v", "-map", "1:a?", *venc, *aenc, "-shortest", str(out)], stdin=subprocess.PIPE)
    prev = None; done = 0; B = a.batch
    smooth = 0 if a.engine == "chroma" else a.alpha_smooth      # a keyed screen needs no temporal smoothing
    k = 3 if a.feather else 0
    while True:
        buf = rd.stdout.read(w * h * 3 * B)
        n = len(buf) // (w * h * 3)
        if n == 0: break
        frames = list(np.frombuffer(buf[:n * w * h * 3], np.uint8).reshape(n, h, w, 3))
        alphas = engine(frames)
        if a.engine == "chroma": frames = [despill(f, a.chroma_key) for f in frames]
        res_alpha = []
        for al in alphas:
            if smooth and prev is not None: al = smooth * prev + (1 - smooth) * al
            prev = al
            x = np.clip((al - a.choke) / (1 - 2 * a.choke), 0, 1) if a.choke else al      # choke: pull the edge in
            if a.feather: x = cv2.GaussianBlur(x, (0, 0), a.feather)
            res_alpha.append(x)
        if transparent:
            for f, x in zip(frames, res_alpha): wr.stdin.write(np.dstack([f, (x * 255).astype(np.uint8)]).tobytes())
        else:
            bgs = bg(frames)
            for f, x, b in zip(frames, res_alpha, bgs):
                x3 = x[..., None]; wr.stdin.write((f * x3 + b * (1 - x3)).astype(np.uint8).tobytes())
        done += n
        if done % (B * 10) < B: print(f"  background: {done} frames", flush=True)
    wr.stdin.close(); wr.wait(); rd.wait()
    if a.mode == "video": bg.__closure__ and [c.cell_contents.kill() for c in bg.__closure__ if isinstance(c.cell_contents, subprocess.Popen)]
    print(f"background: {done} frames -> {out}")


def add_args(ap):
    ap.add_argument("--mode", default="color", choices=["keep", "transparent", "color", "gradient", "image", "video", "blur"])
    ap.add_argument("--engine", default="birefnet", choices=["birefnet", "chroma"])
    ap.add_argument("--model", default="ZhengPeng7/BiRefNet"); ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--color", default="#0B1F3A"); ap.add_argument("--color2", default="#1D6FA5")
    ap.add_argument("--image"); ap.add_argument("--video"); ap.add_argument("--chroma-key", default="#00B140")
    ap.add_argument("--alpha-smooth", type=float, default=0.5); ap.add_argument("--choke", type=float, default=0.08)
    ap.add_argument("--feather", type=float, default=2); ap.add_argument("--batch", type=int, default=4)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("out"); add_args(ap); run(ap.parse_args())
