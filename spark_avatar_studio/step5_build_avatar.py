#!/usr/bin/env python3
"""Step 5 - Build the avatar from your photos or idle video.

Runs on the Spark (the Mac side just starts it). From avatars/<name>/source/ it makes:
  idle_src.mp4   your footage normalised to constant 25 fps, even dimensions, at most 1080 lines
  avatar.json    consent record, mode (video or photo), loop window, size, fps
  bg_preview.png the first second composited on the background you configured, so you can judge the matte before any render
With a video, it finds the loop window: the pair of moments, at least 6 s apart, where your face looks most alike, so the
ping-pong footage under long speech has no visible jump. With photos only, it makes 12 s of still head with a gentle drift.

    python step5_build_avatar.py --i-am-the-subject
    python step5_build_avatar.py --i-am-the-subject --background gradient     # preview another background
    python step5_build_avatar.py --i-am-the-subject --photo photo_front_neutral.png
"""
import argparse, datetime, json, subprocess, sys
from pathlib import Path
from studio.common import ROOT, Spark, cfg, say, require_owner, set_state
from studio.media import best_loop, ff, probe

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}; PHOTO_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def build_here(args):
    c = cfg(); name = c["avatar"]["name"]; home = Path(ROOT) / "avatars" / name; src = home / "source"
    cons = home / "consent.json"
    if not cons.exists() or not json.loads(cons.read_text()).get("subject_is_owner"): sys.exit("refused: no consent.json from step 4")
    vids = sorted(p for p in src.glob("*") if p.suffix.lower() in VIDEO_EXT)
    pics = sorted(p for p in src.glob("*") if p.suffix.lower() in PHOTO_EXT)
    fps = c["render"]["fps"]; out = home / "idle_src.mp4"
    if vids and not args.photo:
        v = max(vids, key=lambda p: probe(p)["seconds"]); mode = "video"
        say(f"normalising {v.name}")
        ff("-i", v, "-an", "-r", fps, "-vf", "scale=-2:'min(1080,ih)'", "-c:v", "libx264", "-crf", 14, "-pix_fmt", "yuv420p", out)
        s, e, d = best_loop(out); say(f"loop window {s:.1f} to {e:.1f} s (frame difference {d:.1f}; under 6 is seamless)")
        loop = [round(s, 2), round(e, 2)]
        if e - s < 6: say("warning: under 6 s of usable footage; long speech will show the ping-pong. Record 30 s or more.")
    else:
        pic = (src / args.photo) if args.photo else next((p for p in pics if "front_neutral" in p.name), pics[0] if pics else None)
        if pic is None: sys.exit(f"no video or photo in {src}")
        mode = "photo"; say(f"photo only: building a 12 s drifting still from {pic.name}")
        # zoompan: slow in-out zoom (1.00 to 1.02) and a few pixels of sway; the head itself stays still.
        from PIL import Image
        pw, ph = Image.open(pic).size; ow = 2 * round(1080 * pw / ph / 2)
        ff("-loop", 1, "-i", pic, "-t", 12, "-vf",
           f"scale=-2:1080,scale=2*trunc(iw/2):2*trunc(ih/2),zoompan=z='1.01+0.01*sin(2*PI*on/{12*fps})':x='iw/2-(iw/zoom/2)+4*sin(2*PI*on/{6*fps})':y='ih/2-(ih/zoom/2)':d=1:s={ow}x1080:fps={fps}",
           "-c:v", "libx264", "-crf", 14, "-pix_fmt", "yuv420p", out)
        loop = [0.0, 12.0]
    i = probe(out)
    (home / "avatar.json").write_text(json.dumps({"owner": c["owner"], "consent": "subject is owner (consent.json)", "mode": mode, "loop": loop, "fps": fps,
        "width": i["width"], "height": i["height"], "seconds": round(i["seconds"], 2), "date": datetime.date.today().isoformat(), "cursor": 0.0}, indent=1))
    # background preview: 1 s of the footage through the configured background
    b = c["background"]; mode_bg = args.background or b["mode"]
    if mode_bg != "keep":
        clip = home / "_preview.mp4"; ff("-i", out, "-t", 1, "-an", "-c:v", "libx264", "-crf", 14, "-pix_fmt", "yuv420p", clip)
        cmd = [sys.executable, "-m", "studio.background", str(clip), str(home / "_preview_bg.mp4"), "--mode", mode_bg, "--engine", b["engine"], "--model", c["models"]["matting"],
               "--size", str(b["matte_size"]), "--color", b["color"], "--color2", b["color2"], "--chroma-key", b["chroma_key"], "--alpha-smooth", str(b["alpha_smooth"]),
               "--choke", str(b["choke"]), "--feather", str(b["feather"])]
        for k in ("image", "video"):
            if b.get(k): cmd += [f"--{k}", str(b[k])]
        subprocess.run(cmd, cwd=ROOT, check=True)
        ff("-i", home / "_preview_bg.mp4", "-vf", "select=eq(n\\,12)", "-frames:v", 1, home / "bg_preview.png")
        say(f"background preview: {home / 'bg_preview.png'}")
    say(f"avatar '{name}' built ({mode} mode, {i['width']}x{i['height']}, {i['seconds']:.0f} s)")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--i-am-the-subject", action="store_true"); ap.add_argument("--here", action="store_true", help="internal: run on this machine")
    ap.add_argument("--photo"); ap.add_argument("--background"); args = ap.parse_args(); require_owner(args)
    if args.here: build_here(args); return
    sp = Spark(); sp.sync_code()
    say("building the avatar on the Spark")
    extra = (f" --photo {args.photo}" if args.photo else "") + (f" --background {args.background}" if args.background else "")
    sp.run(f"PYTHONPATH=. python step5_build_avatar.py --here --i-am-the-subject{extra}")
    name = cfg()["avatar"]["name"]; loc = ROOT / "avatars" / name; loc.mkdir(parents=True, exist_ok=True)
    for f in ("avatar.json", "bg_preview.png"):
        try: sp.get(f"avatars/{name}/{f}", loc / f)
        except Exception: pass
    set_state(avatar_built=name)
    if (loc / "bg_preview.png").exists(): say(f"open {loc / 'bg_preview.png'} to judge the background")


if __name__ == "__main__":
    main()
