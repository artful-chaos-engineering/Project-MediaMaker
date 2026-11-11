#!/usr/bin/env python3
"""Step 7 - Render the talking avatar: audio in, lip-synced video out, optional new background.

Mac side: uploads the audio, runs the render on the Spark, downloads the results to out/<name>/.
Spark side (--here): cuts the audio at silences into chunks of at most render.max_chunk_seconds, builds ping-pong footage
for each chunk from your avatar's loop window (the footage cursor carries on from chunk to chunk), runs LatentSync 1.6 on
each, joins the chunks, lays your original audio back on (full quality, unshifted), then runs the background stage.
Chunks are cached by hash: re-run after an interruption or a setting change and only the affected chunks redo.

Outputs in out/<name>/:   <name>_raw.mp4 (lip-synced, original background) and <name>_final.mp4 (after the background stage).
Because the raw file is kept, step 8 can change the background in minutes without repeating the lip-sync.

    python step7_render_avatar.py --i-am-the-subject --audio data/audio/lesson.wav
    python step7_render_avatar.py --i-am-the-subject --audio lesson.wav --background gradient --steps 30
    python step7_render_avatar.py --i-am-the-subject --tone 3 --steps 10 --name smoke     # smoke test, no audio needed
"""
import argparse, hashlib, json, shutil, subprocess, sys, tempfile
from pathlib import Path
from studio.common import ROOT, Spark, cfg, say, require_owner, set_state, state
from studio.media import chunk_plan, ff, pingpong_footage, probe, silence_points

DISCLOSURE = "Synthetic video: lip-synced footage of the owner driven by a cloned voice."


def bg_cmd(c, args, src, out):
    b = c["background"]; mode = args.background or b["mode"]
    cmd = [sys.executable, "-m", "studio.background", str(src), str(out), "--mode", mode, "--engine", args.engine or b["engine"], "--model", c["models"]["matting"],
           "--size", str(b["matte_size"]), "--color", args.color or b["color"], "--color2", b["color2"], "--chroma-key", b["chroma_key"],
           "--alpha-smooth", str(b["alpha_smooth"]), "--choke", str(b["choke"]), "--feather", str(b["feather"])]
    for k, v in (("image", args.bg_image or b.get("image")), ("video", args.bg_video or b.get("video"))):
        if v: cmd += [f"--{k}", str(v)]
    return cmd, mode


def render_here(args):
    c = cfg(); r = c["render"]; name = args.name; av = Path(ROOT) / "avatars" / c["avatar"]["name"]
    meta = json.loads((av / "avatar.json").read_text())
    if "owner" not in meta.get("consent", ""): sys.exit("refused: avatar.json carries no owner consent")
    steps, guid, seed, off = args.steps or r["steps"], args.guidance or r["guidance"], r["seed"], args.offset if args.offset is not None else r["audio_offset"]
    out = Path(ROOT) / "out" / name; work = out / "work"; work.mkdir(parents=True, exist_ok=True)
    wav = Path(args.audio_path); total = probe(wav)["seconds"]
    plan = chunk_plan(total, silence_points(wav), r["max_chunk_seconds"]); say(f"{total:.1f} s of audio in {len(plan)} chunk(s)")
    idle = av / "idle_src.mp4"; ls = Path(c["models"]["latentsync_dir"])
    cache = work / "chunks.json"; done = json.loads(cache.read_text()) if cache.exists() else {}
    cursor = done.setdefault("start_cursor", meta.get("cursor", 0.0))      # fixed per render name, so a re-run hits the chunk cache
    parts = []
    for i, (a, b) in enumerate(plan):
        d = b - a; tmp = work / f"c{i:03d}"; tmp.mkdir(exist_ok=True)
        ff("-i", wav, "-ss", f"{a:.3f}", "-t", f"{d:.3f}", "-af", f"adelay={int(max(off,0)*1000)}|{int(max(off,0)*1000)}" if off > 0 else "anull", "-ac", 1, "-ar", 16000, tmp / "a.wav")
        key = hashlib.sha256(f"{tmp.joinpath('a.wav').read_bytes()[:200000].hex()}|{d}|{steps}|{guid}|{seed}|{meta['date']}|{meta['loop']}|{cursor:.2f}".encode()).hexdigest()[:16]
        cv = work / f"c{i:03d}_ls.mp4"
        if done.get(str(i)) == key and cv.exists(): say(f"chunk {i+1}/{len(plan)}: cached")
        else:
            say(f"chunk {i+1}/{len(plan)}: {d:.1f} s, footage from {cursor:.1f} s")
            pingpong_footage(idle, meta["loop"][0], meta["loop"][1], cursor, d, tmp / "v.mp4", tmp / "segs", fps=r["fps"])
            subprocess.run([sys.executable, "-m", "scripts.inference", "--unet_config_path", "configs/unet/stage2_512.yaml", "--inference_ckpt_path", "checkpoints/latentsync_unet.pt",
                            "--inference_steps", str(steps), "--guidance_scale", str(guid), "--seed", str(seed),
                            "--video_path", str(tmp / "v.mp4"), "--audio_path", str(tmp / "a.wav"), "--video_out_path", str(cv)], cwd=ls, check=True)
            done[str(i)] = key; cache.write_text(json.dumps(done))
        # advance the cursor exactly as the footage builder would have (ping-pong over the loop window)
        L = meta["loop"][1] - meta["loop"][0]; cyc = (cursor + d) % (2 * L); cursor = cyc if cyc <= L else 2 * L - cyc
        parts.append(cv)
    lst = work / "join.txt"; lst.write_text("".join(f"file '{p.resolve()}'\n" for p in parts))
    joined = work / "joined.mp4"; ff("-f", "concat", "-safe", 0, "-i", lst, "-an", "-c:v", "libx264", "-crf", 14, "-pix_fmt", "yuv420p", "-r", r["fps"], joined)
    raw = out / f"{name}_raw.mp4"
    ff("-i", joined, "-i", wav, "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", "-metadata", f"comment={DISCLOSURE}", raw)
    cmd, mode = bg_cmd(c, args, raw, out / f"{name}_final.mp4")
    if mode == "keep": shutil.copy(raw, out / f"{name}_final.mp4")
    else:
        say(f"background stage: {mode}"); subprocess.run(cmd, cwd=ROOT, check=True)
    meta["cursor"] = round(cursor, 3); (av / "avatar.json").write_text(json.dumps(meta, indent=1))   # next render starts where this one ended
    (out / "manifest.json").write_text(json.dumps({"name": name, "seconds": round(total, 2), "chunks": len(plan), "steps": steps, "guidance": guid, "offset": off,
        "background": mode, "avatar": c["avatar"]["name"], "disclosure": DISCLOSURE}, indent=1))
    say(f"done: {out / (name + '_final.mp4')}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--i-am-the-subject", action="store_true"); ap.add_argument("--here", action="store_true")
    ap.add_argument("--audio"); ap.add_argument("--audio-path", help="internal"); ap.add_argument("--tone", type=float); ap.add_argument("--name")
    ap.add_argument("--steps", type=int); ap.add_argument("--guidance", type=float); ap.add_argument("--offset", type=float)
    ap.add_argument("--background"); ap.add_argument("--engine"); ap.add_argument("--color"); ap.add_argument("--bg-image"); ap.add_argument("--bg-video")
    args = ap.parse_args(); require_owner(args)
    if args.here: render_here(args); return
    c = cfg(); sp = Spark(); name = args.name or (Path(args.audio).stem if args.audio else "smoke")
    tmpd = Path(tempfile.mkdtemp())
    if args.tone:                                        # syllable-like test signal: 220 Hz tone, 4 Hz tremolo
        src = tmpd / f"{name}.wav"; subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"sine=f=220:d={args.tone}", "-af", "tremolo=f=4:d=0.9,volume=0.8", "-ar", "24000", str(src)], check=True)
    elif args.audio: src = Path(args.audio)
    else: sys.exit("give --audio FILE (step 6 writes data/audio/<name>.wav) or --tone SECONDS")
    sp.sync_code(); a = c["avatar"]["name"]
    sp.put(src, f"avatars/{a}/in/{name}.wav")
    tuned = state().get("tuned", {})                      # step 9 --apply stores the settings you scored best
    for k in ("steps", "guidance", "offset"):
        if getattr(args, k) is None and k in tuned: setattr(args, k, tuned[k])
    for k, ck in (("bg_image", "image"), ("bg_video", "video")):      # backdrops live on the Mac; send them along
        v = getattr(args, k) or (c["background"].get(ck) if (args.background or c["background"]["mode"]) == ck else None)
        if v: sp.put(Path(v), f"backdrops/{Path(v).name}"); setattr(args, k, f"backdrops/{Path(v).name}")
    flags = [f"--{k.replace('_','-')} {v}" for k, v in vars(args).items() if k in ("steps", "guidance", "offset", "background", "engine", "color", "bg_image", "bg_video") and v is not None]
    say("rendering on the Spark (LatentSync loads for each run; expect minutes per minute of audio)")
    sp.run(f"PYTHONPATH=. python step7_render_avatar.py --here --i-am-the-subject --name {name} --audio-path avatars/{a}/in/{name}.wav " + " ".join(flags))
    dst = ROOT / "out" / name; dst.mkdir(parents=True, exist_ok=True)
    for f in (f"{name}_raw.mp4", f"{name}_final.mp4", "manifest.json"): sp.get(f"out/{name}/{f}", dst / f)
    set_state(last_render=name); say(f"results in {dst}")


if __name__ == "__main__":
    main()
