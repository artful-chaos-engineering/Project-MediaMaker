#!/usr/bin/env python3
"""Step 10 - Train (and keep retraining) your face LoRA for the InfiniteTalk / Wan 2.1 renderer.

The LoRA teaches the 14B video model what YOU look like and how you move, so renders hold your likeness at 720p. It is
incremental: drop new videos or photos into input/ (or any folder you pass) and run again; existing material is not
re-processed, and each run writes a new numbered version instead of overwriting the last one.

    python step10_train_face_lora.py --i-am-the-subject --all                    # prep + merge + train + convert, in one go
    python step10_train_face_lora.py --i-am-the-subject --all --resume          # continue from the newest version with the new material
    python step10_train_face_lora.py prep    --videos input new_clips --photos photo
    python step10_train_face_lora.py merge                                      # one-time: join the Wan shards into one file
    python step10_train_face_lora.py train   --epochs 20 --rank 32 [--resume]
    python step10_train_face_lora.py convert [--version 3]                      # musubi format -> InfiniteTalk format
    python step10_train_face_lora.py list
Needs the trainer installed once:  pipeline/setup_lora.sh   (musubi-tuner in ~/avatar-studio/.venv-train).

Material: videos are cut into 81-frame, 16 fps clips (about 5 s); photos become image samples. A caption comes from, in order,
a sidecar <name>.txt next to the file, else lora.caption in studio.yaml. Keep footage varied (lighting, head angle, expression,
framing) and the background plain; more variety matters more than more minutes.
"""
import argparse, hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path
from studio.common import ROOT, cfg, require_owner, say

HOME = Path.home() / "avatar-studio"
IT = HOME / "InfiniteTalk"
MUSUBI = HOME / "musubi-tuner"
PY = HOME / ".venv-train" / "bin" / "python"
BASE = IT / "weights" / "Wan2.1-I2V-14B-480P"
MERGED = HOME / "models" / "wan2.1_i2v_480p_14B_bf16.safetensors"
DATA = ROOT / "data" / "lora"
MANIFEST = DATA / "manifest.json"
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}; PHOTO_EXT = {".png", ".jpg", ".jpeg", ".webp"}


def lcfg() -> dict:
    d = {"name": "face", "width": 960, "height": 544, "rank": 32, "alpha": 32, "lr": 2e-4, "epochs": 20, "save_every": 2, "seed": 42,
         "clip_frames": 81, "clip_fps": 16, "block_swap": 0, "caption": "A person talks naturally to the camera, plain background, soft even light."}
    d.update(cfg().get("lora") or {}); return d


def sh(cmd, cwd=None):
    cmd = [str(c) for c in cmd]; say("$ " + " ".join(cmd))
    if subprocess.run(cmd, cwd=cwd).returncode: sys.exit(f"failed: {' '.join(cmd[:4])} ...")


def need_trainer():
    if not PY.exists() or not MUSUBI.exists():
        sys.exit("the LoRA trainer is not installed: run pipeline/setup_lora.sh once")


def sha(p: Path) -> str:
    h = hashlib.sha1(); h.update(f"{p.name}:{p.stat().st_size}".encode()); return h.hexdigest()[:12]


def gather(paths, exts):
    for p in map(Path, paths):
        if p.is_dir(): yield from (f for f in sorted(p.iterdir()) if f.suffix.lower() in exts and not f.name.startswith("."))
        elif p.suffix.lower() in exts and p.exists(): yield p


def caption_for(src: Path, default: str) -> str:
    side = src.with_suffix(".txt")
    return side.read_text().strip() if side.exists() and side.read_text().strip() else default


def prep(videos, photos, c) -> int:
    """Cut new videos into clips and resize new photos into the dataset folders; returns the number of items added."""
    (DATA / "videos").mkdir(parents=True, exist_ok=True); (DATA / "images").mkdir(parents=True, exist_ok=True)
    man = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    w, h, n, fps = c["width"], c["height"], c["clip_frames"], c["clip_fps"]; seg = n / fps; added = 0
    vf = f"fps={fps},scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}"
    for src in gather(videos, VIDEO_EXT):
        key = sha(src)
        if key in man: continue
        dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(src)], capture_output=True, text=True).stdout or 0)
        cap = caption_for(src, c["caption"]); made = []; t = 0.0; k = 0
        while t + seg <= dur + 0.05:
            out = DATA / "videos" / f"{src.stem}_{key}_{k:02d}.mp4"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.3f}", "-t", f"{seg:.3f}", "-i", str(src), "-an", "-vf", vf, "-frames:v", str(n),
                            "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", str(out)], check=True)
            out.with_suffix(".txt").write_text(cap); made.append(out.name); t += seg; k += 1
        man[key] = {"source": str(src), "kind": "video", "items": made}; added += len(made); say(f"{src.name}: {len(made)} clips")
    for src in gather(photos, PHOTO_EXT):
        key = sha(src)
        if key in man: continue
        out = DATA / "images" / f"{src.stem}_{key}.png"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vf", f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}", str(out)], check=True)
        out.with_suffix(".txt").write_text(caption_for(src, c["caption"]))
        man[key] = {"source": str(src), "kind": "photo", "items": [out.name]}; added += 1; say(f"{src.name}: 1 photo")
    MANIFEST.write_text(json.dumps(man, indent=2))
    toml = f'[general]\nresolution = [{w}, {h}]\ncaption_extension = ".txt"\nbatch_size = 1\nenable_bucket = false\n\n'
    if any((DATA / "videos").glob("*.mp4")):
        toml += f'[[datasets]]\nvideo_directory = "{DATA / "videos"}"\ncache_directory = "{DATA / "cache_video"}"\ntarget_frames = [{n}]\nframe_extraction = "head"\nnum_repeats = 1\n\n'
    if any((DATA / "images").glob("*.png")):
        toml += f'[[datasets]]\nimage_directory = "{DATA / "images"}"\ncache_directory = "{DATA / "cache_image"}"\nnum_repeats = 2\n'
    (DATA / "dataset.toml").write_text(toml)
    nv = len(list((DATA / "videos").glob("*.mp4"))); ni = len(list((DATA / "images").glob("*.png")))
    say(f"dataset: {nv} clips, {ni} photos ({added} new)  ->  {DATA / 'dataset.toml'}")
    if nv + ni == 0: sys.exit("no training material found")
    return added


def merge():
    """Musubi-tuner reads one DiT file; the Wan download is 7 shards. Joined once, then reused."""
    if MERGED.exists(): say(f"base already merged: {MERGED}"); return
    from safetensors import safe_open
    from safetensors.torch import save_file
    MERGED.parent.mkdir(parents=True, exist_ok=True); tensors = {}
    for shard in sorted(BASE.glob("diffusion_pytorch_model-*.safetensors")):
        say(f"reading {shard.name}")
        with safe_open(shard, framework="pt") as f:
            for k in f.keys(): tensors[k] = f.get_tensor(k)
    say(f"writing {MERGED} ({len(tensors)} tensors)"); save_file(tensors, str(MERGED) + ".tmp"); os.replace(str(MERGED) + ".tmp", MERGED)


def versions():
    d = HOME / "lora_out"; return sorted(d.glob(f"{lcfg()['name']}_v*"), key=lambda p: int(p.name.rsplit("_v", 1)[1])) if d.exists() else []


def train(c, epochs, rank, resume):
    need_trainer(); merge()
    mu = MUSUBI / "src" / "musubi_tuner"
    sh([PY, mu / "wan_cache_latents.py", "--dataset_config", DATA / "dataset.toml", "--vae", BASE / "Wan2.1_VAE.pth", "--i2v", "--skip_existing",
        "--clip", BASE / "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"], cwd=MUSUBI)
    sh([PY, mu / "wan_cache_text_encoder_outputs.py", "--dataset_config", DATA / "dataset.toml", "--t5", BASE / "models_t5_umt5-xxl-enc-bf16.pth", "--batch_size", 4, "--skip_existing"], cwd=MUSUBI)
    prev = versions(); ver = (int(prev[-1].name.rsplit("_v", 1)[1]) + 1) if prev else 1
    out = HOME / "lora_out" / f"{c['name']}_v{ver}"; out.mkdir(parents=True, exist_ok=True)
    cmd = [PY, "-m", "accelerate.commands.launch", "--num_cpu_threads_per_process", 1, "--mixed_precision", "bf16", mu / "wan_train_network.py",
           "--task", "i2v-14B", "--dit", MERGED, "--clip", BASE / "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
           "--dataset_config", DATA / "dataset.toml", "--sdpa", "--mixed_precision", "bf16", "--fp8_base", "--gradient_checkpointing",
           "--optimizer_type", "adamw8bit", "--learning_rate", c["lr"], "--network_module", "networks.lora_wan", "--network_dim", rank,
           "--network_alpha", c["alpha"], "--timestep_sampling", "shift", "--discrete_flow_shift", 3.0, "--max_data_loader_n_workers", 2,
           "--persistent_data_loader_workers", "--max_train_epochs", epochs, "--save_every_n_epochs", c["save_every"], "--seed", c["seed"],
           "--output_dir", out, "--output_name", c["name"], "--logging_dir", out / "logs"]
    if c["block_swap"]: cmd += ["--blocks_to_swap", c["block_swap"]]
    if resume and prev:
        last = sorted(prev[-1].glob("*.safetensors"), key=os.path.getmtime)
        if last: cmd += ["--network_weights", last[-1]]; say(f"continuing from {last[-1].name}")
    (out / "train.json").write_text(json.dumps({"epochs": epochs, "rank": rank, "resume": bool(resume), "date": time.strftime("%Y-%m-%d %H:%M"), "dataset": json.loads(MANIFEST.read_text())}, indent=1))
    t0 = time.time(); sh(cmd, cwd=MUSUBI); say(f"trained version {ver} in {(time.time() - t0) / 60:.0f} min -> {out}")


def convert(version=None):
    """Musubi writes lora_unet_<module_path> keys with an alpha; InfiniteTalk wants diffusion_model.<path>.lora_down/up with the scale folded in."""
    from safetensors import safe_open
    from safetensors.torch import save_file
    vs = versions()
    if not vs: sys.exit("no trained versions yet: run train")
    d = vs[-1] if version is None else next((v for v in vs if v.name.endswith(f"_v{version}")), None) or sys.exit(f"no version {version}")
    src = sorted(d.glob(f"{lcfg()['name']}*.safetensors"), key=os.path.getmtime)
    src = [s for s in src if not s.name.endswith("_it.safetensors")]
    if not src: sys.exit(f"no checkpoints in {d}")
    src = src[-1]; dst = d / "face_infinitetalk.safetensors"
    with safe_open(MERGED, framework="pt") as f: names = {"lora_unet_" + k[:-7].replace(".", "_"): k[:-7] for k in f.keys() if k.endswith(".weight")}
    out = {}
    with safe_open(src, framework="pt") as f:
        for k in f.keys():
            if not k.endswith(".lora_down.weight"): continue
            mod = k[: -len(".lora_down.weight")]; path = names.get(mod)
            if path is None: sys.exit(f"cannot map LoRA module {mod} onto the Wan model")
            down = f.get_tensor(k); up = f.get_tensor(mod + ".lora_up.weight"); alpha = float(f.get_tensor(mod + ".alpha")) if mod + ".alpha" in f.keys() else down.shape[0]
            out[f"diffusion_model.{path}.lora_down.weight"] = down.contiguous(); out[f"diffusion_model.{path}.lora_up.weight"] = (up * (alpha / down.shape[0])).contiguous()
    save_file(out, str(dst)); say(f"{src.name} -> {dst} ({len(out) // 2} modules)")
    return dst


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", nargs="?", choices=["prep", "merge", "train", "convert", "list"])
    ap.add_argument("--all", action="store_true", help="prep + merge + train + convert")
    ap.add_argument("--i-am-the-subject", action="store_true", help="confirm the footage is of you")
    ap.add_argument("--videos", nargs="*", default=[str(ROOT / "input")]); ap.add_argument("--photos", nargs="*", default=[str(ROOT / "photo")])
    ap.add_argument("--epochs", type=int); ap.add_argument("--rank", type=int); ap.add_argument("--resume", action="store_true"); ap.add_argument("--version", type=int)
    a = ap.parse_args(); c = lcfg()
    if not (a.action or a.all): ap.error("give an action or --all")
    if a.action in (None, "prep", "train") and not a.i_am_the_subject:
        sys.exit("refused: add --i-am-the-subject to confirm the videos and photos are of you")
    if a.i_am_the_subject: a.i_am_the_speaker = True; require_owner(a)
    if a.action == "list" :
        for v in versions(): print(v.name, "(converted)" if (v / "face_infinitetalk.safetensors").exists() else "")
        return
    if a.all or a.action == "prep": prep(a.videos, a.photos, c)
    if a.all or a.action == "merge": merge()
    if a.all or a.action == "train": train(c, a.epochs or c["epochs"], a.rank or c["rank"], a.resume)
    if a.all or a.action == "convert": convert(a.version)


if __name__ == "__main__":
    main()
