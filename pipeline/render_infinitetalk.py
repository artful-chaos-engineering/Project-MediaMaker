#!/usr/bin/env python3
"""render_infinitetalk.py - HeyGen-style render: your real footage + new speech -> new video of you saying it.

InfiniteTalk (Wan 2.1 I2V 14B + audio) re-animates the whole face, head and shoulders from a reference clip, so you get natural
movement instead of a still photo with a moving mouth. Uses the 4-step lightx2v distillation LoRA for speed.

    python render_infinitetalk.py --audio ../spark_avatar_studio/data/audio/example.wav --name example --clip s03
Output: ~/avatar-studio/out/<name>/<name>_final.mp4  (copied to spark_avatar_studio/out/<name>/ too)
"""
import argparse, json, shutil, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
INPUT = HERE.parent / "spark_avatar_studio" / "input"
IT = Path.home() / "avatar-studio" / "InfiniteTalk"
PY = Path.home() / "avatar-studio" / ".venv-it" / "bin" / "python"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True); ap.add_argument("--name", required=True)
    ap.add_argument("--clip", default="s01", help="reference clip in spark_avatar_studio/input (s01, s03, s10, s14) or a path")
    ap.add_argument("--steps", type=int, default=4); ap.add_argument("--size", default="infinitetalk-720", choices=["infinitetalk-480", "infinitetalk-720"])
    ap.add_argument("--lora", default="auto", help="face LoRA: auto (newest converted version, if any), none, or a .safetensors path")
    ap.add_argument("--lora-scale", type=float, default=1.0)
    ap.add_argument("--prompt", default="A man is talking naturally to the camera"); a = ap.parse_args()
    clip = Path(a.clip) if Path(a.clip).exists() else INPUT / f"{a.clip}.mp4"
    audio = Path(a.audio).resolve()
    if not clip.exists() or not audio.exists(): sys.exit(f"missing input: {clip if not clip.exists() else audio}")
    work = IT / "test" / a.name; work.mkdir(parents=True, exist_ok=True)
    job = work / "job.json"
    job.write_text(json.dumps({"prompt": a.prompt, "cond_video": str(clip.resolve()), "cond_audio": {"person1": str(audio)}}))
    out = work / "result"
    face, scales = [], []
    if a.lora != "none":
        cand = Path(a.lora) if a.lora != "auto" else next(iter(sorted((Path.home() / "avatar-studio/lora_out").glob("*_v*/face_infinitetalk.safetensors"), key=lambda p: int(p.parent.name.rsplit("_v", 1)[1]), reverse=True)), None)
        if cand and cand.exists(): face, scales = [str(cand)], [str(a.lora_scale)]; print(f"face LoRA: {cand}")
        elif a.lora != "auto": sys.exit(f"LoRA not found: {a.lora}")
        else: print("no face LoRA trained yet (step10_train_face_lora.py); rendering without it")
    cmd = [str(PY), "-u", "generate_infinitetalk.py", "--ckpt_dir", "weights/Wan2.1-I2V-14B-480P", "--wav2vec_dir", "weights/chinese-wav2vec2-base",
           "--infinitetalk_dir", "weights/InfiniteTalk/single/infinitetalk.safetensors",
           "--lora_dir", "weights/lora/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors", *face, "--lora_scale", "1.0", *scales,
           "--input_json", str(job), "--size", a.size, "--sample_text_guide_scale", "1.0", "--sample_audio_guide_scale", "2.0",
           "--sample_steps", str(a.steps), "--sample_shift", "2", "--mode", "streaming", "--motion_frame", "9", "--save_file", str(out)]
    print("$ (cd InfiniteTalk && " + " ".join(cmd) + ")", flush=True)
    if subprocess.run(cmd, cwd=IT).returncode: sys.exit("InfiniteTalk failed; see output above")
    res = Path(str(out) + ".mp4")
    if not res.exists(): sys.exit(f"no video produced at {res}")
    for dest in (HERE.parent / "spark_avatar_studio" / "out" / a.name, Path.home() / "avatar-studio" / "out" / a.name):
        dest.mkdir(parents=True, exist_ok=True)
        for suffix in ("final", "raw"): shutil.copy(res, dest / f"{a.name}_{suffix}.mp4")
    print(f"done: {res}")


if __name__ == "__main__":
    main()
