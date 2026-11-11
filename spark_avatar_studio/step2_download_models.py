#!/usr/bin/env python3
"""Step 2 - Download the models the avatar needs, on the Spark.

  * LatentSync 1.6 lip-sync checkpoint (latentsync_unet.pt, about 5 GB) and its Whisper-tiny audio encoder
  * stabilityai/sd-vae-ft-mse, the VAE LatentSync loads on first run (fetched now so a render never stalls)
  * BiRefNet, the background-removal model (about 0.9 GB)
Everything lands under models/ on the Spark and is linked into the LatentSync checkpoints folder.
Wav2Lip (fast, lower quality) is optional: its weights sit behind a request form, so --wav2lip only prints where to put them.

    python step2_download_models.py
    python step2_download_models.py --only matting
    python step2_download_models.py --wav2lip
"""
import argparse
from studio.common import Spark, cfg, say, set_state

CMDS = {
    "lipsync": r'''
mkdir -p models/latentsync "__LS__/checkpoints"
huggingface-cli download __REPO__ latentsync_unet.pt --local-dir models/latentsync
huggingface-cli download __REPO__ whisper/tiny.pt --local-dir models/latentsync
ln -sf "$PWD/models/latentsync/latentsync_unet.pt" "__LS__/checkpoints/latentsync_unet.pt"
ln -sfn "$PWD/models/latentsync/whisper" "__LS__/checkpoints/whisper"
ls -la "__LS__/checkpoints"
''',
    "vae": "huggingface-cli download __VAE__",
    "matting": "huggingface-cli download __MAT__",
}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--only", choices=list(CMDS)); ap.add_argument("--wav2lip", action="store_true"); args = ap.parse_args()
    c = cfg(); m = c["models"]; sp = Spark()
    sp.sync_code()
    sp.run("df -h . | tail -1", venv=False)
    for name in ([args.only] if args.only else list(CMDS)):
        say(f"downloading: {name}")
        cmd = CMDS[name].replace("__LS__", m["latentsync_dir"]).replace("__REPO__", m["latentsync_repo"]).replace("__VAE__", m["vae"]).replace("__MAT__", m["matting"])
        sp.run(cmd)
    if args.wav2lip:
        say("Wav2Lip: request wav2lip_gan.pth and s3fd.pth from the project page and place them in models/wav2lip/ on the Spark")
    set_state(models_downloaded=True)
    say("models ready; next: step 3 loads them to confirm they fit in memory")


if __name__ == "__main__":
    main()
