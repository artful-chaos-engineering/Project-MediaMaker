#!/usr/bin/env python3
"""Step 3 - Load the models into the Spark's memory and prove they fit.

Loads the BiRefNet matting model onto the GPU and times one 1024-pixel frame; confirms the LatentSync checkpoint
files are present and the right size; reports unified-memory use. LatentSync itself loads per render (its inference
script builds the pipeline each run, about 30 to 60 s), so the proof for it is --smoke: a 3 s tone is lip-synced onto
your avatar with 10 steps. Run --smoke after step 5.

    python step3_load_models.py
    python step3_load_models.py --smoke
"""
import argparse, subprocess, sys
from studio.common import ROOT, Spark, cfg, say, set_state

CHECK = r'''
import os, time, torch
from pathlib import Path
ls = Path("__LS__")
for f, lo in (("checkpoints/latentsync_unet.pt", 3e9), ("checkpoints/whisper/tiny.pt", 5e7)):
    p = ls / f; ok = p.exists() and p.resolve().stat().st_size > lo
    print(("OK  " if ok else "MISSING  ") + str(p), f"{p.resolve().stat().st_size/1e9:.2f} GB" if p.exists() else "")
    if not ok: raise SystemExit(2)
from studio.background import BiRefNet
import numpy as np
t = time.time(); m = BiRefNet("__MAT__", __SIZE__); print(f"BiRefNet loaded in {time.time()-t:.1f} s on {m.dev}")
fr = [np.random.randint(0, 255, (1080, 1920, 3), np.uint8)]
m(fr); torch.cuda.synchronize(); t = time.time(); m(fr); torch.cuda.synchronize(); print(f"one 1080p frame matted in {(time.time()-t)*1000:.0f} ms")
free, total = torch.cuda.mem_get_info(); print(f"memory in use {(total-free)/2**30:.1f} of {total/2**30:.0f} GiB")
'''


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true"); args = ap.parse_args()
    c = cfg(); sp = Spark(); sp.sync_code()
    say("loading the matting model and checking the LatentSync files on the Spark")
    sp.run("PYTHONPATH=. python - <<'PY'\n" + CHECK.replace("__LS__", c["models"]["latentsync_dir"]).replace("__MAT__", c["models"]["matting"]).replace("__SIZE__", str(c["background"]["matte_size"])) + "\nPY")
    set_state(models_loaded_ok=True)
    if args.smoke:
        say("smoke test: 3 s tone lip-synced onto the avatar")
        rc = subprocess.call([sys.executable, str(ROOT / "step7_render_avatar.py"), "--i-am-the-subject", "--tone", "3", "--steps", "10", "--name", "smoke", "--background", "keep"], cwd=ROOT)
        sys.exit(rc)
    say("models fit; run --smoke after step 5 to prove the lip-sync path end to end")


if __name__ == "__main__":
    main()
