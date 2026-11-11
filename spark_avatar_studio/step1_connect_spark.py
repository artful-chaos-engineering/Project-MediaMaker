#!/usr/bin/env python3
"""Step 1 - Connect to the DGX Spark and prepare the avatar environment.

Checks SSH and the GPU, creates the studio venv on the Spark with the cu130 PyTorch wheels, clones LatentSync
(Apache 2.0) with its torch pins stripped so it cannot replace the Spark's PyTorch, installs the packages
(onnxruntime CPU, face-alignment, transformers for the matting model), syncs this folder, and records the
result in state.json.

    python step1_connect_spark.py            # connect, check, install what is missing
    python step1_connect_spark.py --check    # connect and report only
"""
import argparse, sys
from studio.common import Spark, cfg, say, set_state

SETUP = r'''
set -e
sudo -n apt-get install -y -qq ffmpeg git git-lfs libgl1 libglib2.0-0 >/dev/null 2>&1 || true
command -v python3.12 >/dev/null || sudo -n apt-get install -y -qq python3.12 python3.12-venv >/dev/null
[ -d .venv ] || python3.12 -m venv .venv
source .venv/bin/activate
pip install -q --upgrade pip wheel
if ! python -c 'import importlib.util,sys; sys.exit(0 if importlib.util.find_spec("torch") else 1)'; then
  echo "== installing PyTorch (CUDA 13 aarch64 wheels)"
  pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
fi
pip install -q numpy pillow pyyaml requests soundfile librosa imageio imageio-ffmpeg huggingface_hub "transformers" accelerate kornia timm einops opencv-python-headless
echo "== LatentSync"
if [ ! -d "__LS__" ]; then git clone -q --depth 1 __GIT__ "__LS__"; fi
sed -i 's/^mediapipe.*/mediapipe/; /^decord/d; /^eva-decord/d; /^torch[>=< ]/d; /^torch$/d; /^torchvision/d; /^torchaudio/d; /^onnxruntime-gpu/d; /^xformers/d; /^opencv-python[>=< ]/d; /^opencv-python$/d' "__LS__/requirements.txt"
# decord has no aarch64 wheel; LatentSync inference only needs AudioReader (and VideoReader for one optional path)
cat > "__LS__/latentsync/utils/decord_compat.py" <<'COMPAT'
import subprocess
import numpy as np


class _Arr:
    def __init__(self, a): self.a = a
    def asnumpy(self): return self.a


class AudioReader:
    def __init__(self, path, sample_rate=16000, mono=True):
        cmd = ["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1" if mono else "2", "-ar", str(sample_rate), "-f", "f32le", "-"]
        raw = subprocess.run(cmd, check=True, stdout=subprocess.PIPE).stdout
        a = np.frombuffer(raw, dtype=np.float32)
        self.a = a.reshape(1, -1) if mono else a.reshape(-1, 2).T.copy()

    def __getitem__(self, idx): return _Arr(self.a[:, idx] if isinstance(idx, slice) else self.a)


class VideoReader:
    def __init__(self, path):
        import cv2
        cap = cv2.VideoCapture(path); fr = []
        while True:
            ok, f = cap.read()
            if not ok: break
            fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        cap.release(); self.fr = np.stack(fr)

    def __len__(self): return len(self.fr)
    def __getitem__(self, idx): return _Arr(self.fr[idx])
    def seek(self, i): pass
COMPAT
sed -i 's/^from decord import AudioReader, VideoReader/from .decord_compat import AudioReader, VideoReader/' "__LS__/latentsync/utils/util.py"
pip install -q -r "__LS__/requirements.txt"
pip install -q onnxruntime                 # the -gpu package has no aarch64 wheel
pip install -q mediapipe 2>/dev/null || pip install -q face-alignment
mkdir -p models data avatars out checkpoints
echo "== environment ready"
'''

GPU_CHECK = r'''
python - <<'PY'
import torch
ok = torch.cuda.is_available()
print("torch", torch.__version__, "cuda", torch.version.cuda, "device", torch.cuda.get_device_name(0) if ok else "NONE")
if ok:
    print("capability", torch.cuda.get_device_capability(0), "arch", [a for a in torch.cuda.get_arch_list() if "120" in a])
    free, total = torch.cuda.mem_get_info(); print(f"memory free {free/2**30:.0f} of {total/2**30:.0f} GiB (unified)")
    import torchaudio.compliance.kaldi as k
    try:
        k.fbank(torch.randn(1, 16000, device="cuda"), num_mel_bins=80); print("fbank on GPU: OK")
    except Exception as e:
        print("fbank on GPU FAILED:", str(e)[:120]); print("-> NVRTC fix needed: see the voice-clone guide, Table 1"); raise SystemExit(3)
raise SystemExit(0 if ok else 1)
PY
'''


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--check", action="store_true"); args = ap.parse_args()
    c = cfg(); sp = Spark()
    say(f"connecting to {'this machine' if sp.local else c['spark']['user'] + '@' + c['spark']['host']}")
    out = sp.run("hostname && uname -m && nvidia-smi --query-gpu=name,driver_version --format=csv,noheader && df -h . | tail -1", venv=False)
    if "aarch64" not in out: say("warning: this does not look like a DGX Spark (aarch64 expected)")
    sp.sync_code()
    if not args.check:
        say("preparing the avatar environment on the Spark (first run takes several minutes)")
        sp.run(SETUP.replace("__LS__", c["models"]["latentsync_dir"]).replace("__GIT__", c["models"]["latentsync_git"]), venv=False)
    say("checking the GPU from PyTorch")
    try:
        sp.run(GPU_CHECK); set_state(connected=True, gpu_ok=True); say("connected; GPU ready")
    except RuntimeError as e:
        set_state(connected=True, gpu_ok=False); say(str(e).splitlines()[0]); sys.exit(1)


if __name__ == "__main__":
    main()
