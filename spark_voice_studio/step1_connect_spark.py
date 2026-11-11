#!/usr/bin/env python3
"""Step 1 - Connect to the DGX Spark.

Checks SSH, the GPU, and the Python environment on the Spark; creates the studio venv there with the
cu130 PyTorch wheels (sm_120 kernels run on the GB10's sm_121), the engine packages and the model server
dependencies; syncs this folder to the Spark; records the result in state.json.

    python step1_connect_spark.py            # connect, check, install what is missing
    python step1_connect_spark.py --check    # connect and report only
"""
import argparse, sys
from studio.common import Spark, cfg, say, set_state

SETUP = r'''
set -e
command -v python3.12 >/dev/null || sudo apt-get install -y -qq python3.12 python3.12-venv >/dev/null
[ -d .venv ] || python3.12 -m venv .venv
source .venv/bin/activate
pip install -q --upgrade pip wheel
if ! python -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('torch') else 1)"; then
  echo "== installing PyTorch (CUDA 13 aarch64 wheels)"
  pip install -q torch torchaudio --index-url https://download.pytorch.org/whl/cu130
fi
pip install -q soundfile librosa numpy pyyaml fastapi "uvicorn[standard]" huggingface_hub transformers accelerate qwen-tts
sudo apt-get install -y -qq ffmpeg sox libsndfile1 >/dev/null 2>&1 || true
mkdir -p models data/voice data/keywords out checkpoints
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
    sp = Spark()
    say(f"connecting to {'this machine' if sp.local else cfg()['spark']['user'] + '@' + cfg()['spark']['host']}")
    out = sp.run("hostname && uname -m && nvidia-smi --query-gpu=name,driver_version --format=csv,noheader", venv=False)
    if "aarch64" not in out:
        say("warning: this does not look like a DGX Spark (aarch64 expected)")
    sp.sync_code()
    if not args.check:
        say("preparing the studio environment on the Spark (first run takes a few minutes)")
        sp.run(SETUP, venv=False)
    say("checking the GPU from PyTorch")
    try:
        sp.run(GPU_CHECK)
        set_state(connected=True, gpu_ok=True)
        say("connected; GPU ready")
    except RuntimeError as e:
        set_state(connected=True, gpu_ok=False)
        say(str(e).splitlines()[0]); sys.exit(1)


if __name__ == "__main__":
    main()
