#!/usr/bin/env bash
# One-time install of the face-LoRA trainer (musubi-tuner) next to the avatar studio. It reuses the torch already in .venv-it.
# Safe to re-run. After this:  cd spark_avatar_studio && python3 step10_train_face_lora.py --i-am-the-subject --all
set -euo pipefail
H="$HOME/avatar-studio"; cd "$H"
[ -d musubi-tuner ] || git clone --depth 1 https://github.com/kohya-ss/musubi-tuner
[ -d .venv-train ] || python3.12 -m venv .venv-train
echo "$H/.venv-it/lib/python3.12/site-packages" > .venv-train/lib/python3.12/site-packages/zz_it.pth
.venv-train/bin/pip install -q --upgrade pip
(cd musubi-tuner && ../.venv-train/bin/pip install -q -e . --no-deps)
.venv-train/bin/pip install -q av toml voluptuous ftfy easydict sentencepiece bitsandbytes
.venv-train/bin/python -c "import torch, peft, musubi_tuner; print('trainer ready, torch', torch.__version__, 'cuda', torch.cuda.is_available())"
# the training code lives in the project; keep the installed studio in step with it
cp -u "$(dirname "$0")/../spark_avatar_studio/step10_train_face_lora.py" "$H/" 2>/dev/null || true
