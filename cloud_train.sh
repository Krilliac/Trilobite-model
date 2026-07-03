#!/usr/bin/env bash
# cloud_train.sh — one-shot QLoRA fine-tune of trilobite on a rented cloud GPU.
#
# Where to run: any Linux box with an NVIDIA GPU + CUDA (RunPod, Lambda,
# vast.ai, Paperspace, a cloud VM with an A100/H100/4090/3090, etc.). A 24 GB
# card (e.g. RTX 4090/3090) comfortably QLoRA-tunes the 7B; 16 GB works with
# the smaller bases.
#
# Usage (on the GPU box):
#   git clone https://github.com/Krilliac/Trilobite-model.git
#   cd Trilobite-model
#   # upload your dataset (see step 3 below) then:
#   bash cloud_train.sh
#
# Env knobs (all optional):
#   TRILOBITE_BASE   HF base model (default Qwen/Qwen2.5-Coder-7B-Instruct)
#   HF_TOKEN         set to also push the adapter to the Hugging Face Hub
#   HF_REPO          e.g. Krilliac/trilobite-lora  (needed if HF_TOKEN set)
set -euo pipefail

BASE="${TRILOBITE_BASE:-Qwen/Qwen2.5-Coder-7B-Instruct}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

echo "======================================================================"
echo " trilobite cloud QLoRA trainer"
echo " base model: $BASE"
echo "======================================================================"

echo "== 1/5 GPU check =="
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found — this needs a CUDA GPU box. Aborting." >&2
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo "== 2/5 python venv + deps =="
if ! command -v python3 >/dev/null 2>&1; then
  (apt-get update -y && apt-get install -y python3 python3-venv python3-pip git) \
    || { echo "install python3/venv/pip/git manually" >&2; exit 1; }
fi
if [ ! -d .venv ]; then python3 -m venv .venv; fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip wheel
# CUDA build of torch first (cu121 works on most current GPU images; change the
# index if your box has a different CUDA runtime).
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121 \
  || python -m pip install torch   # fall back to default wheel if the index fails
python -m pip install -r requirements-train.txt

echo "== 3/5 dataset =="
if [ ! -f training_data.jsonl ]; then
  cat >&2 <<'MSG'
ERROR: training_data.jsonl not found in this directory.

Your training data lives on your LOCAL machine (it is gitignored on purpose —
it is derived from your own coding interactions). Get it onto this GPU box, e.g.
from your local machine:

  # regenerate it locally first if needed:
  ./venv/Scripts/python.exe export_training_data.py
  # then upload it to this box:
  scp training_data.jsonl user@THIS_BOX_IP:PATH/Trilobite-model/

Then re-run:  bash cloud_train.sh
MSG
  exit 1
fi
N=$(wc -l < training_data.jsonl | tr -d ' ')
echo "found training_data.jsonl with $N examples"
if [ "$N" -lt 50 ]; then
  echo "NOTE: only $N examples — this will over/under-fit; it is a pipeline proof."
  echo "Grow the dataset first (run trilobite /train locally) for a real tune."
fi

echo "== 4/5 QLoRA fine-tune (this is the long part) =="
export TRILOBITE_BASE="$BASE"
python qlora_train.py
# qlora_train.py writes the LoRA adapter to ./trilobite-lora/

echo "== 5/5 package the adapter =="
if [ ! -d trilobite-lora ]; then
  echo "WARNING: trilobite-lora/ not produced — check the training log above." >&2
  exit 1
fi
STAMP="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tr ' /' '__')"
TARBALL="trilobite-lora.tar.gz"
tar -czf "$TARBALL" trilobite-lora
echo "adapter packaged -> $HERE/$TARBALL ($(du -h "$TARBALL" | cut -f1))"

# Optional: push to the HF Hub if credentials are provided
if [ -n "${HF_TOKEN:-}" ] && [ -n "${HF_REPO:-}" ]; then
  echo "-- pushing adapter to HF Hub: $HF_REPO --"
  python -m pip install -q huggingface_hub
  python - <<PY
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(os.environ["HF_REPO"], exist_ok=True)
api.upload_folder(folder_path="trilobite-lora", repo_id=os.environ["HF_REPO"])
print("pushed to", os.environ["HF_REPO"])
PY
fi

cat <<MSG

======================================================================
 DONE. LoRA adapter is in:  $HERE/trilobite-lora/   (also $TARBALL)
======================================================================
Get it back to your machine:
  scp user@THIS_BOX_IP:$HERE/$TARBALL .

Then turn it into a runnable Ollama model locally. peft outputs a safetensors
adapter; Ollama's Modelfile ADAPTER wants a GGUF LoRA, so convert with llama.cpp:
  python llama.cpp/convert_lora_to_gguf.py trilobite-lora --outfile trilobite-lora.gguf
  # Modelfile:
  #   FROM $BASE-equivalent-gguf   (or your base ollama model)
  #   ADAPTER ./trilobite-lora.gguf
  ollama create trilobite -f Modelfile
See TRAINING.md in this repo for the full, honest walkthrough + caveats.
MSG
