#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}"

# clone verl
git submodule update --init
cd "${ROOT_DIR}/verl"

pip install -e . --no-build-isolation --no-deps

pip install --no-cache-dir "ray==2.47.1"

pip install --no-cache-dir "vllm==0.10.2" "torch==2.8.0" "torchvision==0.23.0" "torchaudio==2.8.0" tensordict torchdata \
    "transformers[hf_xet]==4.56.0" accelerate datasets peft hf-transfer \
    "numpy<2.0.0" "pyarrow>=15.0.0" pandas \
    codetiming hydra-core pylatexenc qwen-vl-utils dill pybind11 liger-kernel mathruler decord torchcodec \
    pytest yapf py-spy pre-commit ruff uv pipx

pip install sandbox-fusion --user

pip install logfire --user

pip install gradio --user
pip install huggingface_hub --user
pip install protobuf==3.20 --user
pip install wandb==0.16.6 --user

# Install flash-attn-2.8.3
ABI_FLAG="${ABI_FLAG:-FALSE}"
URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abi${ABI_FLAG}-cp310-cp310-linux_x86_64.whl"
wget -nv -P . "${URL}"
pip install --no-cache-dir "./$(basename "${URL}")"
