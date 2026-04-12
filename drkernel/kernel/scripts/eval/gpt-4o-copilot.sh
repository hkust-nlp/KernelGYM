#!/bin/bash
# =============================================================================
# GPT-4o via GitHub Copilot — KernelGYM Evaluation Script
# =============================================================================
# Uses GitHub Copilot API (api.githubcopilot.com) to access GPT-4o
# for Triton kernel generation. Requires an active OpenCode OAuth token
# cached at ~/.local/share/opencode/auth.json.
#
# GPT-4o is a stronger free model on Copilot — used to validate that
# the pipeline produces non-zero correctness scores before scaling up.
#
# Output is saved to /mnt/hstorage/GKG/datasets/distill/ for distillation.
# =============================================================================

# --- Proxy fix: OpenAI SDK crashes with SOCKS proxy ---
unset ALL_PROXY all_proxy HTTP_PROXY http_proxy HTTPS_PROXY https_proxy

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/grading_common.sh"

PROJECT_NAME="kernel-grading"
RUN_NAME="gpt-4o-copilot"
EXPERIMENT_NAME=${RUN_NAME}

REFERENCE_BACKEND="torch_compile"

HDFS_RUNS_PATH="/mnt/hstorage/GKG/datasets/distill"
# Use 10-row subset for small-scale testing (switch to full dataset for production)
# EVAL_DATASET="/mnt/hstorage/GKG/datasets/structured_datasets/drkernel/drkernel-validation-data/validation_data_thinking.parquet"
EVAL_DATASET="/mnt/hstorage/GKG/datasets/structured_datasets/drkernel/drkernel-validation-data/validation_data_thinking_10.parquet"

MULTI_TURN=True
MAX_USER_TURNS=3

GRADIO_VISUALIZATION=True
GRADIO_SHARE=True
VISUALIZE_ONLY=False

MAX_PROMPT_LENGTH=20480
MAX_RESPONSE_LENGTH=16384

OUTPUT_DIR="${HDFS_RUNS_PATH}/${RUN_NAME}/grading_results"
OUTPUT_PATH="${OUTPUT_DIR}/graded_results.parquet"
METRICS_OUTPUT_PATH="${OUTPUT_DIR}/metrics.json"
RAW_RESPONSE_PATH="${OUTPUT_DIR}/raw_responses.jsonl"

# --- Model path (only the tokenizer is loaded — not used for inference) ---
# ORIGINAL_MODEL="${HDFS_MODEL_PATH}/qwen3-8b-base"
ORIGINAL_MODEL="/mnt/hstorage/GKG/pretrained_models/drkernel-8b"

ACTOR_PATH="${ORIGINAL_MODEL}"
HF_MODEL_PATH="${ORIGINAL_MODEL}"
MODEL_NAME="${HF_MODEL_PATH}"
MODEL_PATH="${MODEL_NAME}"

# --- Generation Parameters ---
# For small-scale test: N_SAMPLES=1; for production: N_SAMPLES=8
N_SAMPLES=1
BATCH_SIZE=128
TEMPERATURE=1.0
TOP_P=0.95
DO_SAMPLE=True

# --- Rollout Mode ---
ROLLOUT_MODE="standalone_vllm"
ROLLOUT_GPU_MEMORY_UTIL=0.7
ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=1
FSDP_SIZE=-1
ROLLOUT_ENFORCE_EAGER=True

# --- Evaluation Metrics ---
SOLVE_THRESHOLD=0.99
PASS_AT_K=1

# =============================================================================
# GitHub Copilot API Configuration
# =============================================================================
BACKEND="openai"
OPENAI_MODEL="gpt-4o"

# Dynamically read OAuth token from OpenCode's auth cache
AUTH_FILE="$HOME/.local/share/opencode/auth.json"
if [[ -f "$AUTH_FILE" ]]; then
  OPENAI_API_KEY=$(python3 -c "import json; print(json.load(open('$AUTH_FILE'))['github-copilot']['access'])" 2>/dev/null)
  if [[ -z "$OPENAI_API_KEY" ]]; then
    echo "ERROR: Failed to extract token from $AUTH_FILE"
    exit 1
  fi
  echo "GitHub Copilot token loaded from $AUTH_FILE"
else
  echo "ERROR: OpenCode auth file not found at $AUTH_FILE"
  echo "Please run 'opencode' first to authenticate with GitHub Copilot."
  exit 1
fi

OPENAI_BASE_URL="https://api.githubcopilot.com"
OPENAI_TIMEOUT=120
OPENAI_MAX_RETRIES=3
# GPT-4o is stronger but still free — use moderate concurrency
OPENAI_MAX_CONCURRENCY=8

# Required headers for GitHub Copilot API
OPENAI_EXTRA_HEADERS="{User-Agent: opencode/0.1, Openai-Intent: conversation-edits, x-initiator: user}"

# =============================================================================
# Sandbox / Reward Configuration
# =============================================================================
REWARD_SERVER_URL="${REWARD_SERVER_URL:-${KERNELGYM_SERVER_URL:-"http://192.168.31.68:8001"}}"
# REWARD_SERVER_URL="${REWARD_SERVER_URL:-${KERNELGYM_SERVER_URL:-""}}"
REWARD_MANAGER="kernel_async"
REWARD_FUNC_NAME="calculate_reward_speedup"

# Reward Weights (compilation, correctness, performance)
REWARD_WEIGHTS="0.3_0.4_0.3"

REWARD_ENHANCED=True
REWARD_USE_SANDBOX_RATE_LIMIT=True
REWARD_RATE_LIMIT=64
REWARD_ACQUIRE_TIMEOUT=2400
REWARD_MAX_CONCURRENT=64
REWARD_TIMEOUT=1800
REWARD_MAX_RETRIES=3
REWARD_TASK_TIMEOUT=1800
REWARD_PRINT_STATUS=True
NUM_PERF_TRIALS=10
NUM_CORRECT_TRIALS=5
SPEEDUP_REWARD_UPPER_BOUND=3.0

# Custom Reward Function
CUSTOM_REWARD_PATH="/mnt/hstorage/GKG/framework/KernelGYM/drkernel/kernel/rewards/kernel_reward.py"
CUSTOM_REWARD_NAME="compute_kernel_reward_batch"

NNODES=1
# OpenAI backend does not need local GPUs — all inference is via API
N_GPUS_PER_NODE=0

FIX_QWEN3_CHAT_TEMPLATE=False

# =============================================================================
# Exports
# =============================================================================
export PROJECT_NAME
export RUN_NAME
export EVAL_DATASET
export OUTPUT_PATH
export METRICS_OUTPUT_PATH
export RAW_RESPONSE_PATH

export MODEL_NAME
export MODEL_PATH

export N_SAMPLES
export BATCH_SIZE
export TEMPERATURE
export TOP_P
export DO_SAMPLE

export ROLLOUT_MODE
export ROLLOUT_GPU_MEMORY_UTIL
export ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE

export SOLVE_THRESHOLD
export PASS_AT_K

export REWARD_SERVER_URL
export REWARD_MANAGER
export REWARD_FUNC_NAME
export REWARD_WEIGHTS

export REWARD_ENHANCED
export REWARD_USE_SANDBOX_RATE_LIMIT
export REWARD_RATE_LIMIT
export REWARD_ACQUIRE_TIMEOUT
export REWARD_MAX_CONCURRENT
export REWARD_TIMEOUT
export REWARD_MAX_RETRIES
export REWARD_TASK_TIMEOUT
export REWARD_PRINT_STATUS
export NUM_PERF_TRIALS
export NUM_CORRECT_TRIALS
export SPEEDUP_REWARD_UPPER_BOUND

export CUSTOM_REWARD_PATH
export CUSTOM_REWARD_NAME

export NNODES
export N_GPUS_PER_NODE
export FIX_QWEN3_CHAT_TEMPLATE

# Create output directory
mkdir -p "$OUTPUT_DIR"

main "$@"
