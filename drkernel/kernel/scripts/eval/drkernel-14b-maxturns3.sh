#!/bin/bash


# Source the common grading script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/grading_common.sh"

FSDP_SIZE=-1
PROJECT_NAME="kernel-grading"
RUN_NAME="drkernel-14b-maxturns3"
EXPERIMENT_NAME=${RUN_NAME}

HDFS_RUNS_PATH=""
EVAL_DATASET="hkust-nlp/drkernel-validation-data"

MULTI_TURN=True
MAX_USER_TURNS=3

GRADIO_VISUALIZATION=True
GRADIO_SHARE=True
VISUALIZE_ONLY=False

MAX_PROMPT_LENGTH=20480
MAX_RESPONSE_LENGTH=8192
# Output Paths
OUTPUT_DIR="${HDFS_RUNS_PATH}/${RUN_NAME}/grading_results"
OUTPUT_PATH="${OUTPUT_DIR}/graded_results.parquet"
METRICS_OUTPUT_PATH="${OUTPUT_DIR}/metrics.json"
RAW_RESPONSE_PATH="${OUTPUT_DIR}/raw_responses.jsonl"

HF_MODEL_PATH="hkust-nlp/drkernel-14b"
MODEL_NAME="${HF_MODEL_PATH}"
MODEL_PATH="${MODEL_NAME}"
    
# Generation Parameters
N_SAMPLES=8                  # Generate 4 samples per prompt
BATCH_SIZE=128                 # The whole batch to rollout engine. And it will process data by itself.
TEMPERATURE=1.0              # Sampling temperature
TOP_P=0.95                   # Top-p (nucleus) sampling
DO_SAMPLE=True               # Enable sampling (False for greedy)

# Rollout Mode
ROLLOUT_MODE="async_vllm"  # or "standalone_vllm"
ROLLOUT_GPU_MEMORY_UTIL=0.5
ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=1

# Evaluation Metrics
SOLVE_THRESHOLD=0.99         # Score >= 0.99 considered as "solved"
PASS_AT_K=1                  # Compute pass@1 metric

REWARD_SERVER_URL="${REWARD_SERVER_URL:-${KERNELGYM_SERVER_URL:-""}}"    # set directly or via env


# Reward Manager
REWARD_MANAGER="kernel_async"
REWARD_FUNC_NAME="calculate_reward_speedup"

# Reward Weights (compilation, correctness, performance)
REWARD_WEIGHTS="0.3_0.4_0.3"

# Reward Parameters
REWARD_ENHANCED=True
REWARD_USE_SANDBOX_RATE_LIMIT=True
REWARD_RATE_LIMIT=64
REWARD_ACQUIRE_TIMEOUT=2400
REWARD_MAX_CONCURRENT=64
REWARD_TIMEOUT=2400
REWARD_MAX_RETRIES=3
REWARD_TASK_TIMEOUT=600
REWARD_TASK_TIMEOUT_CLIENT=2400
REWARD_PRINT_STATUS=True
NUM_PERF_TRIALS=10
NUM_CORRECT_TRIALS=5
SPEEDUP_REWARD_UPPER_BOUND=3.0

# Custom Reward Function (optional)
CUSTOM_REWARD_PATH="kernel/rewards/kernel_reward.py"
CUSTOM_REWARD_NAME="compute_kernel_reward_batch"

# System Configuration
# Will use environment variables if available
# NNODES=${ARNOLD_WORKER_NUM:-1}
NNODES=1
# N_GPUS_PER_NODE=${ARNOLD_WORKER_GPU:-1}
N_GPUS_PER_NODE=8

# Qwen3 chat template fix (if needed)
FIX_QWEN3_CHAT_TEMPLATE=False

export PROJECT_NAME
export RUN_NAME
export EVAL_DATASET
export OUTPUT_PATH
export METRICS_OUTPUT_PATH
export RAW_RESPONSE_PATH
# export DATAPROTO_PATH  # Uncomment if using cache

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

# =============================================================================
# Run Grading
# =============================================================================

echo "=========================================="
echo "Kernel Code Grading Example"
echo "=========================================="
echo "This example demonstrates kernel code evaluation"
echo "Modify the configuration above for your use case"
echo "=========================================="
echo ""

# Call main from grading_common.sh with command-line arguments
main "$@"
