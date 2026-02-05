#!/bin/bash

# Common grading script for kernel code evaluation
# This script contains shared logic for evaluating kernel code generation models
# Task-specific scripts should source this and override specific parameters
#
# USAGE:
# 1. Source this script from your task-specific evaluation script
# 2. Set your datasets, model path, and any overrides
# 3. Call main "$@" to run grading with command-line argument support
#
# OUTPUT:
# - Parquet file with solve_rate column
# - Optional: JSON metrics file with detailed statistics
# - Optional: JSONL file with raw responses
# - Optional: DataProto cache for reuse

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../../setup_env.sh"

VAR_SERVER_WITH_TRAINING=${SERVER_WITH_TRAINING:-False}

# =============================================================================
# Default Configuration Values
# =============================================================================

# Dataset and Output Paths (MUST be set by task-specific scripts)
EVAL_DATASET=${EVAL_DATASET:-""}                    # Input dataset path (parquet)
OUTPUT_PATH=${OUTPUT_PATH:-""}                      # Output graded results path (parquet)
RAW_RESPONSE_PATH=${RAW_RESPONSE_PATH:-""}         # Optional: raw responses JSONL
DATAPROTO_PATH=${DATAPROTO_PATH:-""}               # Optional: cache for DataProto
METRICS_OUTPUT_PATH=${METRICS_OUTPUT_PATH:-""}     # Optional: metrics JSON output
FSDP_SIZE=${FSDP_SIZE:-1}                           # Optional: FSDP tensor model parallel size

GRADIO_VISUALIZATION=${GRADIO_VISUALIZATION:-False}
GRADIO_SHARE=${GRADIO_SHARE:-True}
VISUALIZE_ONLY=${VISUALIZE_ONLY:-False}

MULTI_TURN=${MULTI_TURN:-False}
MAX_USER_TURNS=${MAX_USER_TURNS:-3}

REFERENCE_BACKEND=${REFERENCE_BACKEND:-"pytorch"}

# Model Configuration
MODEL_NAME=${MODEL_NAME:-"Qwen3-8B-Base"}
MODEL_PATH=${MODEL_PATH:-""}                        # MUST be set by task script

# Generation Parameters
N_SAMPLES=${N_SAMPLES:-4}                           # Number of samples per prompt
BATCH_SIZE=${BATCH_SIZE:-8}                         # Batch size for generation
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096}
TEMPERATURE=${TEMPERATURE:-0.8}
TOP_P=${TOP_P:-0.95}
TOP_K=${TOP_K:--1}
MIN_P=${MIN_P:-0.0}
DO_SAMPLE=${DO_SAMPLE:-True}
APPLY_CHAT_TEMPLATE=${APPLY_CHAT_TEMPLATE:-True}

MULTI_ITERATION=${MULTI_ITERATION:-False}
MAX_ITERATIONS=${MAX_ITERATIONS:-0}
REMAIN_TURNS=${REMAIN_TURNS:-2}
ITERATION_METHOD=${ITERATION_METHOD:-"last"}
BEST_SELECTION_METRIC=${BEST_SELECTION_METRIC:-"reward"}

# Evaluation Metrics
SOLVE_THRESHOLD=${SOLVE_THRESHOLD:-0.99}           # Threshold for "solved" (0.0-1.0)
PASS_AT_K=${PASS_AT_K:-1}                          # Pass@k metric k value

# Rollout Mode Configuration
ROLLOUT_MODE=${ROLLOUT_MODE:-"sync"}                # "sync", "async_vllm", "async_agent", or "standalone_vllm"
ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=${ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE:-1}
ROLLOUT_GPU_MEMORY_UTIL=${ROLLOUT_GPU_MEMORY_UTIL:-0.75}
ROLLOUT_ENFORCE_EAGER=${ROLLOUT_ENFORCE_EAGER:-False}

BACKEND=${BACKEND:-"vllm"}
OPENAI_MODEL=${OPENAI_MODEL:-""}
OPENAI_THINKING_MODE=${OPENAI_THINKING_MODE:-False}
OPENAI_API_KEY=${OPENAI_API_KEY:-""}
OPENAI_BASE_URL=${OPENAI_BASE_URL:-""}
OPENAI_TIMEOUT=${OPENAI_TIMEOUT:-120}
OPENAI_MAX_RETRIES=${OPENAI_MAX_RETRIES:-3}
OPENAI_MAX_CONCURRENCY=${OPENAI_MAX_CONCURRENCY:-64}

# Reward Manager Configuration
REWARD_MANAGER=${REWARD_MANAGER:-"kernel_async"}
REWARD_SERVER_URL=${REWARD_SERVER_URL:-"${KERNELGYM_SERVER_URL}"}
REWARD_FUNC_NAME=${REWARD_FUNC_NAME:-"calculate_reward_weighted"}

# Kernel Reward Parameters
REWARD_ENHANCED=${REWARD_ENHANCED:-True}
REWARD_USE_SANDBOX_RATE_LIMIT=${REWARD_USE_SANDBOX_RATE_LIMIT:-True}
REWARD_RATE_LIMIT=${REWARD_RATE_LIMIT:-64}
REWARD_ACQUIRE_TIMEOUT=${REWARD_ACQUIRE_TIMEOUT:-2400}
REWARD_MAX_CONCURRENT=${REWARD_MAX_CONCURRENT:-64}
REWARD_TIMEOUT=${REWARD_TIMEOUT:-1800}
REWARD_MAX_RETRIES=${REWARD_MAX_RETRIES:-3}
REWARD_TASK_TIMEOUT=${REWARD_TASK_TIMEOUT:-600}
REWARD_TASK_TIMEOUT_CLIENT=${REWARD_TASK_TIMEOUT_CLIENT:-2400}
REWARD_PRINT_STATUS=${REWARD_PRINT_STATUS:-True}
NUM_PERF_TRIALS=${NUM_PERF_TRIALS:-100}
NUM_CORRECT_TRIALS=${NUM_CORRECT_TRIALS:-5}
SPEEDUP_REWARD_UPPER_BOUND=${SPEEDUP_REWARD_UPPER_BOUND:-3.0}

# Reward Weights (compilation, correctness, performance)
REWARD_WEIGHTS=${REWARD_WEIGHTS:-"0.3_0.4_0.3"}

# Reward Policy (penalties)
REWARD_PENALTY_SCORE=${REWARD_PENALTY_SCORE:-0.0}
REWARD_PENALTY_COMPILATION=${REWARD_PENALTY_COMPILATION:--0.5}
REWARD_PENALTY_CORRECTNESS=${REWARD_PENALTY_CORRECTNESS:--0.3}
REWARD_PENALTY_PERF_DEGRADE=${REWARD_PENALTY_PERF_DEGRADE:--0.1}

# Custom Reward Function
CUSTOM_REWARD_PATH=${CUSTOM_REWARD_PATH:-"kernel/rewards/kernel_reward.py"}
CUSTOM_REWARD_NAME=${CUSTOM_REWARD_NAME:-"compute_kernel_reward_batch"}

MAX_NUM_BATCHED_TOKENS=$(expr $MAX_PROMPT_LENGTH + $MAX_RESPONSE_LENGTH + 1000)

# System Configuration
NNODES=${NNODES:-${ARNOLD_WORKER_NUM:-1}}
if [ "$VAR_SERVER_WITH_TRAINING" = "true" ]; then
    NNODES=$((NNODES - 1))
    echo "Since the last worker will be responsible for starting the server and writing the URL, the number of nodes will be reduced by 1. NNODES: $NNODES"
fi
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-${ARNOLD_WORKER_GPU:-1}}
FIX_QWEN3_CHAT_TEMPLATE=${FIX_QWEN3_CHAT_TEMPLATE:-False}

# Project and Experiment Names
PROJECT_NAME=${PROJECT_NAME:-"kernel-grading"}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-""}              # Will be auto-generated

# =============================================================================
# Helper Functions
# =============================================================================

generate_short_hash() {
  local input_string="$1"
  local hash=$(echo -n "$input_string" | sha256sum | cut -c1-8)
  echo "$hash"
}

generate_suffix() {
  local suffix=""

  while [[ "$#" -gt 0 ]]; do
    case $1 in
      --n_samples) suffix+="_n$2"; shift 2 ;;
      --batch_size) suffix+="_bs$2"; shift 2 ;;
      --temperature) suffix+="_temp$2"; shift 2 ;;
      --top_p) suffix+="_topp$2"; shift 2 ;;
      --rollout_mode) suffix+="_$2"; shift 2 ;;
      --solve_threshold) suffix+="_thresh$2"; shift 2 ;;
      --pass_at_k) suffix+="_pass$2"; shift 2 ;;
      --model_name) suffix+="_$(echo $2 | sed 's/\//_/g')"; shift 2 ;;
      *) shift ;;
    esac
  done

  local suffix_hash=$(generate_short_hash "$suffix")
  echo "_$suffix_hash"
}

parse_reward_weights() {
  local weights="$1"
  local compilation
  local correctness
  local performance

  if [[ $weights =~ ^[0-9.]+_[0-9.]+_[0-9.]+$ ]]; then
    compilation=$(echo $weights | cut -d'_' -f1)
    correctness=$(echo $weights | cut -d'_' -f2)
    performance=$(echo $weights | cut -d'_' -f3)
  else
    echo "Warning: reward_weights '$weights' not in format 'comp_corr_perf', using defaults"
    compilation=0.3
    correctness=0.4
    performance=0.3
  fi

  REWARD_WEIGHT_COMPILATION=$compilation
  REWARD_WEIGHT_CORRECTNESS=$correctness
  REWARD_WEIGHT_PERFORMANCE=$performance
  echo "Reward Weights - Compilation: $compilation, Correctness: $correctness, Performance: $performance"
}

show_help() {
  echo "Kernel Code Grading Script"
  echo ""
  echo "Usage: $0 [OPTIONS]"
  echo ""
  echo "Required Options:"
  echo "  --eval_dataset PATH           Input dataset (parquet file)"
  echo "  --output_path PATH            Output graded results (parquet file)"
  echo "  --model_path PATH             Model checkpoint path"
  echo ""
  echo "Generation Options:"
  echo "  --n_samples N                 Samples per prompt (default: 4)"
  echo "  --batch_size SIZE             Batch size (default: 8)"
  echo "  --temperature TEMP            Sampling temperature (default: 0.8)"
  echo "  --top_p VALUE                 Top-p sampling (default: 0.95)"
  echo "  --rollout_mode MODE           Rollout mode: sync|async_vllm|async_agent|standalone_vllm (default: sync)"
  echo "  --rollout_enforce_eager BOOL  Force eager mode for vLLM (default: False)"
  echo ""
  echo "Evaluation Options:"
  echo "  --solve_threshold THRESH      Solve threshold 0.0-1.0 (default: 0.99)"
  echo "  --pass_at_k K                 Pass@k value (default: 1)"
  echo ""
  echo "Reward Options:"
  echo "  --reward_server_url URL       Kernel server URL"
  echo "  --reward_weights W            Weights as comp_corr_perf (default: 0.3_0.4_0.3)"
  echo ""
  echo "Output Options:"
  echo "  --raw_response_path PATH      Save raw responses JSONL"
  echo "  --metrics_output_path PATH    Save metrics JSON"
  echo "  --dataproto_path PATH         Cache/load DataProto"
  echo ""
  echo "Examples:"
  echo "  $0 --eval_dataset data.parquet --output_path results.parquet --model_path ~/models/qwen"
  echo "  $0 --eval_dataset data.parquet --output_path results.parquet --rollout_mode async_vllm"
  echo "  $0 --eval_dataset data.parquet --output_path results.parquet --rollout_mode standalone_vllm"
  echo ""
}

parse_arguments() {
  echo "Arguments received: $@"

  # Check for help
  for arg in "$@"; do
    if [[ "$arg" == "--help" || "$arg" == "-h" ]]; then
      show_help
      exit 0
    fi
  done

  # Generate suffix for experiment name
  SUFFIX=$(generate_suffix "$@")

  # Parse arguments
  while [[ "$#" -gt 0 ]]; do
    echo "Processing: $1"
    case "$1" in
      --eval_dataset) EVAL_DATASET="$2"; shift 2 ;;
      --output_path) OUTPUT_PATH="$2"; shift 2 ;;
      --raw_response_path) RAW_RESPONSE_PATH="$2"; shift 2 ;;
      --dataproto_path) DATAPROTO_PATH="$2"; shift 2 ;;
      --metrics_output_path) METRICS_OUTPUT_PATH="$2"; shift 2 ;;
      --model_name) MODEL_NAME="$2"; shift 2 ;;
      --model_path) MODEL_PATH="$2"; shift 2 ;;
      --n_samples) N_SAMPLES="$2"; shift 2 ;;
      --batch_size) BATCH_SIZE="$2"; shift 2 ;;
      --max_prompt_length) MAX_PROMPT_LENGTH="$2"; shift 2 ;;
      --max_response_length) MAX_RESPONSE_LENGTH="$2"; shift 2 ;;
      --temperature) TEMPERATURE="$2"; shift 2 ;;
      --top_p) TOP_P="$2"; shift 2 ;;
      --top_k) TOP_K="$2"; shift 2 ;;
      --min_p) MIN_P="$2"; shift 2 ;;
      --do_sample) DO_SAMPLE="$2"; shift 2 ;;
      --apply_chat_template) APPLY_CHAT_TEMPLATE="$2"; shift 2 ;;
      --solve_threshold) SOLVE_THRESHOLD="$2"; shift 2 ;;
      --pass_at_k) PASS_AT_K="$2"; shift 2 ;;
      --rollout_mode) ROLLOUT_MODE="$2"; shift 2 ;;
      --rollout_tp) ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE="$2"; shift 2 ;;
      --rollout_gpu_memory_util) ROLLOUT_GPU_MEMORY_UTIL="$2"; shift 2 ;;
      --rollout_enforce_eager) ROLLOUT_ENFORCE_EAGER="$2"; shift 2 ;;
      --reward_manager) REWARD_MANAGER="$2"; shift 2 ;;
      --reward_server_url) REWARD_SERVER_URL="$2"; shift 2 ;;
      --reward_func_name) REWARD_FUNC_NAME="$2"; shift 2 ;;
      --reward_enhanced) REWARD_ENHANCED="$2"; shift 2 ;;
      --reward_use_sandbox_rate_limit) REWARD_USE_SANDBOX_RATE_LIMIT="$2"; shift 2 ;;
      --reward_rate_limit) REWARD_RATE_LIMIT="$2"; shift 2 ;;
      --reward_acquire_timeout) REWARD_ACQUIRE_TIMEOUT="$2"; shift 2 ;;
      --reward_max_concurrent) REWARD_MAX_CONCURRENT="$2"; shift 2 ;;
      --reward_timeout) REWARD_TIMEOUT="$2"; shift 2 ;;
      --reward_max_retries) REWARD_MAX_RETRIES="$2"; shift 2 ;;
      --reward_task_timeout) REWARD_TASK_TIMEOUT="$2"; shift 2 ;;
      --reward_print_status) REWARD_PRINT_STATUS="$2"; shift 2 ;;
      --reward_weights) REWARD_WEIGHTS="$2"; shift 2 ;;
      --num_perf_trials) NUM_PERF_TRIALS="$2"; shift 2 ;;
      --num_correct_trials) NUM_CORRECT_TRIALS="$2"; shift 2 ;;
      --speedup_reward_upper_bound) SPEEDUP_REWARD_UPPER_BOUND="$2"; shift 2 ;;
      --custom_reward_path) CUSTOM_REWARD_PATH="$2"; shift 2 ;;
      --custom_reward_name) CUSTOM_REWARD_NAME="$2"; shift 2 ;;
      --nnodes) NNODES="$2"; shift 2 ;;
      --n_gpus_per_node) N_GPUS_PER_NODE="$2"; shift 2 ;;
      --fix_qwen3_chat_template) FIX_QWEN3_CHAT_TEMPLATE="$2"; shift 2 ;;
      --project_name) PROJECT_NAME="$2"; shift 2 ;;
      --experiment_name) EXPERIMENT_NAME="$2"; shift 2 ;;
      --gradio_visualization) GRADIO_VISUALIZATION="$2"; shift 2 ;;
      --gradio_share) GRADIO_SHARE="$2"; shift 2 ;;
      --visualize_only) VISUALIZE_ONLY="$2"; shift 2 ;;
      *)
        echo "Unknown option: $1"
        echo "Use --help for usage information"
        exit 1
        ;;
    esac
  done
}

setup_grading_environment() {
  # Validate required parameters
  if [[ -z "$EVAL_DATASET" ]]; then
    echo "Error: --eval_dataset is required"
    exit 1
  fi

  if [[ -z "$OUTPUT_PATH" ]]; then
    echo "Error: --output_path is required"
    exit 1
  fi

  if [[ -z "$MODEL_PATH" ]]; then
    echo "Error: --model_path is required"
    exit 1
  fi

  # Generate experiment name if not provided
  if [[ -z "$EXPERIMENT_NAME" ]]; then
    local dataset_name=$(basename "$EVAL_DATASET" .parquet)
    EXPERIMENT_NAME="${dataset_name}_${MODEL_NAME}${SUFFIX}"
  fi

  # Parse reward weights
  parse_reward_weights "$REWARD_WEIGHTS"

  # Print configuration
  echo "============================================"
  echo "Kernel Code Grading Configuration"
  echo "============================================"
  echo "Experiment: $EXPERIMENT_NAME"
  echo "Project: $PROJECT_NAME"
  echo ""
  echo "Dataset Configuration:"
  echo "  Input Dataset: $EVAL_DATASET"
  echo "  Output Path: $OUTPUT_PATH"
  echo "  Raw Response Path: ${RAW_RESPONSE_PATH:-none}"
  echo "  DataProto Path: ${DATAPROTO_PATH:-none}"
  echo "  Metrics Output Path: ${METRICS_OUTPUT_PATH:-none}"
  echo ""
  echo "Model Configuration:"
  echo "  Model Name: $MODEL_NAME"
  echo "  Model Path: $MODEL_PATH"
  echo ""
  echo "Generation Parameters:"
  echo "  N Samples: $N_SAMPLES"
  echo "  Batch Size: $BATCH_SIZE"
  echo "  Temperature: $TEMPERATURE"
  echo "  Top-P: $TOP_P"
  echo "  Rollout Mode: $ROLLOUT_MODE"
  echo "  Max Prompt Length: $MAX_PROMPT_LENGTH"
  echo "  Max Response Length: $MAX_RESPONSE_LENGTH"
  echo ""
  echo "Evaluation Metrics:"
  echo "  Solve Threshold: $SOLVE_THRESHOLD"
  echo "  Pass@K: $PASS_AT_K"
  echo ""
  echo "Reward Configuration:"
  echo "  Reward Manager: $REWARD_MANAGER"
  echo "  Server URL: ${REWARD_SERVER_URL:-not set}"
  echo "  Reward Function: $REWARD_FUNC_NAME"
  echo "  Compilation Weight: $REWARD_WEIGHT_COMPILATION"
  echo "  Correctness Weight: $REWARD_WEIGHT_CORRECTNESS"
  echo "  Performance Weight: $REWARD_WEIGHT_PERFORMANCE"
  echo ""
  echo "System Configuration:"
  echo "  Nodes: $NNODES"
  echo "  GPUs per Node: $N_GPUS_PER_NODE"
  echo "============================================"
}

run_grading() {
  sleep 1

  # Prepare optional paths as hydra-compatible strings
  local raw_response_arg=""
  local dataproto_arg=""
  local metrics_arg=""

  if [[ -n "$RAW_RESPONSE_PATH" ]]; then
    raw_response_arg="data.raw_response_path=$RAW_RESPONSE_PATH"
  fi

  if [[ -n "$DATAPROTO_PATH" ]]; then
    dataproto_arg="data.dataproto_path=$DATAPROTO_PATH"
  fi

  if [[ -n "$METRICS_OUTPUT_PATH" ]]; then
    metrics_arg="data.metrics_output_path=$METRICS_OUTPUT_PATH"
  fi

  PYTHONUNBUFFERED=1 python -m kernel.main_grading \
      data.path=$EVAL_DATASET \
      data.output_path=$OUTPUT_PATH \
      $raw_response_arg \
      $dataproto_arg \
      $metrics_arg \
      data.n_samples=$N_SAMPLES \
      data.batch_size=$BATCH_SIZE \
      data.max_prompt_length=$MAX_PROMPT_LENGTH \
      data.max_response_length=$MAX_RESPONSE_LENGTH \
      data.solve_threshold=$SOLVE_THRESHOLD \
      data.pass_at_k=$PASS_AT_K \
      data.do_sample=$DO_SAMPLE \
      data.apply_chat_template=$APPLY_CHAT_TEMPLATE \
      model.path=$MODEL_PATH \
      actor_rollout_ref.model.path=$MODEL_PATH \
      actor_rollout_ref.rollout.mode=$ROLLOUT_MODE \
      actor_rollout_ref.rollout.temperature=$TEMPERATURE \
      actor_rollout_ref.rollout.top_p=$TOP_P \
      actor_rollout_ref.rollout.top_k=$TOP_K \
      actor_rollout_ref.rollout.min_p=$MIN_P \
      actor_rollout_ref.rollout.val_kwargs.temperature=$TEMPERATURE \
      actor_rollout_ref.rollout.val_kwargs.top_p=$TOP_P \
      actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE \
      actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTIL \
      actor_rollout_ref.rollout.enforce_eager=$ROLLOUT_ENFORCE_EAGER \
      actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS \
      actor_rollout_ref.rollout.multi_turn.enable=$MULTI_TURN \
      actor_rollout_ref.rollout.multi_turn.max_user_turns=$MAX_USER_TURNS \
      actor_rollout_ref.rollout.multi_turn.multi_iteration.enable=$MULTI_ITERATION \
      actor_rollout_ref.rollout.multi_turn.multi_iteration.max_iterations=$MAX_ITERATIONS \
      actor_rollout_ref.rollout.multi_turn.multi_iteration.remain_turns=$REMAIN_TURNS \
      actor_rollout_ref.rollout.multi_turn.multi_iteration.iteration_method=$ITERATION_METHOD \
      actor_rollout_ref.rollout.multi_turn.multi_iteration.best_selection_metric=$BEST_SELECTION_METRIC \
      actor_rollout_ref.actor.fsdp_config.fsdp_size=$FSDP_SIZE \
      actor_rollout_ref.rollout.backend=$BACKEND \
      actor_rollout_ref.rollout.openai.model=$OPENAI_MODEL \
      actor_rollout_ref.rollout.openai.thinking_mode=$OPENAI_THINKING_MODE \
      actor_rollout_ref.rollout.openai.api_key=$OPENAI_API_KEY \
      actor_rollout_ref.rollout.openai.base_url=$OPENAI_BASE_URL \
      actor_rollout_ref.rollout.openai.timeout=$OPENAI_TIMEOUT \
      actor_rollout_ref.rollout.openai.max_retries=$OPENAI_MAX_RETRIES \
      actor_rollout_ref.rollout.openai.max_concurrency=$OPENAI_MAX_CONCURRENCY \
      reward_model.reward_manager=$REWARD_MANAGER \
      reward_model.reference_backend=$REFERENCE_BACKEND \
      reward_model.server_url='"'$REWARD_SERVER_URL'"' \
      reward_model.reward_func_name=$REWARD_FUNC_NAME \
      reward_model.enhanced=$REWARD_ENHANCED \
      reward_model.use_sandbox_rate_limit=$REWARD_USE_SANDBOX_RATE_LIMIT \
      reward_model.rate_limit=$REWARD_RATE_LIMIT \
      reward_model.acquire_timeout=$REWARD_ACQUIRE_TIMEOUT \
      reward_model.max_concurrent=$REWARD_MAX_CONCURRENT \
      reward_model.timeout=$REWARD_TIMEOUT \
      reward_model.max_retries=$REWARD_MAX_RETRIES \
      reward_model.task_timeout=$REWARD_TASK_TIMEOUT \
      reward_model.task_timeout_in_client=$REWARD_TASK_TIMEOUT_CLIENT \
      reward_model.print_status=$REWARD_PRINT_STATUS \
      reward_model.num_perf_trials=$NUM_PERF_TRIALS \
      reward_model.num_correct_trials=$NUM_CORRECT_TRIALS \
      reward_model.speedup_reward_upper_bound=$SPEEDUP_REWARD_UPPER_BOUND \
      reward_model.reward_weights.compilation=$REWARD_WEIGHT_COMPILATION \
      reward_model.reward_weights.correctness=$REWARD_WEIGHT_CORRECTNESS \
      reward_model.reward_weights.performance=$REWARD_WEIGHT_PERFORMANCE \
      reward_model.reward_policy.penalties.penalty_score=$REWARD_PENALTY_SCORE \
      reward_model.reward_policy.penalties.compilation_fail=$REWARD_PENALTY_COMPILATION \
      reward_model.reward_policy.penalties.correctness_fail=$REWARD_PENALTY_CORRECTNESS \
      reward_model.reward_policy.penalties.perf_degrade=$REWARD_PENALTY_PERF_DEGRADE \
      custom_reward_function.path=$CUSTOM_REWARD_PATH \
      custom_reward_function.name=$CUSTOM_REWARD_NAME \
      trainer.project_name=$PROJECT_NAME \
      trainer.experiment_name=$EXPERIMENT_NAME \
      trainer.nnodes=$NNODES \
      trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
      trainer.fix_qwen3_chat_template=$FIX_QWEN3_CHAT_TEMPLATE \
      gradio=$GRADIO_VISUALIZATION \
      gradio_share=$GRADIO_SHARE \
      visualize_only=$VISUALIZE_ONLY
}

# =============================================================================
# Main Execution
# =============================================================================

main() {
  parse_arguments "$@"
  setup_grading_environment
  run_grading
}

# Only show error if this script is executed directly (not sourced)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Error: This script should not be run directly."
  echo "Please use a task-specific grading script that sources this common script."
  echo "See kernel/scripts/eval/example.sh for an example."
  exit 1
fi
