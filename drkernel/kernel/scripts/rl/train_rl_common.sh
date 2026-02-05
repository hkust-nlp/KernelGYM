#!/bin/bash

# Common RL training script for GRPO and RLOO
# This script contains shared logic for training RL models with GRPO/RLOO algorithms
# Task-specific scripts should source this and override specific parameters
#
# BEST PRACTICE DEFAULTS:
# - LOSS_AGG_MODE="seq-mean-token-sum" (REINFORCE-aligned, recommended over token-mean)
# - LOSS_SCALE_FACTOR=1000.0 (prevents gradient underflow for long sequences)
# - ALGORITHM="grpo" (default algorithm, can be overridden to "rloo")
# - BATCH_STD=False (for GRPO), True (for RLOO) - prevents length bias
#
# USAGE:
# 1. Source this script from your task-specific script
# 2. Set your datasets, reward manager, and any overrides
# 3. Call main "$@" to run training with command-line argument support

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../../setup_env.sh"

# server
VAR_SERVER_WITH_TRAINING=False
if [ -n "$SERVER_WITH_TRAINING" ]; then
    VAR_SERVER_WITH_TRAINING=$SERVER_WITH_TRAINING
    echo "SERVER_WITH_TRAINING: $VAR_SERVER_WITH_TRAINING"
fi
echo "SERVER_WITH_TRAINING: $VAR_SERVER_WITH_TRAINING"

VAR_SERVER_WITH_TRAINING_NODES=0
if [ -n "$SERVER_WITH_TRAINING_NODES" ]; then
    VAR_SERVER_WITH_TRAINING_NODES=$SERVER_WITH_TRAINING_NODES
    echo "SERVER_WITH_TRAINING_NODES: $VAR_SERVER_WITH_TRAINING_NODES"
fi
echo "SERVER_WITH_TRAINING_NODES: $VAR_SERVER_WITH_TRAINING_NODES"

NNODES=${NNODES:-${ARNOLD_WORKER_NUM:-1}}
if [ "$VAR_SERVER_WITH_TRAINING" = "true" ]; then
    # Respect caller-provided server node count (defaults to 0 if unset/invalid)
    RESERVED_NODES=$VAR_SERVER_WITH_TRAINING_NODES
    if ! [[ "$RESERVED_NODES" =~ ^[0-9]+$ ]] || [ "$RESERVED_NODES" -le 0 ]; then
        RESERVED_NODES=0
    fi
    NNODES=$((NNODES - RESERVED_NODES))
    if [ "$NNODES" -lt 0 ]; then
        NNODES=0
    fi
    echo "Server-with-training enabled; reserving $RESERVED_NODES node(s). NNODES: $NNODES"
fi

VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}
if [ "$VAL_BEFORE_TRAIN" = "true" ]; then
    echo "VAL_BEFORE_TRAIN: $VAL_BEFORE_TRAIN"
fi
echo "VAL_BEFORE_TRAIN: $VAL_BEFORE_TRAIN"

COVERAGE_REWARD_TYPE=${COVERAGE_REWARD_TYPE:-"time_coverage"}
COVERAGE_REWARD_WEIGHT=${COVERAGE_REWARD_WEIGHT:-0.25}
COVERAGE_REWARD_ENABLE=${COVERAGE_REWARD_ENABLE:-False}

ENABLE_TWO_GATE_FILTER=${ENABLE_TWO_GATE_FILTER:-False}
GATE1_BIAS_EPSILON=${GATE1_BIAS_EPSILON:-0.01}
GATE2_INSTABILITY_THRESHOLD=${GATE2_INSTABILITY_THRESHOLD:--15.0}
LOG_REJECTED_SAMPLES=${LOG_REJECTED_SAMPLES:-False}
SAVE_REJECTION_STATS=${SAVE_REJECTION_STATS:-True}

ENABLE_MULTI_TURN=${ENABLE_MULTI_TURN:-True}
MAX_TURN=${MAX_TURN:-3}
VAL_MAX_TURN=${VAL_MAX_TURN:-$MAX_TURN}

GAMMA=${GAMMA:-1.0}

DETECT_DECOY_KERNEL=${DETECT_DECOY_KERNEL:-True}


IS_GET_LAST_TURN=${IS_GET_LAST_TURN:-False}
if [ "$IS_GET_LAST_TURN" = "true" ]; then
    echo "IS_GET_LAST_TURN: $IS_GET_LAST_TURN"
fi
echo "IS_GET_LAST_TURN: $IS_GET_LAST_TURN"

ADV_BY_LAST_TURN=${ADV_BY_LAST_TURN:-False}
USE_FINAL_REWARD=${USE_FINAL_REWARD:-False}
# Default values - can be overridden by sourcing scripts or command-line args
# Batch and validation settings
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-512}  # Number of samples per training batch
VAL_SAMPLE_SIZE=${VAL_SAMPLE_SIZE:-100}    # Number of validation samples to evaluate
N_VAL=${N_VAL:-16}                         # Number of generations per validation prompt
VAL_TEMPERATURE=${VAL_TEMPERATURE:-1.0}     # Temperature for validation sampling
VAL_DO_SAMPLE=${VAL_DO_SAMPLE:-True}       # Whether to sample during validation
REWARD_SHAPING=${REWARD_SHAPING:-False}
UNBIASED_SHAPING=${UNBIASED_SHAPING:-True}
# Loss aggregation configuration (critical for training dynamics)
# seq-mean-token-sum: Sum loss per sequence, then average across sequences (recommended)
# token-mean: Average across all tokens (can be length-biased)
# seq-mean-token-mean: Mean per sequence, then average (length-normalized)
# seq-mean-token-sum-norm: Sum per sequence, normalize by max_seq_length
# seq-sum-no-norm: Raw sum across all tokens (no normalization)
LOSS_AGG_MODE=${LOSS_AGG_MODE:-"seq-mean-token-sum"}
# Scale factor to prevent gradient underflow in long sequences
# Default 1000.0 is calibrated for ~10k mean response length (16k max)
# Adjust based on your sequence lengths:
#   - 1.0-100.0: Short sequences (<1k tokens)
#   - 1000.0: Medium sequences (~10k tokens) [DEFAULT]
#   - 10000.0+: Very long sequences (>20k tokens)
# Rule of thumb: loss_scale_factor ≈ mean_response_length / 10
LOSS_SCALE_FACTOR=${LOSS_SCALE_FACTOR:-1000.0}

REWARD_FUNC_NAME=${REWARD_FUNC_NAME:-"calculate_reward_weighted"}
SPEEDUP_REWARD_UPPER_BOUND=${SPEEDUP_REWARD_UPPER_BOUND:-3.0}
SPEEDUP_REWARD_LOWER_BOUND=${SPEEDUP_REWARD_LOWER_BOUND:-1.0}

ROLLOUT_RS=${ROLLOUT_RS:-null}
ROLLOUT_IS=${ROLLOUT_IS:-null}
ROLLOUT_TOKEN_VETO_THRESHOLD=${ROLLOUT_TOKEN_VETO_THRESHOLD:-null}
ROLLOUT_IS_KWARGS=${ROLLOUT_IS_KWARGS:-"{upper:2.0}"}
ROLLOUT_RS_KWARGS=${ROLLOUT_RS_KWARGS:-"{lower:0.5,upper:2.0}"}

# Coverage-based rejection sampling configuration
# - COVERAGE_RS: Aggregation level ("turn" or "geometric", null to disable)
# - COVERAGE_RS_THRESHOLD: Lower threshold for coverage ratio (default: 0.3)
# - COVERAGE_RS_FACTOR: Factor for probability calculation (default: 0.1)
# - COVERAGE_RS_KEY: Which coverage metric to use ("time_coverage" or "num_coverage")
# - SPEEDUP_THRESHOLD: Optional speedup threshold for OR logic (null to disable)
#   If set, samples are kept if EITHER coverage OR speedup meets threshold
#   Example: 1.5 means keep samples with >=1.5x speedup regardless of coverage
COVERAGE_RS=${COVERAGE_RS:-null}
COVERAGE_RS_THRESHOLD=${COVERAGE_RS_THRESHOLD:-0.3}
COVERAGE_RS_FACTOR=${COVERAGE_RS_FACTOR:-0.1}
COVERAGE_RS_KEY=${COVERAGE_RS_KEY:-"time_coverage"}
SPEEDUP_THRESHOLD=${SPEEDUP_THRESHOLD:-null}

MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096}
LEARNING_RATE=${LEARNING_RATE:-1e-6}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}     # Mini-batch size for PPO updates
PPO_MICRO_TOKEN=${PPO_MICRO_TOKEN:-null}           # Auto-calculated based on model size
# Dual-clip PPO parameters (low_high format)
# Standard PPO uses symmetric clipping (e.g., 0.2_0.2)
# Dual-clip uses asymmetric clipping to handle negative advantages better
CLIP_RATIO=${CLIP_RATIO:-0.2_0.28}
# Entropy clipping: 0.0 = disabled, 0.8 = skip training on 80% lowest entropy tokens
ENTROPY_CLIP_RATE=${ENTROPY_CLIP_RATE:-0.0}
# Gradient clipping by norm (critical for training stability)
GRAD_CLIP=${GRAD_CLIP:-1.0}
# vLLM Importance Sampling Correction
# Threshold for truncating importance weights when using vLLM rollouts
# Default 2.0 = conservative, null = disabled, 3.0 = balanced, 5.0 = moderate, 10.0 = aggressive
VLLM_IS_THRESHOLD=${VLLM_IS_THRESHOLD:-2.0}
# Extreme Risk Token Masking (for negative advantage trajectories)
# Masks tokens with π < threshold AND negative advantages to prevent gradient explosion
# Default null = disabled, 1e-5 = aggressive, 1e-6 = balanced, 1e-7 = conservative
EXTREME_RISK_PROB_THRESHOLD=${EXTREME_RISK_PROB_THRESHOLD:-null}

KL_LOSS_COEF=${KL_LOSS_COEF:-0.0}
ENTROPY_COEFFIENT=${ENTROPY_COEFFIENT:-0.0}
KL_LOSS_TYPE=${KL_LOSS_TYPE:-"low_var_kl"}
TEMPERATURE=${TEMPERATURE:-1.0}
MIN_P=${MIN_P:-0.0}
TOP_P=${TOP_P:-1.0}
TOP_K=${TOP_K:--1}
ROLLOUT_N=${ROLLOUT_N:-16}
KL_COEF=${KL_COEF:-0.0}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1000}
ROLLOUT_GPU_MEMORY_UTIL=${ROLLOUT_GPU_MEMORY_UTIL:-0.75}
ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-False}
ACTOR_PARAMETER_OFFLOAD=${ACTOR_PARAMETER_OFFLOAD:-False}
MODEL_NAME=${MODEL_NAME:-Qwen3-8B-Base}
SAVE_FREQ=${SAVE_FREQ:-10}
TEST_FREQ=${TEST_FREQ:-10}
REMOVE_CLIP=${REMOVE_CLIP:-False}
ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=${ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE:-1}
FREE_CACHE_ENGINE=${FREE_CACHE_ENGINE:-True}
ENFORCE_EAGER=${ENFORCE_EAGER:-False}
APPLY_CHAT_TEMPLATE=${APPLY_CHAT_TEMPLATE:-True}
REJECTION_SAMPLE=${REJECTION_SAMPLE:-True}
SP_SIZE=${SP_SIZE:-1}
OVERSAMPLE_M=${OVERSAMPLE_M:-2}              # Oversampling multiplier for rejection sampling
# Algorithm selection: "grpo" (Group Relative Policy Optimization) or "rloo" (REINFORCE Leave-One-Out)
ALGORITHM=${ALGORITHM:-grpo}
# Batch standardization: Prevents length bias by standardizing advantages
# - False for both GRPO and RLOO (default)
# - Can be set to True if needed for specific use cases
BATCH_STD=${BATCH_STD:-False}
VAL_ONLY=${VAL_ONLY:-False}

# These MUST be set by task-specific scripts
TRAIN_DATASET=${TRAIN_DATASET:-()}          # Array of training dataset paths
VALID_DATASET=${VALID_DATASET:-()}          # Array of validation dataset paths
REWARD_MANAGER=${REWARD_MANAGER:-"kernel_async"}        # Reward manager (e.g., "math", "code", "swe")

# Kernel reward-related defaults (can be overridden by task scripts or CLI)
REWARD_SERVER_URL=${KERNELGYM_SERVER_URL:-""}
REWARD_ENHANCED=${REWARD_ENHANCED:-True}
REWARD_USE_SANDBOX_RATE_LIMIT=${REWARD_USE_SANDBOX_RATE_LIMIT:-True}
REWARD_RATE_LIMIT=${REWARD_RATE_LIMIT:-64}
REWARD_ACQUIRE_TIMEOUT=${REWARD_ACQUIRE_TIMEOUT:-2400}
REWARD_MAX_CONCURRENT=${REWARD_MAX_CONCURRENT:-32}
REWARD_TIMEOUT=${REWARD_TIMEOUT:-1800}
REWARD_MAX_RETRIES=${REWARD_MAX_RETRIES:-3}
REWARD_TASK_TIMEOUT=${REWARD_TASK_TIMEOUT:-600}
REWARD_TASK_TIMEOUT_CLIENT=${REWARD_TASK_TIMEOUT_CLIENT:-2400}
REWARD_PRINT_STATUS=${REWARD_PRINT_STATUS:-True}
NUM_PERF_TRIALS=${NUM_PERF_TRIALS:-100}

# Optional dump directories
ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-""}
VALIDATION_DATA_DIR=${VALIDATION_DATA_DIR:-""}

# advanced sampling settings, but it requires the dataset has "solve_rate" column
USE_PRIORITIZED_SAMPLING=${USE_PRIORITIZED_SAMPLING:-False}
AUTOMATIC_OVERSAMPLING=${AUTOMATIC_OVERSAMPLING:-False} # close the automatic oversampling for now
USE_MODERATE_SAMPLING=${USE_MODERATE_SAMPLING:-False}
USE_REFRESH_SAMPLING=${USE_REFRESH_SAMPLING:-False}
SOLVERATE_RATIO=${SOLVERATE_RATIO:-0.1_0.9}
SOLVERATE_MEAN_STD=${SOLVERATE_MEAN_STD:-0.5_0.2}
# Oversampling configuration (compensate for rejection)
PROMPT_OVERSAMPLING_FACTOR=${PROMPT_OVERSAMPLING_FACTOR:-2.0}  # 2.0 recommended with two-gate
SAMPLE_OVERSAMPLING_FACTOR=${SAMPLE_OVERSAMPLING_FACTOR:-1.5}  # 1.5 recommended with two-gate
MAX_SKIP_STEPS=${MAX_SKIP_STEPS:-10}
FIX_QWEN3_CHAT_TEMPLATE=${FIX_QWEN3_CHAT_TEMPLATE:-False}

# Sample selection and group management
SAMPLE_SELECTION_STRATEGY=${SAMPLE_SELECTION_STRATEGY:-efficiency_stochastic}  # Better exploration

# setup rollout mode
ROLLOUT_MODE=${ROLLOUT_MODE:-"async_vllm"}

CALCULATE_LOG_PROBS=${CALCULATE_LOG_PROBS:-True}

generate_model_micro_token() {
  local model_name=$1

  if [ "$PPO_MICRO_TOKEN" = "null" ]; then
    # Extract the model size (e.g., 7B, 14B, 32B) using regex
    if [[ $model_name =~ ([0-9]+)B ]]; then
      local model_size="${BASH_REMATCH[1]}"

      # Set the basic config based on model size
      local micro_token_config
      case $model_size in
        3)
          micro_token_config=16384
          ;;
        7)
          micro_token_config=8192
          ;;
        14)
          micro_token_config=4096
          ;;
        24)
          micro_token_config=3072
          ;;
        32)
          micro_token_config=2048
          ;;
        *)
          micro_token_config=16384
          ;;
      esac

      # if you use tensor parallel, you can increase the micro token number
      if [ "$ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE" -gt 1 ]; then
          micro_token_config=$((micro_token_config * ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE))
      fi

      echo $micro_token_config
    else
      echo 16384
    fi
  else
    echo $PPO_MICRO_TOKEN
  fi
}

generate_short_hash() {
  local input_string="$1"
  local hash=$(echo -n "$input_string" | sha256sum | cut -c1-8)
  echo "$hash"
}

generate_suffix() {
  local suffix=""

  while [[ "$#" -gt 0 ]]; do
    case $1 in
      --train_batch_size) suffix+="_batch$2"; shift 2 ;;
      --max_prompt_length) suffix+="_maxprom$2"; shift 2 ;;
      --max_response_length) suffix+="_maxresp$2"; shift 2 ;;
      --learning_rate) suffix+="_lr$2"; shift 2 ;;
      --ppo_mini_batch_size) suffix+="_ppomini$2"; shift 2 ;;
      --kl_loss_coef) suffix+="_klloss$2"; shift 2 ;;
      --entropy_coeffient) suffix+="_entropy$2"; shift 2 ;;
      --clip_ratio) suffix+="_clip$2"; shift 2 ;;
      --remove_clip) suffix+="_rmclip$2"; shift 1 ;;
      --kl_loss_type) suffix+="_kltype$2"; shift 2 ;;
      --temperature) suffix+="_temp$2"; shift 2 ;;
      --top_p) suffix+="_topp$2"; shift 2 ;;
      --top_k) suffix+="_topk$2"; shift 2 ;;
      --min_p) suffix+="_minp$2"; shift 2 ;;
      --rollout_n) suffix+="_rolln$2"; shift 2 ;;
      --rollout_mode) suffix+="_rollmode$2"; shift 2 ;;
      --oversample_multiplier) suffix+="_oversamp$2"; shift 2 ;;
      --kl_coef) suffix+="_klcoef$2"; shift 2 ;;
      --use_prioritized_sampling) suffix+="_prior$2"; shift 2 ;;
      --automatic_oversampling) suffix+="_autoover$2"; shift 2 ;;
      --use_moderate_sampling) suffix+="_moderate$2"; shift 2 ;;
      --use_refresh_sampling) suffix+="_refresh$2"; shift 2 ;;
      --solverate_ratio) suffix+="_solveratio$2"; shift 2 ;;
      --solverate_mean_std) suffix+="_solvemean$2"; shift 2 ;;
      --entropy_clip_rate) suffix+="_entclip$2"; shift 2 ;;
      --loss_agg_mode) suffix+="_lossagg$2"; shift 2 ;;
      --loss_scale_factor) suffix+="_scale$2"; shift 2 ;;
      --grad_clip) suffix+="_gradclip$2"; shift 2 ;;
      --batch_std) suffix+="_batchstd$2"; shift 2 ;;
      --val_only) suffix+="_val_only$2"; shift 2 ;;
      --enable_multi_turn) suffix+="_multiturn$2"; shift 2 ;;
      --max_turn) suffix+="_maxturn$2"; shift 2 ;;
      --val_before_train) suffix+="_valbefore$2"; shift 2 ;;
      --is_get_last_turn) suffix+="_isgetlastturn$2"; shift 2 ;;
            *) shift ;;
    esac
  done

  local suffix_hash=$(generate_short_hash "$suffix")
  echo "_$suffix_hash"
}

show_help() {
  echo "RL Training Script (GRPO/RLOO)"
  echo ""
  echo "Usage: $0 [OPTIONS]"
  echo ""
  echo "Loss Aggregation Options:"
  echo "  --loss_agg_mode MODE          Loss aggregation mode (default: seq-mean-token-sum)"
  echo "                                Modes: token-mean, seq-mean-token-sum, seq-mean-token-mean,"
  echo "                                       seq-mean-token-sum-norm, seq-sum-no-norm"
  echo "  --loss_scale_factor FACTOR    Loss scaling factor (default: 1.0)"
  echo "                                Use 100.0 for 10k+ token sequences"
  echo ""
  echo "Training Options:"
  echo "  --learning_rate RATE          Learning rate (default: 1e-6)"
  echo "  --train_batch_size SIZE       Training batch size (default: 512)"
  echo "  --max_response_length LENGTH  Max response length (default: 4096)"
  echo "  --entropy_clip_rate RATE      Entropy clipping rate (default: 0.0)"
  echo "  --grad_clip VALUE             Gradient clipping value (default: 1.0)"
  echo "  --extreme_risk_prob_threshold Extreme risk token masking threshold (default: null)"
  echo "                                null=disabled, 1e-5=aggressive, 1e-6=balanced, 1e-7=conservative"
  echo "  --batch_std TRUE/FALSE        Batch standardization (default: False, True for RLOO)"
  echo "  --model_name NAME             Model name (default: Qwen3-8B-Base)"
  echo "  --rollout_is_kwargs KEY=VALUE Additional IS kwargs (default: {})"
  echo "  --rollout_rs_kwargs KEY=VALUE Additional RS kwargs (default: {})"
  echo ""
  echo "Examples:"
  echo "  $0                                    # Standard training"
  echo "  $0 --loss_scale_factor 100.0         # Long sequence training"
  echo "  $0 --loss_agg_mode token-mean        # Token-level aggregation"
  echo "  $0 --enable_multi_turn True         # Enable multi-turn training"
  echo "  $0 --max_turn 3                      # Maximum number of turns"
  echo ""
}

parse_arguments() {
  echo "Arguments received: $@"

  # Check for help request
  for arg in "$@"; do
    if [[ "$arg" == "--help" || "$arg" == "-h" ]]; then
      show_help
      exit 0
    fi
  done
  # Generate a unique suffix based on the input arguments
  SUFFIX=$(generate_suffix "$@")
  RUN_NAME="$RUN_NAME$SUFFIX"
  echo "RUN_NAME: $RUN_NAME"

  # Parse named arguments
  while [[ "$#" -gt 0 ]]; do
    echo "Processing: $1"
    case "$1" in
      --train_batch_size) TRAIN_BATCH_SIZE="$2"; shift 2 ;;
      --val_sample_size) VAL_SAMPLE_SIZE="$2"; shift 2 ;;
      --max_prompt_length) MAX_PROMPT_LENGTH="$2"; shift 2 ;;
      --max_response_length) MAX_RESPONSE_LENGTH="$2"; shift 2 ;;
      --learning_rate) LEARNING_RATE="$2"; shift 2 ;;
      --ppo_mini_batch_size) PPO_MINI_BATCH_SIZE="$2"; shift 2 ;;
      --ppo_micro_token) PPO_MICRO_TOKEN="$2"; shift 2 ;;
      --kl_loss_coef) KL_LOSS_COEF="$2"; shift 2 ;;
      --entropy_coeffient) ENTROPY_COEFFIENT="$2"; shift 2 ;;
      --clip_ratio) CLIP_RATIO="$2"; shift 2 ;;
      --kl_loss_type) KL_LOSS_TYPE="$2"; shift 2 ;;
      --vllm_is_threshold) VLLM_IS_THRESHOLD="$2"; shift 2 ;;
      --extreme_risk_prob_threshold) EXTREME_RISK_PROB_THRESHOLD="$2"; shift 2 ;;
      --temperature) TEMPERATURE="$2"; shift 2 ;;
      --oversample_multiplier) OVERSAMPLE_M="$2"; shift 2 ;;
      --top_p) TOP_P="$2"; shift 2 ;;
      --top_k) TOP_K="$2"; shift 2 ;;
      --min_p) MIN_P="$2"; shift 2 ;;
      --rollout_n) ROLLOUT_N="$2"; shift 2 ;;
      --rollout_mode) ROLLOUT_MODE="$2"; shift 2 ;;
      --n_val) N_VAL="$2"; shift 2 ;;
      --val_temperature) VAL_TEMPERATURE="$2"; shift 2 ;;
      --val_do_sample) VAL_DO_SAMPLE="$2"; shift 2 ;;
      --rollout_gpu_memory_util) ROLLOUT_GPU_MEMORY_UTIL="$2"; shift 2 ;;
      --rollout_tp) ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE="$2"; shift 2 ;;
      --kl_coef) KL_COEF="$2"; shift 2 ;;
      --actor_optimizer_offload) ACTOR_OPTIMIZER_OFFLOAD="$2"; shift 2 ;;
      --actor_parameter_offload) ACTOR_PARAMETER_OFFLOAD="$2"; shift 2 ;;
      --total_epochs) TOTAL_EPOCHS="$2"; shift 2 ;;
      --save_freq) SAVE_FREQ="$2"; shift 2 ;;
      --test_freq) TEST_FREQ="$2"; shift 2 ;;
      --remove_clip) REMOVE_CLIP="$2"; shift 2 ;;
      --apply_chat_template) APPLY_CHAT_TEMPLATE="$2"; shift 2 ;;
      --rejection_sample) REJECTION_SAMPLE="$2"; shift 2 ;;
      --sp_size) SP_SIZE="$2"; shift 2 ;;
      --train_dataset) TRAIN_DATASET=($2); shift 2 ;;
      --valid_dataset) VALID_DATASET=($2); shift 2 ;;
      --model_name) MODEL_NAME="$2"; shift 2 ;;
      --use_prioritized_sampling) USE_PRIORITIZED_SAMPLING="$2"; shift 2 ;;
      --automatic_oversampling) AUTOMATIC_OVERSAMPLING="$2"; shift 2 ;;
      --use_moderate_sampling) USE_MODERATE_SAMPLING="$2"; shift 2 ;;
      --use_refresh_sampling) USE_REFRESH_SAMPLING="$2"; shift 2 ;;
      --solverate_ratio) SOLVERATE_RATIO="$2"; shift 2 ;;
      --solverate_mean_std) SOLVERATE_MEAN_STD="$2"; shift 2 ;;
      --reward_manager) REWARD_MANAGER="$2"; shift 2 ;;
      --reward_server_url) REWARD_SERVER_URL="$2"; shift 2 ;;
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
      --reward_policy) REWARD_POLICY="$2"; shift 2 ;;
      --rollout_data_dir) ROLLOUT_DATA_DIR="$2"; shift 2 ;;
      --validation_data_dir) VALIDATION_DATA_DIR="$2"; shift 2 ;;
      --entropy_clip_rate) ENTROPY_CLIP_RATE="$2"; shift 2 ;;
      --loss_agg_mode) LOSS_AGG_MODE="$2"; shift 2 ;;
      --loss_scale_factor) LOSS_SCALE_FACTOR="$2"; shift 2 ;;
      --grad_clip) GRAD_CLIP="$2"; shift 2 ;;
      --batch_std) BATCH_STD="$2"; shift 2 ;;
      --val_only) VAL_ONLY="$2"; shift 2 ;;
      --cal_log_probs) CALCULATE_LOG_PROBS="$2"; shift 2 ;;
      --max_skip_steps) MAX_SKIP_STEPS="$2"; shift 2 ;;
      --fix_qwen3_chat_template) FIX_QWEN3_CHAT_TEMPLATE="$2"; shift 2 ;;
      --adv_by_last_turn) ADV_BY_LAST_TURN="$2"; shift 2 ;;
      --use_final_reward) USE_FINAL_REWARD="$2"; shift 2 ;;
      --rollout_is) ROLLOUT_IS="$2"; shift 2 ;;
      --rollout_rs) ROLLOUT_RS="$2"; shift 2 ;;
      --rollout_is_kwargs) ROLLOUT_IS_KWARGS="$2"; shift 2 ;;
      --rollout_rs_kwargs) ROLLOUT_RS_KWARGS="$2"; shift 2 ;;
      --rollout_token_veto_threshold) ROLLOUT_TOKEN_VETO_THRESHOLD="$2"; shift 2 ;;
      --enable_multi_turn) ENABLE_MULTI_TURN="$2"; shift 2 ;;
      --max_turn) MAX_TURN="$2"; shift 2 ;;
      --val_before_train) VAL_BEFORE_TRAIN="$2"; shift 2 ;;
      --is_get_last_turn) IS_GET_LAST_TURN="$2"; shift 2 ;;
      --speedup_reward_upper_bound) SPEEDUP_REWARD_UPPER_BOUND="$2"; shift 2 ;;
      --speedup_reward_lower_bound) SPEEDUP_REWARD_LOWER_BOUND="$2"; shift 2 ;;
      --reward_shaping) REWARD_SHAPING="$2"; shift 2 ;;
      --unbiased_shaping) UNBIASED_SHAPING="$2"; shift 2 ;;
      --gamma) GAMMA="$2"; shift 2 ;;
      *)
        echo "Unknown option: $1"
        exit 1
        ;;
    esac
  done
}

parse_clip_ratio() {
    local clip_ratio="$1"
    local low
    local high

    # Check if the clip_ratio is a single number (e.g., "0.2") or in the format "number_number" (e.g., "0.2_0.3")
    if [[ $clip_ratio =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
        low=$clip_ratio
        high=$clip_ratio
    elif [[ $clip_ratio =~ ^[0-9]+(\.[0-9]+)?_[0-9]+(\.[0-9]+)?$ ]]; then
        # If it's in the "number_number" format, split by underscore
        low=$(echo $clip_ratio | cut -d'_' -f1)
        high=$(echo $clip_ratio | cut -d'_' -f2)
    else
        # Print a warning if the format is incorrect
        echo "Warning: clip_ratio '$clip_ratio' is not in the correct format (e.g., '0.2_0.3' or '0.2')."
        return 1
    fi

    CLIP_RATIO_LOW=$low
    CLIP_RATIO_HIGH=$high
    echo "CLIP_RATIO_LOW: $CLIP_RATIO_LOW"
    echo "CLIP_RATIO_HIGH: $CLIP_RATIO_HIGH"
}

parse_solve_ratio() {
    local solve_ratio="$1"
    local low
    local high

    # Check if the solve_ratio is a single number (e.g., "0.1") or in the format "number_number" (e.g., "0.1_0.9")
    if [[ $solve_ratio =~ ^[0-9]+(\.[0-9]+)?_[0-9]+(\.[0-9]+)?$ ]]; then
        # If it's in the "number_number" format, split by underscore
        low=$(echo $solve_ratio | cut -d'_' -f1)
        high=$(echo $solve_ratio | cut -d'_' -f2)
    else
        # Print a warning if the format is incorrect
        echo "Warning: solve_ratio '$solve_ratio' is not in the correct format (e.g., '0.1_0.9')."
        return 1
    fi

    SOLVERATE_LOW=$low
    SOLVERATE_HIGH=$high
    echo "SOLVERATE_LOW: $SOLVERATE_LOW"
    echo "SOLVERATE_HIGH: $SOLVERATE_HIGH"
}

parse_solve_mean_std() {
    local solve_mean_std="$1"
    local mean
    local std

    # Check if the solve_mean_std is in the format "mean_std" (e.g., "0.5_0.1")
    if [[ $solve_mean_std =~ ^[0-9]+(\.[0-9]+)?_[0-9]+(\.[0-9]+)?$ ]]; then
        # If it's in the "mean_std" format, split by underscore
        mean=$(echo $solve_mean_std | cut -d'_' -f1)
        std=$(echo $solve_mean_std | cut -d'_' -f2)
    else
        # Print a warning if the format is incorrect
        echo "Warning: solve_mean_std '$solve_mean_std' is not in the correct format (e.g., '0.5_0.1')."
        return 1
    fi

    SOLVERATE_MEAN=$mean
    SOLVERATE_STD=$std
    echo "SOLVERATE_MEAN: $SOLVERATE_MEAN"
    echo "SOLVERATE_STD: $SOLVERATE_STD"
}

parse_chat_scheduler() {
  local rollout_mode="$1"
  local chat_scheduler=null
  local return_raw_chat=False

  case "$rollout_mode" in
    "async_vllm")
      return_raw_chat=True ;;
  esac

  case "$rollout_mode" in
    "async_agent")
      return_raw_chat=True ;;
  esac

  RETURN_RAW_CHAT=$return_raw_chat
  echo "RETURN_RAW_CHAT: $RETURN_RAW_CHAT"
}

format_dataset_paths() {
  local dataset=("$@")
  local formatted_paths=""
  local resolved_path=""

  for dataset_path in "${dataset[@]}"; do
    if [[ "$dataset_path" == /* || "$dataset_path" == ./* || "$dataset_path" == ../* || "$dataset_path" == *.parquet || "$dataset_path" == *.jsonl ]]; then
      resolved_path="$dataset_path"
    elif [[ -n "${HDFS_DATA_PATH}" ]]; then
      resolved_path="${HDFS_DATA_PATH}/${dataset_path}.parquet"
    else
      resolved_path="${dataset_path}.parquet"
    fi
    formatted_paths+="\"${resolved_path}\","
  done

  # Remove the last comma
  formatted_paths="${formatted_paths%,}"

  echo "[$formatted_paths]"
}

setup_training_environment() {
  # Build dataset name string for run name
  if [ ${#TRAIN_DATASET[@]} -gt 0 ]; then
    for dataset in "${TRAIN_DATASET[@]}"; do
      train_dataset_str+="_$(echo $dataset | sed 's/\//_/g')"
    done
  fi

  # for KL_LOSS_COEF
  if (( $(echo "$KL_LOSS_COEF == 0" | bc -l) )); then
    USE_KL_LOSS=False
  else
    USE_KL_LOSS=True
  fi
  echo "Use KL Loss: $USE_KL_LOSS"

  # for KL_COEF
  if (( $(echo "$KL_COEF == 0" | bc -l) )); then
    USE_KL_COEF=False
  else
    USE_KL_COEF=True
  fi
  echo "Use KL Coef: $USE_KL_COEF"

  RUN_NAME+="$train_dataset_str"
  RUN_NAME+="_$MODEL_NAME"

  if [[ -n "${MODEL_PATH:-}" ]]; then
    MODEL_PATH_RESOLVED="$MODEL_PATH"
  elif [[ -n "${HDFS_MODEL_PATH}" ]]; then
    MODEL_PATH_RESOLVED="${HDFS_MODEL_PATH}/${MODEL_NAME}"
  else
    MODEL_PATH_RESOLVED="$MODEL_NAME"
  fi

  if [[ -n "${HDFS_CHECKPOINT_PATH}" ]]; then
    CHECKPOINT_DIR="${HDFS_CHECKPOINT_PATH}/${RUN_NAME}"
  else
    CHECKPOINT_DIR="checkpoints/${RUN_NAME}"
  fi

  N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-${GPUS_PER_NODE:-${ARNOLD_WORKER_GPU:-8}}}

  export RUN_NAME
  export MODEL_PATH_RESOLVED
  export CHECKPOINT_DIR
  export N_GPUS_PER_NODE
  echo "FULL RUN_NAME: $RUN_NAME"
  echo "Training with the following parameters:"
  echo "Train Batch Size: $TRAIN_BATCH_SIZE"
  echo "Max Prompt Length: $MAX_PROMPT_LENGTH"
  echo "Max Response Length: $MAX_RESPONSE_LENGTH"
  echo "Learning Rate: $LEARNING_RATE"
  echo "PPO Mini Batch Size: $PPO_MINI_BATCH_SIZE"
  echo "KL Loss Coefficient: $KL_LOSS_COEF"
  echo "KL Loss Type: $KL_LOSS_TYPE"
  echo "Temperature: $TEMPERATURE"
  echo "Rollout N: $ROLLOUT_N"
  echo "Rollout Mode: $ROLLOUT_MODE"
  echo "KL Coefficient: $KL_COEF"
  echo "Total Epochs: $TOTAL_EPOCHS"
  echo "Model Name: $MODEL_NAME"
  echo "Model Path: $MODEL_PATH_RESOLVED"
  echo "Checkpoint Dir: $CHECKPOINT_DIR"
  echo "GPUs per Node: $N_GPUS_PER_NODE"
  echo "Remove Clip: $REMOVE_CLIP"
  echo "Reward Manager: $REWARD_MANAGER"
  echo "Automatic Oversampling: $AUTOMATIC_OVERSAMPLING"
  echo "Moderate Sampling: $USE_MODERATE_SAMPLING"
  echo "Refresh Sampling: $USE_REFRESH_SAMPLING"
  echo "Solverate Ratio: $SOLVERATE_RATIO"
  echo "Solverate Mean Std: $SOLVERATE_MEAN_STD"
  echo "Entropy Clip Rate: $ENTROPY_CLIP_RATE"
  echo "Gradient Clip: $GRAD_CLIP"
  echo "Val Only: $VAL_ONLY"
  echo "Fix Qwen3 Chat Template: $FIX_QWEN3_CHAT_TEMPLATE"

  # set ppo micro token
  PPO_MICRO_TOKEN=$(generate_model_micro_token "$MODEL_NAME")
  echo "PPO_MICRO_TOKEN: $PPO_MICRO_TOKEN"
  LOG_PROB_MICRO_TOKEN=$((PPO_MICRO_TOKEN * 2))
  max_num_batched_tokens=$(expr $MAX_PROMPT_LENGTH + $MAX_RESPONSE_LENGTH + 1000)

  # calculate the sum of MAX_PROMPT_LENGTH and MAX_RESPONSE_LENGTH
  required_token_length=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))

  # if sp_size is greater than 1, we can increase the total_micro_token
  total_micro_token=$((PPO_MICRO_TOKEN * SP_SIZE))
  # verify if PPO_MICRO_TOKEN is less than required_token_length, if less then directly exit
  if [ "$total_micro_token" -lt "$required_token_length" ]; then
      echo "Warning: PPO_MICRO_TOKEN is less than the required token length ($required_token_length)."
      echo "Please try increasing your rollout_tp, otherwise the script cannot be run using your current model."
      exit 1
  fi

  parse_clip_ratio "$CLIP_RATIO"
  parse_solve_ratio "$SOLVERATE_RATIO"
  parse_solve_mean_std "$SOLVERATE_MEAN_STD"
  parse_chat_scheduler "$ROLLOUT_MODE"

  TRAIN_FILES=$(format_dataset_paths "${TRAIN_DATASET[@]}")
  VALID_FILES=$(format_dataset_paths "${VALID_DATASET[@]}")
  echo "TRAIN_FILES: $TRAIN_FILES"
  echo "VALID_FILES: $VALID_FILES"
}

run_training() {
  sleep 3

  PYTHONUNBUFFERED=1 python -m kernel.main_kernel \
      trainer.val_before_train=$VAL_BEFORE_TRAIN \
      algorithm.adv_estimator=$ALGORITHM \
      algorithm.is_get_last_turn=$IS_GET_LAST_TURN \
      data.train_files=$TRAIN_FILES \
      data.val_files=$VALID_FILES \
      data.return_raw_chat=$RETURN_RAW_CHAT \
      data.train_batch_size=$TRAIN_BATCH_SIZE \
      data.val_sample_size=$VAL_SAMPLE_SIZE \
      data.max_prompt_length=$MAX_PROMPT_LENGTH \
      data.max_response_length=$MAX_RESPONSE_LENGTH \
      data.apply_chat_template=$APPLY_CHAT_TEMPLATE \
      data.use_prioritized_sampling=$USE_PRIORITIZED_SAMPLING \
      data.update_success_rates_every=1 \
      data.prompt_oversampling_factor=$PROMPT_OVERSAMPLING_FACTOR \
      data.sample_oversampling_factor=$SAMPLE_OVERSAMPLING_FACTOR \
      data.sample_selection_strategy=$SAMPLE_SELECTION_STRATEGY \
      data.automatic_oversampling=$AUTOMATIC_OVERSAMPLING \
      data.use_moderate_sampling=$USE_MODERATE_SAMPLING \
      data.use_refresh_sampling=$USE_REFRESH_SAMPLING \
      data.solverate_low=$SOLVERATE_LOW \
      data.solverate_high=$SOLVERATE_HIGH \
      data.solverate_mean=$SOLVERATE_MEAN \
      data.solverate_std=$SOLVERATE_STD \
      trainer.fix_qwen3_chat_template=$FIX_QWEN3_CHAT_TEMPLATE \
      +algorithm.rollout_is_kwargs=$ROLLOUT_IS_KWARGS \
      +algorithm.rollout_rs_kwargs=$ROLLOUT_RS_KWARGS \
      algorithm.rollout_rs=$ROLLOUT_RS \
      algorithm.rollout_token_veto_threshold=$ROLLOUT_TOKEN_VETO_THRESHOLD \
      actor_rollout_ref.rollout.multi_turn.enable=$ENABLE_MULTI_TURN \
      actor_rollout_ref.rollout.multi_turn.max_user_turns=$MAX_TURN \
      actor_rollout_ref.model.path=$MODEL_PATH_RESOLVED \
      actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
      actor_rollout_ref.model.use_remove_padding=True \
      actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
      actor_rollout_ref.actor.use_dynamic_bsz=True \
      actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MICRO_TOKEN \
      actor_rollout_ref.actor.use_kl_loss=$USE_KL_LOSS \
      actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
      actor_rollout_ref.actor.kl_loss_type=$KL_LOSS_TYPE \
      actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFFIENT \
      actor_rollout_ref.actor.clip_ratio_high=$CLIP_RATIO_HIGH \
      actor_rollout_ref.actor.clip_ratio_low=$CLIP_RATIO_LOW \
      actor_rollout_ref.actor.entropy_clip_rate=$ENTROPY_CLIP_RATE \
      actor_rollout_ref.actor.loss_agg_mode=$LOSS_AGG_MODE \
      actor_rollout_ref.actor.loss_scale_factor=$LOSS_SCALE_FACTOR \
      actor_rollout_ref.actor.extreme_risk_prob_threshold=$EXTREME_RISK_PROB_THRESHOLD \
      actor_rollout_ref.actor.grad_clip=$GRAD_CLIP \
      actor_rollout_ref.model.enable_gradient_checkpointing=True \
      actor_rollout_ref.actor.fsdp_config.param_offload=$ACTOR_PARAMETER_OFFLOAD \
      actor_rollout_ref.actor.fsdp_config.optimizer_offload=$ACTOR_OPTIMIZER_OFFLOAD \
      actor_rollout_ref.actor.ulysses_sequence_parallel_size=$SP_SIZE \
      actor_rollout_ref.rollout.enforce_eager=$ENFORCE_EAGER \
      actor_rollout_ref.rollout.free_cache_engine=$FREE_CACHE_ENGINE \
      actor_rollout_ref.rollout.temperature=$TEMPERATURE \
      actor_rollout_ref.rollout.top_p=$TOP_P \
      actor_rollout_ref.rollout.top_k=$TOP_K \
      actor_rollout_ref.rollout.min_p=$MIN_P \
      actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$LOG_PROB_MICRO_TOKEN \
      actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE \
      actor_rollout_ref.rollout.name=vllm \
      actor_rollout_ref.rollout.mode=$ROLLOUT_MODE \
      actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTIL \
      actor_rollout_ref.rollout.n=$ROLLOUT_N \
      actor_rollout_ref.rollout.val_kwargs.n=$N_VAL \
      actor_rollout_ref.rollout.val_kwargs.do_sample=$VAL_DO_SAMPLE \
      actor_rollout_ref.rollout.val_kwargs.temperature=$VAL_TEMPERATURE \
      actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
      actor_rollout_ref.rollout.val_kwargs.max_user_turns=$VAL_MAX_TURN \
      actor_rollout_ref.rollout.max_num_batched_tokens=$max_num_batched_tokens \
      actor_rollout_ref.rollout.calculate_log_probs=$CALCULATE_LOG_PROBS \
      actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$LOG_PROB_MICRO_TOKEN \
      actor_rollout_ref.ref.fsdp_config.param_offload=True \
      actor_rollout_ref.ref.ulysses_sequence_parallel_size=$SP_SIZE\
      reward_model.enable=False \
      reward_model.reward_manager=$REWARD_MANAGER \
      reward_model.enhanced=$REWARD_ENHANCED \
      reward_model.use_sandbox_rate_limit=$REWARD_USE_SANDBOX_RATE_LIMIT \
      reward_model.server_url='"'$REWARD_SERVER_URL'"' \
      reward_model.rate_limit=$REWARD_RATE_LIMIT \
      reward_model.acquire_timeout=$REWARD_ACQUIRE_TIMEOUT \
      reward_model.max_concurrent=$REWARD_MAX_CONCURRENT \
      reward_model.task_timeout=$REWARD_TASK_TIMEOUT \
      reward_model.task_timeout_in_client=$REWARD_TASK_TIMEOUT_CLIENT \
      reward_model.max_retries=$REWARD_MAX_RETRIES \
      reward_model.task_timeout=$REWARD_TASK_TIMEOUT \
      reward_model.num_perf_trials=$NUM_PERF_TRIALS \
      reward_model.print_status=$REWARD_PRINT_STATUS \
      reward_model.reward_func_name=$REWARD_FUNC_NAME \
      reward_model.speedup_reward_upper_bound=$SPEEDUP_REWARD_UPPER_BOUND \
      reward_model.speedup_reward_lower_bound=$SPEEDUP_REWARD_LOWER_BOUND \
      reward_model.coverage_reward.reward_type=$COVERAGE_REWARD_TYPE \
      reward_model.coverage_reward.weight=$COVERAGE_REWARD_WEIGHT \
      reward_model.coverage_reward.enable=$COVERAGE_REWARD_ENABLE \
      reward_model.coverage_rs=$COVERAGE_RS \
      reward_model.coverage_rs_threshold=$COVERAGE_RS_THRESHOLD \
      reward_model.coverage_rs_factor=$COVERAGE_RS_FACTOR \
      reward_model.coverage_rs_key=$COVERAGE_RS_KEY \
      reward_model.speedup_threshold=$SPEEDUP_THRESHOLD \
      reward_model.detect_decoy_kernel=$DETECT_DECOY_KERNEL \
      algorithm.reward_shaping=$REWARD_SHAPING \
      algorithm.unbiased_shaping=$UNBIASED_SHAPING \
      algorithm.adv_estimator=${ALGORITHM:-grpo} \
      algorithm.use_kl_in_reward=$USE_KL_COEF \
      algorithm.kl_ctrl.kl_coef=$KL_COEF \
      algorithm.batch_std=${BATCH_STD:-False} \
      algorithm.adv_by_last_turn=$ADV_BY_LAST_TURN \
      algorithm.use_final_reward=$USE_FINAL_REWARD \
      algorithm.gamma=$GAMMA \
      critic.ppo_micro_batch_size_per_gpu=4 \
      trainer.critic_warmup=0 \
      trainer.logger=['console','wandb'] \
      trainer.rejection_sample=$REJECTION_SAMPLE \
      trainer.project_name=$PROJECT_NAME \
      trainer.experiment_name=$RUN_NAME \
      trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
      trainer.nnodes=$NNODES \
      trainer.remove_clip=$REMOVE_CLIP \
      trainer.rollout_data_dir=$ROLLOUT_DATA_DIR \
      trainer.validation_data_dir=$VALIDATION_DATA_DIR \
      trainer.log_val_generations=10 \
      trainer.save_freq=$SAVE_FREQ \
      trainer.test_freq=$TEST_FREQ \
      trainer.default_local_dir=$CHECKPOINT_DIR \
      trainer.total_epochs=$TOTAL_EPOCHS \
      trainer.val_only=$VAL_ONLY \
      trainer.max_skip_steps=$MAX_SKIP_STEPS \
      rejection_sampling.enable_two_gate_filter=$ENABLE_TWO_GATE_FILTER \
      rejection_sampling.gate1.enabled=$GATE1_ENABLED \
      rejection_sampling.gate1.bias_epsilon=$GATE1_BIAS_EPSILON \
      rejection_sampling.gate2.enabled=$GATE2_ENABLED \
      rejection_sampling.gate2.instability_threshold=$GATE2_INSTABILITY_THRESHOLD \
      rejection_sampling.log_rejected_samples=$LOG_REJECTED_SAMPLES \
      rejection_sampling.save_rejection_stats=$SAVE_REJECTION_STATS
}

# Main execution function
main() {
  parse_arguments "$@"
  setup_training_environment
  run_training
}

# Only show error if this script is executed directly (not sourced)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Error: This script should not be run directly."
  echo "Please use one of the task-specific scripts (train_grpo_math_tune.sh, train_grpo_swe_tune.sh, etc.)"
  exit 1
fi
