
HDFS_LOG_PATH=""  # you need to set your own log path
HDFS_CHECKPOINT_PATH=""  # you need to set your own checkpoint path
HDFS_MODEL_PATH=""  # you need to set your own model path

# Default values
TRAIN_BATCH_SIZE=64
MICRO_BATCH_SIZE_PER_GPU=2
MAX_LENGTH=18432
TOTAL_EPOCHS=4
SAVE_FREQ=50
MODEL_NAME=qwen3-14b-base

DATASET_NAME=hkust-nlp/drkernel-coldstart-8k
TRAIN_FILE_NAME=train_2000
VAL_FILE_NAME=train_2000

RUN_NAME=drkernel-14b-coldstart
PROJECT_NAME=kernel-sft
PROMPT_KEY=prompt
RESPONSE_KEY=response
TRUNCATION=right
LEARNING_RATE=2e-5
SP_SIZE=4
TRAIN_DATA_PATH=${DATASET_NAME} # you need to set your own training data path
NNODES=$ARNOLD_WORKER_NUM
GPUS_PER_NODE=$ARNOLD_WORKER_GPU
if [ -z "$ARNOLD_WORKER_GPU" ]; then
    GPUS_PER_NODE=8
fi

export GPUS_PER_NODE="${GPUS_PER_NODE:-${ARNOLD_WORKER_GPU:-8}}"
export NNODES="${NNODES:-${ARNOLD_WORKER_NUM:-1}}"
export NODE_RANK="${NODE_RANK:-${ARNOLD_ID:-0}}"
export MASTER_ADDR="${MASTER_ADDR:-${ARNOLD_WORKER_0_HOST:-127.0.0.1}}"

MASTER_PORT=$(echo "${ARNOLD_WORKER_0_PORT:-29500}" | cut -d',' -f1)

is_port_in_use() {
  local port=$1
  if [ -z "$port" ]; then
    return 1
  fi

  if command -v lsof >/dev/null 2>&1; then
    lsof -iTCP:$port -sTCP:LISTEN -t >/dev/null 2>&1
  elif command -v ss >/dev/null 2>&1; then
    ss -ltn | awk '{print $4}' | grep -E "[.:]$port$" >/dev/null 2>&1
  else
    netstat -tuln 2>/dev/null | awk '{print $4}' | grep -E "[.:]$port$" >/dev/null 2>&1
  fi
}

case "$MASTER_PORT" in
  ''|*[!0-9]*)
    MASTER_PORT=29500
    ;;
esac

while is_port_in_use "$MASTER_PORT"; do
  echo "MASTER_PORT $MASTER_PORT in use, trying $((MASTER_PORT + 1))"
  MASTER_PORT=$((MASTER_PORT + 1))
done

export MASTER_PORT

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
  echo "Processing: $1"
  case "$1" in
    --train_batch_size) TRAIN_BATCH_SIZE="$2"; shift 2 ;;
    --micro_batch_size_per_gpu) MICRO_BATCH_SIZE_PER_GPU="$2"; shift 2 ;;
    --max_length) MAX_LENGTH="$2"; shift 2 ;;
    --total_epochs) TOTAL_EPOCHS="$2"; shift 2 ;;
    --save_freq) SAVE_FREQ="$2"; shift 2 ;;
    --learning_rate) LEARNING_RATE="$2"; shift 2 ;;
    --model_name) MODEL_NAME="$2"; shift 2 ;;
    --dataset_name) DATASET_NAME="$2"; shift 2 ;;
    --train_file_name) TRAIN_FILE_NAME="$2"; shift 2 ;;
    --val_file_name) VAL_FILE_NAME="$2"; shift 2 ;;
    --train_data_path) TRAIN_DATA_PATH="$2"; shift 2 ;;
    --truncation) TRUNCATION="$2"; shift 2 ;;
    --sp_size) SP_SIZE="$2"; shift 2 ;;
    --nnodes) NNODES="$2"; shift 2 ;;
    --gpus_per_node) GPUS_PER_NODE="$2"; shift 2 ;;
    --overwrite_run_name) OVERWRITE_RUN_NAME="$2"; shift 2 ;;
    --suffix) SUFFIX="$2"; shift 2 ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

if [ -n "$OVERWRITE_RUN_NAME" ]; then
    RUN_NAME="$OVERWRITE_RUN_NAME"
fi

generate_suffix() {
  local suffix=""
  local dataset_provided=false
  local model_provided=false
  local suffix_provided=false

  while [[ "$#" -gt 0 ]]; do
    case $1 in
      --train_batch_size) suffix+="_batch$2"; shift 2 ;;
      --micro_batch_size_per_gpu) suffix+="_micro$2"; shift 2 ;;
      --max_length) suffix+="_maxlen$2"; shift 2 ;;
      --total_epochs) suffix+="_epochs$2"; shift 2 ;;
      --learning_rate) suffix+="_lr$2"; shift 2 ;;
      --train_file_name) suffix+="_trainf$2"; shift 2 ;;
      --val_file_name) suffix+="_valf$2"; shift 2 ;;
      --dataset_name) suffix+="_$2"; dataset_provided=true; shift 2 ;;
      --model_name) suffix+="_$2"; model_provided=true; shift 2 ;;
      --truncation) suffix+="_trunc$2"; shift 2 ;;
      --suffix) input_suffix="$2"; suffix_provided=true; shift 2 ;;
      *) shift ;;
    esac
  done

  if [ "$dataset_provided" = false ]; then
    suffix+="_$DATASET_NAME"
  fi

  if [ "$model_provided" = false ]; then
    suffix+="_$MODEL_NAME"
  fi

  if [ "$suffix_provided" = true ]; then
    suffix+="_$input_suffix"
  fi

  echo "$suffix"
}

echo "Arguments received: $@"

# Generate a unique suffix based on the input arguments
SUFFIX=$(generate_suffix "$@")
RUN_NAME="$RUN_NAME$SUFFIX"
# LOG_FILE_PATH="$HDFS_LOG_PATH/$RUN_NAME.log"
mkdir -p ./logs
LOG_FILE_PATH=./logs/$RUN_NAME.log

echo "RUN_NAME: $RUN_NAME"
echo "LOG_FILE_PATH: $LOG_FILE_PATH"


if [ -n "$TRAIN_DATA_PATH" ]; then
    ACTUAL_DATA_PATH="$TRAIN_DATA_PATH"
else
    ACTUAL_DATA_PATH="$HDFS_DATA_PATH/$DATASET_NAME/$TRAIN_FILE_NAME.parquet"
fi

echo "RUN_NAME: $RUN_NAME" | tee -a $LOG_FILE_PATH
echo "LOG_FILE_PATH: $LOG_FILE_PATH" | tee -a $LOG_FILE_PATH
echo "Training with the following parameters:" | tee -a $LOG_FILE_PATH
echo "Train Batch Size: $TRAIN_BATCH_SIZE" | tee -a $LOG_FILE_PATH
echo "Micro Batch Size per GPU: $MICRO_BATCH_SIZE_PER_GPU" | tee -a $LOG_FILE_PATH
echo "Max Length: $MAX_LENGTH" | tee -a $LOG_FILE_PATH
echo "Total Epochs: $TOTAL_EPOCHS" | tee -a $LOG_FILE_PATH
echo "Save Frequency: $SAVE_FREQ" | tee -a $LOG_FILE_PATH
echo "Model Name: $MODEL_NAME" | tee -a $LOG_FILE_PATH
echo "Dataset Name: $DATASET_NAME" | tee -a $LOG_FILE_PATH
echo "Train File Name: $TRAIN_FILE_NAME" | tee -a $LOG_FILE_PATH
echo "Val File Name: $VAL_FILE_NAME" | tee -a $LOG_FILE_PATH
echo "Prompt Key: $PROMPT_KEY" | tee -a $LOG_FILE_PATH
echo "Response Key: $RESPONSE_KEY" | tee -a $LOG_FILE_PATH
echo "Truncation: $TRUNCATION" | tee -a $LOG_FILE_PATH
echo "Learning Rate: $LEARNING_RATE" | tee -a $LOG_FILE_PATH
echo "SP Size: $SP_SIZE" | tee -a $LOG_FILE_PATH
echo "Number of Nodes: $NNODES" | tee -a $LOG_FILE_PATH
echo "GPUs per Node: $GPUS_PER_NODE" | tee -a $LOG_FILE_PATH
echo "Train Data Path: $TRAIN_DATA_PATH" | tee -a $LOG_FILE_PATH
sleep 3

torchrun --nproc-per-node $GPUS_PER_NODE \
  --master-addr $MASTER_ADDR \
  --node-rank $NODE_RANK \
  --master-port $MASTER_PORT \
  --nnodes $NNODES -m kernel.fsdp_sft_trainer \
  data.multiturn.enable=True \
  data.train_files=$ACTUAL_DATA_PATH \
  data.val_files=$ACTUAL_DATA_PATH \
  data.train_batch_size=$TRAIN_BATCH_SIZE \
  data.micro_batch_size_per_gpu=$MICRO_BATCH_SIZE_PER_GPU \
  data.prompt_key=$PROMPT_KEY \
  data.response_key=$RESPONSE_KEY \
  data.max_length=$MAX_LENGTH \
  data.truncation=$TRUNCATION \
  model.partial_pretrain=$HDFS_MODEL_PATH/$MODEL_NAME \
  model.enable_gradient_checkpointing=True \
  model.fsdp_config.model_dtype=bf16 \
  model.fsdp_config.cpu_offload=True  \
  model.fsdp_config.offload_params=False \
  ulysses_sequence_parallel_size=$SP_SIZE \
  use_remove_padding=True \
  model.strategy=fsdp \
  optim.lr=$LEARNING_RATE \
  trainer.project_name=$PROJECT_NAME \
  trainer.default_local_dir=$HDFS_CHECKPOINT_PATH/$RUN_NAME \
  trainer.experiment_name=$RUN_NAME \
  trainer.total_epochs=$TOTAL_EPOCHS \
  trainer.save_freq=$SAVE_FREQ \
  trainer.logger=["console","wandb"]
