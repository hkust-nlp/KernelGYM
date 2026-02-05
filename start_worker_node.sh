#!/usr/bin/env bash
# Worker-only node startup for KernelGym (connects to remote API/Redis)

set -euo pipefail

BLUE='\033[0;34m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'

echo -e "${BLUE}Starting KernelGym worker node${NC}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}"
cd "$ROOT_DIR"

# Optional arg: server.env path (default ./server.env)
SERVER_ENV_PATH="${1:-${ROOT_DIR}/server.env}"

if [[ ! -f "$SERVER_ENV_PATH" ]]; then
  echo -e "${RED}server.env not found: $SERVER_ENV_PATH${NC}"
  echo "Usage: ./start_worker_node.sh [/path/to/server.env]"
  exit 1
fi

# If .env is missing, copy from server.env
if [[ ! -f .env ]]; then
  echo -e "${YELLOW}.env not found; copying ${SERVER_ENV_PATH} -> .env${NC}"
  cp "$SERVER_ENV_PATH" .env
else
  echo -e "${GREEN}Using existing .env (can override GPU_DEVICES/NODE_ID locally)${NC}"
fi

# Load .env
set -o allexport
source .env
set +o allexport

# Validate required config
for v in API_HOST API_PORT REDIS_HOST REDIS_PORT; do
  if [[ -z "${!v:-}" ]]; then
    echo -e "${RED}Missing required env var: $v (set in server.env/.env)${NC}"
    exit 1
  fi
done

# Pre-flight connectivity checks
echo -e "${BLUE}Checking API and Redis connectivity...${NC}"

# Redis check
echo -n "  Redis... "
if redis-cli -h "${REDIS_HOST}" -p "${REDIS_PORT}" -a "${REDIS_PASSWORD:-}" --no-auth-warning PING >/dev/null 2>&1; then
  echo -e "${GREEN}OK${NC}"
else
  echo -e "${RED}FAILED${NC}"
  echo -e "${RED}Cannot connect to Redis: ${REDIS_HOST}:${REDIS_PORT}${NC}"
  exit 1
fi

# API check
API_URL_CHECK="http://${API_HOST}:${API_PORT}"
if [[ "${API_HOST}" == *":"* ]] && [[ "${API_HOST}" != "["* ]]; then
  API_URL_CHECK="http://[${API_HOST}]:${API_PORT}"
fi

echo -n "  API... "
if curl -s --max-time 5 "${API_URL_CHECK}/health" >/dev/null 2>&1; then
  echo -e "${GREEN}OK${NC}"
else
  echo -e "${RED}FAILED${NC}"
  echo -e "${RED}Cannot reach API server: ${API_URL_CHECK}/health${NC}"
  exit 1
fi

echo -e "${GREEN}Connectivity checks passed${NC}"

# Clear problematic env vars
unset GPU_ARCH

# Node allocation / registration
HOSTNAME_VAL=$(hostname)
API_BASE="http://${API_HOST}:${API_PORT}"
if [[ "$API_HOST" == *:* && "$API_HOST" != \[*\] ]]; then
  API_BASE="http://[${API_HOST}]:${API_PORT}"
fi

if [[ -z "${NODE_ID:-}" ]]; then
  echo -e "${BLUE}Requesting server-assigned node_id...${NC}"
  set +e
  RESP=$(curl -sS -X POST "${API_BASE}/node/allocate?hostname=${HOSTNAME_VAL}" -w "\n%{http_code}")
  CURL_RC=$?
  set -e
  ALLOC_BODY="${RESP%$'\n'*}"
  ALLOC_CODE="${RESP##*$'\n'}"
  echo -e "${BLUE}/node/allocate http_code=${ALLOC_CODE}${NC}"
  if [[ -n "${ALLOC_BODY}" ]]; then
    echo -e "${BLUE}/node/allocate body: ${ALLOC_BODY}${NC}"
  fi
  if [[ $CURL_RC -eq 0 && "${ALLOC_CODE}" == "200" && -n "${ALLOC_BODY}" ]]; then
    NODE_ID=$(printf '%s' "${ALLOC_BODY}" | python3 -c 'import sys,json
try:
    data=json.loads(sys.stdin.read())
    print(data.get("node_id",""))
except Exception:
    pass')
    ALLOC_HOSTNAME=$(printf '%s' "${ALLOC_BODY}" | python3 -c 'import sys,json
try:
    data=json.loads(sys.stdin.read())
    print(data.get("hostname",""))
except Exception:
    pass')
    if [[ -n "$NODE_ID" ]]; then
      echo -e "${GREEN}Assigned NODE_ID=${NODE_ID}${NC}"
      export NODE_ID
      if grep -q '^NODE_ID=' .env; then
        sed -i.bak "s/^NODE_ID=.*/NODE_ID=${NODE_ID}/" .env && rm -f .env.bak
      else
        echo "NODE_ID=${NODE_ID}" >> .env
      fi
      if [[ -n "${ALLOC_HOSTNAME}" ]]; then
        export WORKER_NAME_PREFIX="${ALLOC_HOSTNAME}"
        if grep -q '^WORKER_NAME_PREFIX=' .env; then
          sed -i.bak "s/^WORKER_NAME_PREFIX=.*/WORKER_NAME_PREFIX=${WORKER_NAME_PREFIX}/" .env && rm -f .env.bak
        else
          echo "WORKER_NAME_PREFIX=${WORKER_NAME_PREFIX}" >> .env
        fi
      fi
    else
      echo -e "${YELLOW}Unexpected /node/allocate response; continuing with local NODE_ID${NC}"
    fi
  else
    echo -e "${YELLOW}Node allocation failed; worker will fall back to hostname at runtime${NC}"
  fi
else
  echo -e "${BLUE}Using NODE_ID=${NODE_ID}; registering with server...${NC}"
  set +e
  RESP=$(curl -sS -X POST "${API_BASE}/node/allocate?hostname=${HOSTNAME_VAL}&node_name=${NODE_ID}" -w "\n%{http_code}")
  CURL_RC=$?
  set -e
  ALLOC_BODY="${RESP%$'\n'*}"
  ALLOC_CODE="${RESP##*$'\n'}"
  echo -e "${BLUE}/node/allocate http_code=${ALLOC_CODE}${NC}"
  if [[ -n "${ALLOC_BODY}" ]]; then
    echo -e "${BLUE}/node/allocate body: ${ALLOC_BODY}${NC}"
  fi
  if [[ $CURL_RC -eq 0 && "${ALLOC_CODE}" == "200" ]]; then
    echo -e "${GREEN}Registered NODE_ID=${NODE_ID}${NC}"
  elif [[ "${ALLOC_CODE}" == "409" ]]; then
    echo -e "${RED}NODE_ID=${NODE_ID} already in use by another host${NC}"
    exit 1
  else
    echo -e "${YELLOW}Node registration failed; continuing (worker will retry)${NC}"
  fi
fi

echo -e "${GREEN}API: ${API_HOST}:${API_PORT} | Redis: ${REDIS_HOST}:${REDIS_PORT} | NODE_ID: ${NODE_ID:-<auto>}${NC}"

# Logs
mkdir -p logs

# Cleanup old processes
echo -e "${BLUE}Cleaning old worker processes...${NC}"
pkill -f "python.*kernelgym.worker.gpu_worker" 2>/dev/null || true
sleep 1

# PYTHONPATH
export PYTHONPATH="${PYTHONPATH:-}:${ROOT_DIR}"

# Start worker manager
echo -e "${BLUE}Starting GPU WorkerManager...${NC}"
nohup python3 -m kernelgym.worker.gpu_worker > logs/worker_manager.log 2>&1 &
WORKER_MGR_PID=$!
echo -e "${GREEN}WorkerManager PID: ${WORKER_MGR_PID}${NC}"
echo ${WORKER_MGR_PID} > logs/worker_manager.pid

cleanup() {
  echo -e "${BLUE}Stopping WorkerManager (PID: ${WORKER_MGR_PID})...${NC}"
  set +e
  if kill -0 "${WORKER_MGR_PID}" 2>/dev/null; then
    kill -TERM "${WORKER_MGR_PID}" 2>/dev/null || true
    pkill -TERM -P "${WORKER_MGR_PID}" 2>/dev/null || true
    wait "${WORKER_MGR_PID}" 2>/dev/null || true
  fi
  rm -f logs/worker_manager.pid
  set -e
  echo -e "${GREEN}WorkerManager stopped${NC}"
  exit 0
}
trap cleanup SIGINT SIGTERM

# Record expected worker ids (best-effort)
PREFIX_TO_USE="${NODE_ID:-}"
if [[ -z "$PREFIX_TO_USE" ]]; then
  PREFIX_TO_USE="${WORKER_NAME_PREFIX:-$(hostname)}"
fi
RAW_GPU_DEV="${GPU_DEVICES:-}"
GPU_LIST=""
if [[ "$RAW_GPU_DEV" =~ ^\[.*\]$ ]]; then
  STR=${RAW_GPU_DEV#[}; STR=${STR%]}; STR=${STR//[[:space:]]/}
  IFS=',' read -ra ARR <<< "$STR"; GPU_LIST="${ARR[@]}"
elif [[ -n "$RAW_GPU_DEV" ]]; then
  IFS=',' read -ra ARR <<< "$RAW_GPU_DEV"; GPU_LIST="${ARR[@]}"
else
  GPU_LIST="0 1 2 3 4 5 6 7"
fi
{
  for i in $GPU_LIST; do
    echo "${PREFIX_TO_USE}_gpu_${i}"
  done
} > logs/worker_ids.list

echo -e "${BLUE}Waiting for workers to register (10s)...${NC}"
sleep 10

# Simple status check
API_BASE="http://${API_HOST}:${API_PORT}"
if [[ "$API_HOST" == *:* && "$API_HOST" != \[*\] ]]; then
  API_BASE="http://[${API_HOST}]:${API_PORT}"
fi
set +e
WS=$(curl -sS "${API_BASE}/workers/status" || true)
set -e
if [[ -n "$WS" ]]; then
  echo -e "${GREEN}Workers connected to ${API_BASE}${NC}"
else
  echo -e "${YELLOW}Unable to fetch workers status; check logs/worker_manager.log${NC}"
fi

echo -e "${GREEN}Worker node started. Logs: logs/worker_manager.log${NC}"

# Foreground wait
echo -e "${BLUE}Foreground wait for WorkerManager (PID: ${WORKER_MGR_PID}); Ctrl+C to stop${NC}"
set +e
wait "${WORKER_MGR_PID}"
RC=$?
set -e
rm -f logs/worker_manager.pid
if [[ $RC -ne 0 ]]; then
  echo -e "${YELLOW}WorkerManager exited with code ${RC}. Check logs/worker_manager.log${NC}"
else
  echo -e "${GREEN}WorkerManager exited normally${NC}"
fi
exit $RC
