#!/bin/bash

# KernelGym auto-configuration script.
# Generates a .env file with detected IP and available ports (ARNOLD-aware).

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ROOT_DIR}/.env"

FORCE=false
USE_INDEXED_PORTS=false
ENABLE_SAVE_RESULTS=false
for arg in "$@"; do
    case "${arg}" in
        --force)
            FORCE=true
            ;;
        --use-indexed-ports)
            USE_INDEXED_PORTS=true
            ;;
        --save-eval-results)
            ENABLE_SAVE_RESULTS=true
            ;;
        *)
            ;;
    esac
done

if [ -f "${ENV_FILE}" ] && [ "${FORCE}" = false ]; then
    echo "Found existing .env at ${ENV_FILE}. Use --force to overwrite."
    exit 0
fi

get_available_ports() {
    local role="${ARNOLD_ROLE:-}"
    local worker_id="${ARNOLD_ID:-}"

    local varname="ARNOLD_${role^^}_${worker_id}_PORT"
    local allports="${!varname:-}"

    local need_probe=false
    if [ "${USE_INDEXED_PORTS}" = true ]; then
        need_probe=true
        allports=""
    elif [ -z "${allports}" ] || [[ "${allports}" != *,* ]]; then
        need_probe=true
    fi
    if [ "${need_probe}" = true ]; then
        local ports_list=()
        local idx=0
        while true; do
            local indexed_varname="PORT${idx}"
            local pv="${!indexed_varname:-}"
            if [ -z "${pv}" ]; then
                break
            fi
            ports_list+=("${pv}")
            idx=$((idx+1))
        done
        if [ ${#ports_list[@]} -gt 0 ]; then
            local joined=""
            for p in "${ports_list[@]}"; do
                if [ -n "${joined}" ]; then
                    joined="${joined},${p}"
                else
                    joined="${p}"
                fi
            done
            allports="${joined}"
            echo "Using indexed environment ports: PORT0..$((idx-1))" >&2
        fi
    fi

    if [ -z "${allports}" ]; then
        echo "No ARNOLD ports found, using fallback ports" >&2
        allports="8000,8001,8002,8003,8004,8005,8006,8007,8008,8009"
    fi

    echo "${allports}"
}

get_ip_address() {
    local role="${ARNOLD_ROLE:-}"
    local worker_id="${ARNOLD_ID:-}"
    local ipvarname="ARNOLD_${role^^}_${worker_id}_HOST"
    local ipaddress="${!ipvarname:-}"

    if [ -z "${ipaddress}" ]; then
        ipaddress="$(hostname -I | awk '{print $1}')"
        if [ -z "${ipaddress}" ]; then
            ipaddress="127.0.0.1"
        fi
    fi

    echo "${ipaddress}"
}

is_port_available() {
    local port="$1"
    if command -v ss >/dev/null 2>&1; then
        ! ss -tlnp | grep -Fq ":${port} "
    else
        ! lsof -iTCP -sTCP:LISTEN -P 2>/dev/null | grep -Fq ":${port}"
    fi
}

select_ports() {
    local available_ports="$1"
    local needed_ports=("redis" "api" "metrics")

    IFS=',' read -ra PORT_ARRAY <<< "${available_ports}"

    local selected_ports=()
    local port_index=0

    for service in "${needed_ports[@]}"; do
        local found_port=""

        for ((i=port_index; i<${#PORT_ARRAY[@]}; i++)); do
            local port="${PORT_ARRAY[i]}"
            port="${port//[[:space:]]/}"
            if is_port_available "${port}"; then
                found_port="${port}"
                port_index=$((i+1))
                break
            fi
        done

        if [ -z "${found_port}" ]; then
            echo "Could not find available port for ${service}"
            exit 1
        fi

        selected_ports+=("${found_port}")
    done

    echo "${selected_ports[@]}"
}

detect_gpus_json() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        local count
        count="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
        if [ "${count}" -gt 0 ]; then
            local devices="["
            for ((i=0; i<"${count}"; i++)); do
                if [ "${i}" -gt 0 ]; then
                    devices="${devices},"
                fi
                devices="${devices}${i}"
            done
            devices="${devices}]"
            echo "${devices}"
            return
        fi
    fi
    echo "[0]"
}

API_HOST="${API_HOST:-$(get_ip_address)}"

AVAILABLE_PORTS="$(get_available_ports)"
SELECTED_PORTS="$(select_ports "${AVAILABLE_PORTS}")"

REDIS_PORT="$(echo "${SELECTED_PORTS}" | awk '{print $1}')"
API_PORT="$(echo "${SELECTED_PORTS}" | awk '{print $2}')"
METRICS_PORT="$(echo "${SELECTED_PORTS}" | awk '{print $3}')"
GPU_DEVICES="${GPU_DEVICES:-$(detect_gpus_json)}"

REDIS_HOST="${REDIS_HOST:-localhost}"
REDIS_DB="${REDIS_DB:-0}"
REDIS_PASSWORD="${REDIS_PASSWORD:-}"
REDIS_KEY_PREFIX="${REDIS_KEY_PREFIX:-kernelgym}"

API_WORKERS="${API_WORKERS:-4}"
API_RELOAD="${API_RELOAD:-false}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
LOG_DIR="${LOG_DIR:-logs}"
ENABLE_METRICS="${ENABLE_METRICS:-true}"
ENABLE_PROFILING="${ENABLE_PROFILING:-true}"
VERBOSE_ERROR_TRACEBACK="${VERBOSE_ERROR_TRACEBACK:-true}"
SAVE_EVAL_RESULTS="${SAVE_EVAL_RESULTS:-false}"
EVAL_RESULTS_PATH="${EVAL_RESULTS_PATH:-logs/eval_results.jsonl}"

if [ "${ENABLE_SAVE_RESULTS}" = true ]; then
    SAVE_EVAL_RESULTS=true
fi

DEFAULT_TOOLKIT="${DEFAULT_TOOLKIT:-kernelbench}"
DEFAULT_BACKEND_ADAPTER="${DEFAULT_BACKEND_ADAPTER:-kernelbench}"
DEFAULT_BACKEND="${DEFAULT_BACKEND:-triton}"

NODE_ID="${NODE_ID:-$(hostname 2>/dev/null || echo "")}"
WORKER_POOL_SIZE="${WORKER_POOL_SIZE:-1}"
MAX_TASKS_PER_WORKER="${MAX_TASKS_PER_WORKER:-1}"

cat > "${ENV_FILE}" <<EOF
# KernelGym Auto-Generated Configuration
# Generated on: $(date)

# Network
API_HOST=${API_HOST}
API_PORT=${API_PORT}
API_WORKERS=${API_WORKERS}
API_RELOAD=${API_RELOAD}

# GPU
GPU_DEVICES=${GPU_DEVICES}
GPU_MEMORY_LIMIT=16GB
NODE_ID=${NODE_ID}

# Redis
REDIS_HOST=${REDIS_HOST}
REDIS_PORT=${REDIS_PORT}
REDIS_DB=${REDIS_DB}
REDIS_PASSWORD=${REDIS_PASSWORD}
REDIS_KEY_PREFIX=${REDIS_KEY_PREFIX}

# Worker pool
WORKER_POOL_SIZE=${WORKER_POOL_SIZE}
MAX_TASKS_PER_WORKER=${MAX_TASKS_PER_WORKER}

# Defaults
DEFAULT_TOOLKIT=${DEFAULT_TOOLKIT}
DEFAULT_BACKEND_ADAPTER=${DEFAULT_BACKEND_ADAPTER}
DEFAULT_BACKEND=${DEFAULT_BACKEND}

# Logging
LOG_LEVEL=${LOG_LEVEL}
LOG_DIR=${LOG_DIR}

# Metrics
ENABLE_METRICS=${ENABLE_METRICS}
METRICS_PORT=${METRICS_PORT}

# Profiling
ENABLE_PROFILING=${ENABLE_PROFILING}

# Errors
VERBOSE_ERROR_TRACEBACK=${VERBOSE_ERROR_TRACEBACK}

# Result persistence
SAVE_EVAL_RESULTS=${SAVE_EVAL_RESULTS}
EVAL_RESULTS_PATH=${EVAL_RESULTS_PATH}
EOF

echo "Wrote configuration to ${ENV_FILE}"
