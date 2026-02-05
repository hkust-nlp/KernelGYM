#!/bin/bash
# Stop KernelGym API server, workers, and monitor, then optionally clear Redis keys.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ROOT_DIR}/.env"

REDIS_HOST="localhost"
REDIS_PORT=""
REDIS_PASSWORD=""
REDIS_KEY_PREFIX="kernelgym"
API_PORT=""

if [ -f "${ENV_FILE}" ]; then
    API_PORT="$(grep "^API_PORT=" "${ENV_FILE}" 2>/dev/null | cut -d'=' -f2 | tr -d '"' | tr -d ' ')"
    REDIS_HOST="$(grep "^REDIS_HOST=" "${ENV_FILE}" 2>/dev/null | cut -d'=' -f2 | tr -d '"' | tr -d ' ')"
    REDIS_PORT="$(grep "^REDIS_PORT=" "${ENV_FILE}" 2>/dev/null | cut -d'=' -f2 | tr -d '"' | tr -d ' ')"
    REDIS_PASSWORD="$(grep "^REDIS_PASSWORD=" "${ENV_FILE}" 2>/dev/null | cut -d'=' -f2 | tr -d '"' | tr -d ' ')"
    REDIS_KEY_PREFIX="$(grep "^REDIS_KEY_PREFIX=" "${ENV_FILE}" 2>/dev/null | cut -d'=' -f2 | tr -d '"' | tr -d ' ')"
fi

if [ -z "${REDIS_HOST}" ]; then
    REDIS_HOST="localhost"
fi
if [ -z "${REDIS_KEY_PREFIX}" ]; then
    REDIS_KEY_PREFIX="kernelgym"
fi

kill_processes() {
    local pattern="$1"
    local description="$2"

    echo "Stopping ${description}..."
    local pids=""
    if command -v pgrep >/dev/null 2>&1; then
        pids="$(pgrep -f "${pattern}" || true)"
    else
        pids="$(ps aux | grep "${pattern}" | grep -v grep | awk '{print $2}' || true)"
    fi

    if [ -z "${pids}" ]; then
        echo "No ${description} processes found."
        return
    fi

    echo "${pids}" | xargs -r kill -TERM || true
    sleep 2

    local remaining=""
    if command -v pgrep >/dev/null 2>&1; then
        remaining="$(pgrep -f "${pattern}" || true)"
    else
        remaining="$(ps aux | grep "${pattern}" | grep -v grep | awk '{print $2}' || true)"
    fi

    if [ -n "${remaining}" ]; then
        echo "Force killing ${description}..."
        echo "${remaining}" | xargs -r kill -KILL || true
    fi
}

echo "Stopping KernelGym processes..."

kill_processes "kernelgym.server.api.server" "KernelGym API server"
kill_processes "kernelgym.worker.worker_monitor" "KernelGym worker monitor"
kill_processes "kernelgym.worker.single_worker" "KernelGym GPU workers"
kill_processes "kernelgym.worker.gpu_worker" "KernelGym GPU worker core"
kill_processes "uvicorn.*kernelgym" "Uvicorn server"

echo "Stopping multiprocessing worker processes..."
kill_processes "multiprocessing.spawn" "multiprocessing spawn workers"
kill_processes "multiprocessing.resource_tracker" "multiprocessing resource tracker"

if command -v redis-cli >/dev/null 2>&1; then
    if [ -n "${REDIS_PORT}" ]; then
        REDIS_AUTH_ARGS=()
        if [ -n "${REDIS_PASSWORD}" ]; then
            REDIS_AUTH_ARGS=(-a "${REDIS_PASSWORD}" --no-auth-warning)
        fi
        echo "Clearing Redis keys with prefix '${REDIS_KEY_PREFIX}:' on ${REDIS_HOST}:${REDIS_PORT}..."
        redis-cli -h "${REDIS_HOST}" -p "${REDIS_PORT}" "${REDIS_AUTH_ARGS[@]}" \
            --scan --pattern "${REDIS_KEY_PREFIX}:*" \
            | xargs -r redis-cli -h "${REDIS_HOST}" -p "${REDIS_PORT}" "${REDIS_AUTH_ARGS[@]}" DEL >/dev/null 2>&1 || true
    else
        echo "REDIS_PORT not set; skipping Redis cleanup."
    fi
else
    echo "redis-cli not found; skipping Redis cleanup."
fi

echo "KernelGym stopped."
