#!/bin/bash
# Helper script to write Redis configuration to a lock file

set -e

ENV_FILE="${1:-.env}"
LOCK_FILE="${2:-}"

if [[ -z "$LOCK_FILE" ]]; then
    echo "Error: LOCK_FILE path required as second argument"
    exit 1
fi

if [[ ! -f "$ENV_FILE" ]]; then
    echo "Error: Environment file $ENV_FILE not found"
    exit 1
fi

# Extract Redis configuration from .env file
redis_host=$(grep -E '^REDIS_HOST=' "$ENV_FILE" | cut -d= -f2 | tr -d '\r\n')
redis_port=$(grep -E '^REDIS_PORT=' "$ENV_FILE" | cut -d= -f2 | tr -d '\r\n')
redis_password=$(grep -E '^REDIS_PASSWORD=' "$ENV_FILE" | cut -d= -f2 | tr -d '\r\n')
api_host=$(grep -E '^API_HOST=' "$ENV_FILE" | cut -d= -f2 | tr -d '\r\n')

# Fallback to API_HOST if REDIS_HOST is not set
if [[ -z "$redis_host" ]]; then
    redis_host="$api_host"
fi

# Validate extracted values
if [[ -z "$redis_host" || -z "$redis_port" ]]; then
    echo "Error: Could not extract Redis configuration from $ENV_FILE"
    echo "  REDIS_HOST: ${redis_host:-<missing>}"
    echo "  REDIS_PORT: ${redis_port:-<missing>}"
    exit 1
fi

# Ensure lock directory exists
lock_dir="$(dirname "$LOCK_FILE")"
mkdir -p "$lock_dir" || {
    echo "Error: Failed to create lock directory: $lock_dir"
    exit 1
}

# Write Redis configuration to lock file (3 lines)
tmp_file="${LOCK_FILE}.tmp.$$"
{
    printf '%s\n' "$redis_host"
    printf '%s\n' "$redis_port"
    printf '%s\n' "$redis_password"
} > "$tmp_file"

sync || true
mv -f "$tmp_file" "$LOCK_FILE"
chmod 0644 "$LOCK_FILE" 2>/dev/null || true
sync || true

echo "Redis lock file written: $LOCK_FILE"
echo "  Host: $redis_host"
echo "  Port: $redis_port"
echo "  Password: [hidden]"
