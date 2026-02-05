#!/bin/bash

# KernelGym Multi-Node Worker Startup Wrapper
# For worker-only nodes that connect to a remote KernelGym API/Redis server

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}KernelGym Multi-Node Worker Startup${NC}"
echo "==================================================="

# Resolve repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}"
cd "$ROOT_DIR"

# Check .env
if [[ ! -f ".env" ]]; then
    echo -e "${RED}Error: .env file not found${NC}"
    echo ""
    echo -e "${YELLOW}This script requires a worker-only .env configuration.${NC}"
    echo ""
    echo "Create one with at least:"
    echo "  API_HOST=<main_server_ip>"
    echo "  API_PORT=<main_server_port>"
    echo "  REDIS_HOST=<main_server_ip>"
    echo "  REDIS_PORT=<redis_port>"
    echo ""
    echo "Optional:"
    echo "  REDIS_PASSWORD=<password>"
    echo "  GPU_DEVICES=[0,1] or 0,1"
    echo "  NODE_ID=<node-name>"
    exit 1
fi

# Load .env
set -o allexport
source .env
set +o allexport

# Verify intended usage (NODE_ID recommended)
if [[ -z "${NODE_ID:-}" ]]; then
    echo -e "${YELLOW}Warning: NODE_ID not set in .env${NC}"
    echo -e "${YELLOW}This script is intended for multi-node deployments.${NC}"
    echo -e "${YELLOW}For single-node deployments, use: ${GREEN}./start_all_with_monitor.sh${NC}"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo -e "${BLUE}Configuration:${NC}"
echo -e "  Node ID:     ${GREEN}${NODE_ID:-<auto>}${NC}"
echo -e "  API Server:  ${GREEN}${API_HOST}:${API_PORT}${NC}"
echo -e "  Redis:       ${GREEN}${REDIS_HOST}:${REDIS_PORT}${NC}"
echo -e "  GPU Devices: ${GREEN}${GPU_DEVICES:-all}${NC}"
echo ""

# Create server.env symlink for compatibility
if [[ ! -f "server.env" ]]; then
    echo -e "${BLUE}Creating server.env symlink to .env...${NC}"
    ln -sf .env server.env
    echo -e "${GREEN}Created server.env -> .env${NC}"
    echo ""
fi

# Start worker-only node
echo -e "${BLUE}Starting worker-only node...${NC}"
echo ""

exec bash start_worker_node.sh
