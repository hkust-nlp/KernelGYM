#!/usr/bin/env bash

# Shared environment defaults for DR.Kernel scripts.
# This file is sourced by training/evaluation scripts.

DRKERNEL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${DRKERNEL_ROOT}/.." && pwd)"

export DRKERNEL_ROOT
export REPO_ROOT
export PYTHONPATH="${DRKERNEL_ROOT}:${REPO_ROOT}:${PYTHONPATH:-}"

# Common runtime defaults
export PROJECT_NAME="${PROJECT_NAME:-drkernel}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"

# Local-friendly defaults (can be overridden by environment)
export HDFS_DATA_PATH="${HDFS_DATA_PATH:-${DRKERNEL_ROOT}/data}"
export HDFS_MODEL_PATH="${HDFS_MODEL_PATH:-}"
export HDFS_CHECKPOINT_PATH="${HDFS_CHECKPOINT_PATH:-${DRKERNEL_ROOT}/checkpoints}"
