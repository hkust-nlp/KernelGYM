#!/usr/bin/env bash
set -euo pipefail

HF_REPO="${1:-}"
LOCALPATH="${2:-}"
REPOTYPE="${3:-}"

if [[ -z "${HF_REPO}" || -z "${LOCALPATH}" || -z "${REPOTYPE}" ]]; then
  echo "Usage: $0 <hf_repo> <local_path> <repo_type>" >&2
  echo "Example: $0 \"org/name\" ./data dataset" >&2
  exit 2
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is not set. Please export HF_TOKEN=<your_huggingface_token>." >&2
  exit 2
fi

python drkernel/kernel/scripts/preprocess/push_to_hub.py \
  --repo_id "${HF_REPO}" \
  --local_path "${LOCALPATH}" \
  --repo_type "${REPOTYPE}" \
  --commit_message "upload ckpt" \
  --create_repo
