#!/bin/bash
set -euo pipefail

# IAMFlow interactive inference launcher.
# Usage: bash run_iamflow.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

echo "=========================================="
echo "IAMFlow Interactive Inference"
echo "=========================================="

PRETRAINED_ROOT="${PRETRAINED_ROOT:-pretrained}"
export WAN_MODEL_PATH="${WAN_MODEL_PATH:-${PRETRAINED_ROOT}/Wan2.1-T2V-1.3B}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export VLLM_HOST_IP="${VLLM_HOST_IP:-127.0.0.1}"
export VLLM_LOOPBACK_IP="${VLLM_LOOPBACK_IP:-127.0.0.1}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" python -m iamflow.run_iamflow \
  --config_path configs/iamflow.yaml \
  --dit_quantized_ckpt "${PRETRAINED_ROOT}/iamflow_models/iamflow_fp8.safetensors" \
  --llm_model_path "${PRETRAINED_ROOT}/Qwen3-4B-Instruct-2507" \
  --vlm_model_path "${PRETRAINED_ROOT}/Qwen3-VL-2B-Instruct" \
  --max_memory_frames 3 \
  --save_dir data/agent_frames
