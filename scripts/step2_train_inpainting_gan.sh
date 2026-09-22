#!/usr/bin/env bash
# Step 2 — train the LaMa inpainting GAN.
# Env overrides: CONFIG, EPOCHS, OUTPUT_DIR
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

CONFIG="${CONFIG:-configs/default.yaml}"
ARGS=(--config "${CONFIG}")
[[ -n "${EPOCHS:-}" ]] && ARGS+=(--epochs "${EPOCHS}")
[[ -n "${OUTPUT_DIR:-}" ]] && ARGS+=(--output-dir "${OUTPUT_DIR}")

python -m step2_train_inpainting_gan.train "${ARGS[@]}" "$@"
