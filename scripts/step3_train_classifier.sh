#!/usr/bin/env bash
# Step 3 — train the classifier with GAN augmentation + cooperative G feedback.
# Requires the step-2 generator checkpoint (G_latest.pt).
# Env overrides: CONFIG, EPOCHS, OUTPUT_DIR, GAN_CHECKPOINT
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

CONFIG="${CONFIG:-configs/default.yaml}"

if [[ ! -f "$(python -c "from common.config import load_config; print(load_config('${CONFIG}').paths.site_stats)")" ]]; then
  echo "[setup] computing train-set site statistics"
  python -m step3_train_classifier_with_gan.compute_site_statistics --config "${CONFIG}"
fi

ARGS=(--config "${CONFIG}")
[[ -n "${EPOCHS:-}" ]] && ARGS+=(--epochs "${EPOCHS}")
[[ -n "${OUTPUT_DIR:-}" ]] && ARGS+=(--output-dir "${OUTPUT_DIR}")
[[ -n "${GAN_CHECKPOINT:-}" ]] && ARGS+=(--gan-checkpoint "${GAN_CHECKPOINT}")

python -m step3_train_classifier_with_gan.train "${ARGS[@]}" "$@"
