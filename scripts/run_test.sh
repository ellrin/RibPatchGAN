#!/usr/bin/env bash
# Test — single inference pass over every configured test set:
# per-image ROC score CSVs + eval_operating_points.csv + final_test_metrics.csv.
# Env overrides: CONFIG, MODEL_DIR
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

CONFIG="${CONFIG:-configs/default.yaml}"
ARGS=(--config "${CONFIG}")
[[ -n "${MODEL_DIR:-}" ]] && ARGS+=(--model-dir "${MODEL_DIR}")

python -m step3_train_classifier_with_gan.test "${ARGS[@]}" "$@"
