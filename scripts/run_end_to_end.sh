#!/usr/bin/env bash
# End-to-end: step 2 (LaMa GAN) -> step 3 (classifier) -> test.
# Step 1 (rib maskpoints) is standalone and assumed already done.
#
# Env overrides:
#   CONFIG              default configs/default.yaml
#   GAN_EPOCHS          default: config value (300)
#   CLS_EPOCHS          default: config value (10)
#   RESULTS_ROOT        default: config value (results/)
#   RUN_GAN / RUN_CLS / RUN_TEST   default 1 (set 0 to skip a stage)
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

CONFIG="${CONFIG:-configs/default.yaml}"
RUN_GAN="${RUN_GAN:-1}"
RUN_CLS="${RUN_CLS:-1}"
RUN_TEST="${RUN_TEST:-1}"

RESULTS_ROOT="${RESULTS_ROOT:-$(python -c "from common.config import load_config; print(load_config('${CONFIG}').paths.results_root)")}"
GAN_OUT="${RESULTS_ROOT}/step2_inpainting_gan"
CLS_OUT="${RESULTS_ROOT}/step3_classifier"
GAN_CKPT="${GAN_OUT}/checkpoints/G_latest.pt"
STATS="$(python -c "from common.config import load_config; print(load_config('${CONFIG}').paths.site_stats)")"

echo "=========================================================="
echo "[pipeline] RibPatchGAN_v5 end-to-end"
echo "[pipeline] config=${CONFIG} results=${RESULTS_ROOT}"
echo "=========================================================="
mkdir -p "${RESULTS_ROOT}"

if [[ ! -f "${STATS}" ]]; then
  echo "[setup] computing train-set site statistics"
  python -m step3_train_classifier_with_gan.compute_site_statistics --config "${CONFIG}"
fi

if [[ "${RUN_GAN}" == "1" ]]; then
  echo ""
  echo "[step2] training LaMa inpainting GAN"
  mkdir -p "${GAN_OUT}"
  GAN_ARGS=(--config "${CONFIG}" --output-dir "${GAN_OUT}")
  [[ -n "${GAN_EPOCHS:-}" ]] && GAN_ARGS+=(--epochs "${GAN_EPOCHS}")
  python -m step2_train_inpainting_gan.train "${GAN_ARGS[@]}" \
    2>&1 | tee "${GAN_OUT}/run.log"
fi

if [[ ! -f "${GAN_CKPT}" ]]; then
  echo "[err] missing generator checkpoint: ${GAN_CKPT}"
  exit 1
fi

if [[ "${RUN_CLS}" == "1" ]]; then
  echo ""
  echo "[step3] training classifier"
  mkdir -p "${CLS_OUT}"
  CLS_ARGS=(--config "${CONFIG}" --output-dir "${CLS_OUT}" --gan-checkpoint "${GAN_CKPT}")
  [[ -n "${CLS_EPOCHS:-}" ]] && CLS_ARGS+=(--epochs "${CLS_EPOCHS}")
  python -m step3_train_classifier_with_gan.train "${CLS_ARGS[@]}" \
    2>&1 | tee "${CLS_OUT}/run.log"
fi

if [[ "${RUN_TEST}" == "1" ]]; then
  echo ""
  echo "[test] ROC export + test metrics"
  python -m step3_train_classifier_with_gan.test \
    --config "${CONFIG}" --model-dir "${CLS_OUT}" \
    2>&1 | tee "${CLS_OUT}/test.log"
fi

echo ""
echo "=========================================================="
echo "[done] results in ${RESULTS_ROOT}"
echo "  GAN checkpoints : ${GAN_OUT}/checkpoints/"
echo "  classifier      : ${CLS_OUT}/best_auc.pt"
echo "  ROC scores      : ${CLS_OUT}/roc_scores/"
echo "  test metrics    : ${CLS_OUT}/final_test_metrics.csv"
echo "=========================================================="
