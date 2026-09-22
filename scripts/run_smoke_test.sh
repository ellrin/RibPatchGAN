#!/usr/bin/env bash
# Tiny end-to-end sanity run: LaMa GAN (1 epoch, few batches) -> classifier
# (1 epoch, few batches) -> test on class-balanced subsets.
# Verifies the wiring only — the numbers are meaningless.
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

CONFIG="${CONFIG:-configs/default.yaml}"
RESULTS_ROOT="${RESULTS_ROOT:-${ROOT}/results/smoke_test}"
GAN_OUT="${RESULTS_ROOT}/step2_inpainting_gan"
CLS_OUT="${RESULTS_ROOT}/step3_classifier"
GAN_CKPT="${GAN_OUT}/checkpoints/G_latest.pt"
STATS="$(python -c "from common.config import load_config; print(load_config('${CONFIG}').paths.site_stats)")"

echo "[smoke] results -> ${RESULTS_ROOT}"
mkdir -p "${GAN_OUT}" "${CLS_OUT}"

if [[ ! -f "${STATS}" ]]; then
  python -m step3_train_classifier_with_gan.compute_site_statistics --config "${CONFIG}"
fi

echo ""
echo "[smoke] step 2: LaMa GAN, 1 epoch x 12 batches"
python -m step2_train_inpainting_gan.train \
  --config "${CONFIG}" \
  --epochs 1 \
  --max-batches-per-epoch 12 \
  --num-workers 2 \
  --output-dir "${GAN_OUT}" \
  2>&1 | tee "${GAN_OUT}/run.log"

echo ""
echo "[smoke] step 3: classifier, 1 epoch x 20 train / 20 eval batches"
python -m step3_train_classifier_with_gan.train \
  --config "${CONFIG}" \
  --epochs 1 \
  --max-train-batches 20 \
  --max-eval-batches 20 \
  --num-workers 2 \
  --output-dir "${CLS_OUT}" \
  --gan-checkpoint "${GAN_CKPT}" \
  2>&1 | tee "${CLS_OUT}/run.log"

echo ""
echo "[smoke] test: ROC export + metrics on 60-image subsets"
python -m step3_train_classifier_with_gan.test \
  --config "${CONFIG}" \
  --model-dir "${CLS_OUT}" \
  --limit-images 60 \
  2>&1 | tee "${CLS_OUT}/test.log"

echo ""
echo "[smoke] checking outputs..."
fail=0
for f in \
  "${GAN_CKPT}" \
  "${CLS_OUT}/best_auc.pt" \
  "${CLS_OUT}/metrics.csv" \
  "${CLS_OUT}/eval_operating_points.csv" \
  "${CLS_OUT}/final_test_metrics.csv"; do
  if [[ -f "$f" ]]; then
    echo "  ok   $f"
  else
    echo "  MISS $f"
    fail=1
  fi
done
n_roc=$(ls "${CLS_OUT}/roc_scores/"*_scores.csv 2>/dev/null | wc -l)
if [[ "${n_roc}" -ge 1 ]]; then
  echo "  ok   ${CLS_OUT}/roc_scores/ (${n_roc} csv)"
else
  echo "  MISS ${CLS_OUT}/roc_scores/*_scores.csv"
  fail=1
fi

if [[ "${fail}" == "1" ]]; then
  echo "[smoke] FAILED — missing outputs"
  exit 1
fi
echo "[smoke] PASSED — end-to-end pipeline wiring verified"
