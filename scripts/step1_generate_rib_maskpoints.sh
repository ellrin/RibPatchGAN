#!/usr/bin/env bash
# Step 1 — rib maskpoint generation (standalone).
# Usage: ./scripts/step1_generate_rib_maskpoints.sh <image_dir> [extra args...]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <image_dir> [--output DIR] [--device cuda:0] ..."
  exit 1
fi

cd "${ROOT}/step1_rib_maskpoint_generation"
python generate_rib_maskpoints.py --input "$@"
