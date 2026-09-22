"""Compute train-set image mean/std (uint8 space) -> paths.site_stats JSON.

All datasets are standardized to the TRAIN set statistics at train/valid/test
time. Run once; both train.py and test.py require the output file.

Usage:
    python -m step3_train_classifier_with_gan.compute_site_statistics \
        --config configs/default.yaml
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import load_config
from step3_train_classifier_with_gan.data.items import build_items
from step3_train_classifier_with_gan.data.preprocessing import load_cxr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="YAML config (default: configs/default.yaml)")
    args = ap.parse_args()
    cfg = load_config(args.config)

    items = build_items(cfg.datasets.train)
    print(f"[stats] computing from {len(items)} {cfg.datasets.train.name} train images")

    means = [load_cxr(it.img_path).astype(np.float32).mean(axis=(0, 1)) for it in items]
    mean = np.mean(means, axis=0)
    vars_ = [((load_cxr(it.img_path).astype(np.float32) - mean) ** 2).mean(axis=(0, 1)) for it in items]
    std = np.sqrt(np.mean(vars_, axis=0))

    out_path = Path(cfg.paths.site_stats)
    with out_path.open("w") as f:
        json.dump({"mean": mean.tolist(), "std": std.tolist()}, f, indent=2)
    print(f"[stats] wrote {out_path}: mean={mean} std={std}")


if __name__ == "__main__":
    main()
