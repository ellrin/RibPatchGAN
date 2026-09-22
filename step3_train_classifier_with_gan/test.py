"""Single test-inference pass for a trained classifier, over every test set
listed in the config.

Loads <model-dir>/best_auc.pt, forwards the valid set (for the Youden
threshold) and each test set ONCE, then writes:

  <model-dir>/roc_scores/<arm>__best_auc__<testset>_scores.csv
        per-image columns: filename, label, pred (image-topk score)
  <model-dir>/eval_operating_points.csv
        one row per test set at its configured operating point
        (threshold: youden_valid | spec_target, see configs/*.yaml)
  <model-dir>/final_test_metrics.csv
        per test set at the valid-Youden threshold, rebuilt from the
        exported scores — there is never a second inference pass.

Usage:
    python -m step3_train_classifier_with_gan.test --config configs/default.yaml \
        [--model-dir results/step3_classifier] [--limit-images N]
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import load_config
from step3_train_classifier_with_gan.data.dataset import PatchDataset, collate_patch_bags
from step3_train_classifier_with_gan.data.items import build_items
from step3_train_classifier_with_gan.data.preprocessing import load_site_stats
from step3_train_classifier_with_gan.engine import flatten_bags, seed_all, split_bags
from step3_train_classifier_with_gan.metrics import (
    aggregate_topk,
    binary_metrics,
    roc_auc,
    threshold_summary,
)
from step3_train_classifier_with_gan.models.patch_classifier import build_patch_classifier


def _make_eval_loader(items, cls_cfg) -> DataLoader:
    ds = PatchDataset(items, cls_cfg, is_train=False)
    return DataLoader(ds, batch_size=cls_cfg.train.batch_size, shuffle=False,
                      num_workers=0, collate_fn=collate_patch_bags,
                      pin_memory=False, drop_last=False)


def _limit_items(items, n: int):
    """Cap an eval set at n images while keeping both classes represented."""
    if n <= 0 or len(items) <= n:
        return items
    pos = [it for it in items if it.label == 1][: max(1, n // 2)]
    neg = [it for it in items if it.label == 0][: n - len(pos)]
    return neg + pos


@torch.no_grad()
def collect_image_scores(model, loader, cls_cfg, device, tag: str):
    """Forward every bag; return image-topk scores, labels and paths."""
    model.eval()
    scores: List[float] = []
    labels: List[int] = []
    paths: List[str] = []
    total = len(loader)
    t0 = time.time()
    for bi, batch in enumerate(loader):
        flat_patches, _, sizes = flatten_bags(batch["patches"], batch["patch_labels"])
        flat_patches = flat_patches.to(device, non_blocking=True)
        with autocast(enabled=cls_cfg.train.amp and device.type == "cuda"):
            out = model(flat_patches)
        probs = out["patch_probs"].detach().float().cpu()
        for bag in split_bags(probs, sizes):
            scores.append(aggregate_topk(bag, cls_cfg.train.topk_k))
        labels.extend(batch["image_label"].tolist())
        paths.extend(batch["path"])
        if (bi + 1) % 20 == 0 or bi == 0 or (bi + 1) == total:
            print(f"[{tag}] batch {bi + 1:>4}/{total} | {time.time() - t0:.0f}s", flush=True)
    return np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=np.int64), paths


def find_spec_threshold(scores: np.ndarray, labels: np.ndarray, target_spec: float) -> float:
    """Threshold at the target_spec-th percentile of negative scores."""
    neg_scores = scores[labels == 0]
    if len(neg_scores) == 0:
        return 0.5
    return float(np.percentile(neg_scores, target_spec * 100.0))


def _write_scores_csv(path: Path, img_paths: List[str], labels: np.ndarray, scores: np.ndarray):
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "label", "pred"])
        for fn, lbl, sc in zip(img_paths, labels, scores):
            w.writerow([Path(fn).name, int(lbl), f"{sc:.8f}"])
    print(f"[csv] wrote {path} ({len(labels)} rows)", flush=True)


def _metric_row(site: str, y: np.ndarray, p: np.ndarray, thr: float, thr_source: str,
                epoch: int) -> dict:
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    auc = roc_auc(p, y) if n_pos and n_neg else float("nan")
    m = binary_metrics(p, y, threshold=thr)
    return {
        "selection": "best_auc", "epoch": epoch, "site": site, "pool": "topk",
        "thr_source": thr_source, "thr": thr, "auc": auc, "acc": m["acc"],
        "sen": m["sen"] if n_pos else float("nan"),
        "spec": m["spec"] if n_neg else float("nan"),
        "f1": m["f1"] if n_pos and n_neg else float("nan"),
        "n_pos": n_pos, "n_neg": n_neg,
    }


def _write_metric_rows(path: Path, rows: Sequence[dict]):
    fields = ["selection", "epoch", "site", "pool", "thr_source", "thr",
              "auc", "acc", "sen", "spec", "f1", "n_pos", "n_neg"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if v != v else f"{v:.6f}") if isinstance(v, float) else v
                        for k, v in r.items()})
    print(f"[csv] wrote {path} ({len(rows)} rows)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="YAML config (default: configs/default.yaml)")
    ap.add_argument("--model-dir", type=Path, default=None,
                    help="Step-3 output dir containing best_auc.pt "
                         "(default: <results_root>/step3_classifier)")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--eval-patch-chunk-size", type=int, default=24)
    ap.add_argument("--limit-images", type=int, default=0,
                    help="Cap each eval set at N images (smoke test). 0 = full.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cls_cfg = cfg.classifier
    cls_cfg.train.batch_size = args.batch_size
    cls_cfg.train.eval_patch_chunk_size = args.eval_patch_chunk_size
    cls_cfg.train.num_workers = 0
    load_site_stats(cfg.paths.site_stats)
    seed_all(cls_cfg.train.seed)

    model_dir = Path(args.model_dir or cfg.step3_output_dir()).resolve()
    ckpt_path = model_dir / "best_auc.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"[err] checkpoint not found: {ckpt_path}")
    arm = model_dir.name

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    epoch = int(ckpt.get("epoch", -1))
    model = build_patch_classifier(cls_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"[load] {ckpt_path} (ep{epoch})", flush=True)

    # ---- valid set: Youden threshold ----
    valid_items = _limit_items(build_items(cfg.datasets.valid), args.limit_images)
    print(f"[items] valid({cfg.datasets.valid.name})={len(valid_items)}", flush=True)
    v_scores, v_labels, _ = collect_image_scores(
        model, _make_eval_loader(valid_items, cls_cfg), cls_cfg, device, f"{arm}/valid")
    ts = threshold_summary(v_scores, v_labels, cls_cfg.loss.target_sen, cls_cfg.loss.target_spec)
    youden_thr = float(ts["youden_best_thr"])

    roc_dir = model_dir / "roc_scores"
    roc_dir.mkdir(exist_ok=True)
    op_rows: List[dict] = []
    final_rows: List[dict] = []

    # ---- each test set: one forward pass, then offline metrics ----
    for spec in cfg.datasets.tests:
        items = _limit_items(build_items(spec), args.limit_images)
        print(f"[items] test({spec.name})={len(items)}", flush=True)
        scores, labels, paths = collect_image_scores(
            model, _make_eval_loader(items, cls_cfg), cls_cfg, device, f"{arm}/{spec.name}")

        _write_scores_csv(
            roc_dir / f"{arm}__best_auc__{spec.name}_scores.csv", paths, labels, scores)

        if spec.threshold == "spec_target":
            op_thr = find_spec_threshold(scores, labels, spec.target_spec)
            thr_source = f"spec{int(spec.target_spec * 100)}_{spec.name}_test"
        else:
            op_thr = youden_thr
            thr_source = "youden_valid"
        op_rows.append(_metric_row(spec.name, labels, scores, op_thr, thr_source, epoch))

        final_rows.append(_metric_row(spec.name, labels, scores, youden_thr, "youden_valid", epoch))

    _write_metric_rows(model_dir / "eval_operating_points.csv", op_rows)
    _write_metric_rows(model_dir / "final_test_metrics.csv", final_rows)

    print(f"\n{'site':<24} {'thr_source':<26} {'thr':>8} {'AUC':>6} {'Sen':>6} {'Spec':>6} {'F1':>6}")
    for r in final_rows + op_rows:
        print(f"{r['site']:<24} {r['thr_source']:<26} {r['thr']:>8.4f} {r['auc']:>6.3f} "
              f"{r['sen']:>6.3f} {r['spec']:>6.3f} {r['f1']:>6.3f}")


if __name__ == "__main__":
    main()
