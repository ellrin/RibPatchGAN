"""Patch- and image-level metrics with threshold sweeps."""
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import torch


# --- image-level pooling ------------------------------------------------------

def aggregate_topk(probs: torch.Tensor, k: int) -> float:
    """s(x) = mean of the top-k patch probabilities."""
    if probs.numel() == 0:
        return float("nan")
    k = max(1, min(int(k), int(probs.numel())))
    return float(probs.topk(k).values.mean().item())


# --- scalar metrics ------------------------------------------------------------

def roc_auc(scores: np.ndarray, y: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    pos = scores[y == 1]
    neg = scores[y == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[y == 1].sum() - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def binary_metrics(probs, targets, threshold: float = 0.5) -> Dict[str, float]:
    p = np.asarray(probs, dtype=np.float64).reshape(-1)
    y = np.asarray(targets, dtype=np.int64).reshape(-1)
    pred = (p >= threshold).astype(np.int64)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    sen = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    f1 = 2 * prec * sen / max(prec + sen, 1e-12)
    return {"acc": acc, "sen": sen, "spec": spec, "f1": f1, "auc": roc_auc(p, y)}


def _threshold_row(p, y, thr) -> Dict[str, float]:
    m = binary_metrics(p, y, threshold=thr)
    return {
        "thr": float(thr), "acc": m["acc"], "sen": m["sen"], "spec": m["spec"],
        "f1": m["f1"], "youden": m["sen"] + m["spec"] - 1.0,
        "bal_acc": 0.5 * (m["sen"] + m["spec"]),
    }


def threshold_summary(probs, targets, target_sen: float = 0.70, target_spec: float = 0.80) -> Dict[str, float]:
    """Sweep all thresholds; return F1-best / Youden-best / target-sen /
    target-spec operating points."""
    p = np.asarray(probs, dtype=np.float64).reshape(-1)
    if p.size == 0:
        out = {}
        for k in ("best", "f1_best", "youden_best", "target_sen", "target_spec"):
            for suf in ("_thr", "_acc", "_sen", "_spec", "_f1"):
                out[f"{k}{suf}"] = float("nan")
        out["youden_best_youden"] = float("nan")
        out["youden_best_bal_acc"] = float("nan")
        return out
    y = np.asarray(targets, dtype=np.int64).reshape(-1)
    thrs = np.unique(p)
    thrs = np.concatenate([
        np.array([np.nextafter(thrs[0], -np.inf)]),
        thrs,
        np.array([np.nextafter(thrs[-1], np.inf)]),
    ])
    rows = [_threshold_row(p, y, float(t)) for t in thrs]
    f1_best = max(rows, key=lambda r: (r["f1"], r["acc"], r["thr"]))
    yo_best = max(rows, key=lambda r: (r["youden"], r["sen"], r["spec"], -r["thr"]))
    sen_rows = [r for r in rows if r["sen"] >= target_sen]
    sen_best = (max(sen_rows, key=lambda r: (r["spec"], r["sen"], r["f1"], -r["thr"]))
                if sen_rows else max(rows, key=lambda r: (r["sen"], r["spec"], r["f1"], -r["thr"])))
    spec_rows = [r for r in rows if r["spec"] >= target_spec]
    spec_best = (max(spec_rows, key=lambda r: (r["sen"], r["spec"], r["f1"], -r["thr"]))
                 if spec_rows else max(rows, key=lambda r: (r["spec"], r["sen"], r["f1"], -r["thr"])))
    return {
        "best_thr": f1_best["thr"], "best_acc": f1_best["acc"], "best_sen": f1_best["sen"],
        "best_spec": f1_best["spec"], "best_f1": f1_best["f1"],
        "f1_best_thr": f1_best["thr"], "f1_best_acc": f1_best["acc"], "f1_best_sen": f1_best["sen"],
        "f1_best_spec": f1_best["spec"], "f1_best_f1": f1_best["f1"],
        "youden_best_thr": yo_best["thr"], "youden_best_acc": yo_best["acc"],
        "youden_best_sen": yo_best["sen"], "youden_best_spec": yo_best["spec"],
        "youden_best_f1": yo_best["f1"], "youden_best_youden": yo_best["youden"],
        "youden_best_bal_acc": yo_best["bal_acc"],
        "target_sen_thr": sen_best["thr"], "target_sen_acc": sen_best["acc"],
        "target_sen_sen": sen_best["sen"], "target_sen_spec": sen_best["spec"],
        "target_sen_f1": sen_best["f1"],
        "target_spec_thr": spec_best["thr"], "target_spec_acc": spec_best["acc"],
        "target_spec_sen": spec_best["sen"], "target_spec_spec": spec_best["spec"],
        "target_spec_f1": spec_best["f1"],
    }


def compute_eval_metrics(
    bag_probs: List[torch.Tensor],
    bag_patch_labels: List[torch.Tensor],
    image_labels: torch.Tensor,
    sites: List[str] | None,
    cfg,
) -> Dict[str, float]:
    """Patch-level metrics plus top-k image-level aggregation (eq. s(x)),
    overall and per site (prefix `<site>_`)."""
    out: Dict[str, float] = {}
    targets_per_site = {"__all__": list(range(len(bag_probs)))}
    if sites is not None:
        for s in set(sites):
            targets_per_site[s] = [i for i, x in enumerate(sites) if x == s]

    for site_key, idxs in targets_per_site.items():
        if not idxs:
            continue
        prefix = "" if site_key == "__all__" else f"{site_key}_"

        patch_probs = torch.cat([bag_probs[i] for i in idxs])
        patch_lbls = torch.cat([bag_patch_labels[i] for i in idxs])
        p_np = patch_probs.detach().cpu().numpy()
        l_np = patch_lbls.detach().cpu().numpy()
        m = binary_metrics(p_np, l_np)
        thr_p = threshold_summary(p_np, l_np, cfg.loss.target_sen, cfg.loss.target_spec)
        for k in ("acc", "sen", "spec", "f1", "auc"):
            out[f"{prefix}patch_{k}"] = m[k]
        for k, v in thr_p.items():
            out[f"{prefix}patch_{k}"] = v

        img_lbls = image_labels[idxs].detach().cpu().numpy() if isinstance(image_labels, torch.Tensor) \
            else np.asarray([image_labels[i] for i in idxs])
        img_probs = np.array(
            [aggregate_topk(bag_probs[i], cfg.train.topk_k) for i in idxs], dtype=np.float64)
        m = binary_metrics(img_probs, img_lbls)
        thr = threshold_summary(img_probs, img_lbls, cfg.loss.target_sen, cfg.loss.target_spec)
        for k in ("acc", "sen", "spec", "f1", "auc"):
            out[f"{prefix}image_topk_{k}"] = m[k]
        for k, v in thr.items():
            out[f"{prefix}image_topk_{k}"] = v
    return out


def fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and math.isnan(v):
        return ""
    return f"{float(v):.6f}" if isinstance(v, float) else str(v)
