"""Shared loaders, scheduler, dynamic loss weights and evaluation loop."""
from __future__ import annotations

import math
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from .data.dataset import FxBalancedBatchSampler, PatchDataset, collate_patch_bags
from .metrics import compute_eval_metrics


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(items, cls_cfg, is_train: bool) -> DataLoader:
    ds = PatchDataset(items, cls_cfg, is_train=is_train)
    if is_train and cls_cfg.sampling.use_fx_balanced_batches:
        sampler = FxBalancedBatchSampler(items, cls_cfg.train.batch_size,
                                         cls_cfg.sampling.fx_bags_per_batch,
                                         seed=cls_cfg.train.seed)
        return DataLoader(ds, batch_sampler=sampler, num_workers=cls_cfg.train.num_workers,
                          collate_fn=collate_patch_bags, pin_memory=True)
    if is_train:
        return DataLoader(ds, batch_size=cls_cfg.train.batch_size, shuffle=True,
                          num_workers=cls_cfg.train.num_workers, collate_fn=collate_patch_bags,
                          pin_memory=True, drop_last=True)
    # Eval must stay single-process: DataLoader worker IPC has leaked /dev/shm
    # to ~90 GB on a 9.7k-image test set, and the GPU forward is the
    # bottleneck anyway.
    return DataLoader(ds, batch_size=cls_cfg.train.batch_size, shuffle=False,
                      num_workers=0, collate_fn=collate_patch_bags,
                      pin_memory=False, drop_last=False)


def make_warmup_cosine_scheduler(optimizer, total_steps: int, warmup_steps: int):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def dynamic_loss_weights(prev_metrics: Dict[str, float] | None, cls_cfg) -> Tuple[float, float]:
    """Valid-Youden controller: raise w_fg when sensitivity falls short of the
    target, w_bg when specificity does."""
    loss_cfg = cls_cfg.loss
    if not loss_cfg.use_dynamic_weight or not prev_metrics:
        return loss_cfg.w_fg, loss_cfg.w_bg
    sen = prev_metrics.get("image_topk_youden_best_sen") or prev_metrics.get("patch_youden_best_sen")
    spec = prev_metrics.get("image_topk_youden_best_spec") or prev_metrics.get("patch_youden_best_spec")
    if sen is None or spec is None or (isinstance(sen, float) and math.isnan(sen)):
        return loss_cfg.w_fg, loss_cfg.w_bg
    eta = loss_cfg.dyn_weight_eta
    lo, hi = loss_cfg.dyn_weight_min, loss_cfg.dyn_weight_max
    w_fg = float(np.clip(1.0 + eta * max(0.0, loss_cfg.target_sen - sen), lo, hi))
    w_bg = float(np.clip(1.0 + eta * max(0.0, loss_cfg.target_spec - spec), lo, hi))
    return w_fg, w_bg


def flatten_bags(patches_list: List[torch.Tensor], labels_list: List[torch.Tensor]):
    """Concatenate per-image bags into one flat tensor; remember bag sizes."""
    sizes = [p.size(0) for p in patches_list]
    return torch.cat(patches_list, dim=0), torch.cat(labels_list, dim=0), sizes


def split_bags(flat: torch.Tensor, sizes: List[int]) -> List[torch.Tensor]:
    out, cur = [], 0
    for s in sizes:
        out.append(flat[cur:cur + s])
        cur += s
    return out


@torch.no_grad()
def evaluate(model, loader, cls_cfg, device, log_prefix: str | None = None,
             log_every: int = 0) -> Dict[str, float]:
    model.eval()
    bag_probs: List[torch.Tensor] = []
    bag_labels: List[torch.Tensor] = []
    image_labels: List[int] = []
    sites: List[str] = []
    max_batches = int(cls_cfg.train.max_eval_batches or 0)
    total_batches = len(loader)
    for bi, batch in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        flat_patches, _, sizes = flatten_bags(batch["patches"], batch["patch_labels"])
        flat_patches = flat_patches.to(device, non_blocking=True)
        with autocast(enabled=cls_cfg.train.amp and device.type == "cuda"):
            out = model(flat_patches)
        probs = out["patch_probs"].detach().float().cpu()
        bag_probs.extend(split_bags(probs, sizes))
        bag_labels.extend(batch["patch_labels"])
        image_labels.extend(batch["image_label"].tolist())
        sites.extend(batch["site"])
        if log_prefix and log_every and ((bi + 1) == 1 or (bi + 1) % log_every == 0 or (bi + 1) == total_batches):
            print(f"{log_prefix} batch {bi + 1}/{total_batches}", flush=True)
    if not bag_probs:
        return {}
    return compute_eval_metrics(
        bag_probs=bag_probs,
        bag_patch_labels=bag_labels,
        image_labels=torch.tensor(image_labels, dtype=torch.long),
        sites=sites,
        cfg=cls_cfg,
    )
