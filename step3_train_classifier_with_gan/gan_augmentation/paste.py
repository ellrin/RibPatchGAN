"""Synthetic-fracture paste (mixed_pos).

Operates on FLAT patches (N_total, C, H, W) with per-patch labels. Selected
patches get their centered 128 region un-normalized, inpainted by the
fx-conditioned generator, feather-pasted back and re-normalized; all
generated patches are labelled 1.
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import torch

from ..data.preprocessing import denormalize_tensor, normalize_tensor
from .generator_wrapper import (
    PretrainedInpaintingGan,
    crop_center_region,
    feather_alpha_from_mask,
    make_center_mask,
)


def _patch_quality(region: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    gray = region.mean(dim=1)
    return gray.std(dim=(1, 2), unbiased=False), gray.mean(dim=(1, 2))


def quality_mask(region: torch.Tensor, min_std=0.03, min_mean=0.02, max_mean=0.98) -> torch.Tensor:
    std, mean = _patch_quality(region)
    return std.ge(min_std) & mean.ge(min_mean) & mean.le(max_mean)


def select_paste_targets(
    patch_labels: torch.Tensor,
    image_label_per_patch: torch.Tensor,
    region_unnorm: torch.Tensor,
    mode: str,
    fraction: float,
) -> torch.Tensor:
    """Boolean (N,) of patches to inpaint, after quality gating.
    real_pos: positive patches on fx images (stay labelled 1).
    neg_to_pos: negative patches on nofx images (flipped to 1, counterfactual).
    """
    if mode not in ("real_pos", "neg_to_pos"):
        raise ValueError(f"unknown paste mode {mode}")
    if fraction <= 0:
        return torch.zeros_like(patch_labels, dtype=torch.bool)

    qok = quality_mask(region_unnorm)
    if mode == "real_pos":
        candidate = patch_labels.eq(1) & image_label_per_patch.eq(1) & qok
    else:
        candidate = patch_labels.eq(0) & image_label_per_patch.eq(0) & qok

    n_candidate = int(candidate.sum().item())
    if n_candidate == 0:
        return candidate
    n_select = min(max(1, int(math.ceil(n_candidate * fraction))), n_candidate)
    idx = torch.nonzero(candidate, as_tuple=False).flatten()
    perm = torch.randperm(idx.numel(), device=idx.device)[:n_select]
    out = torch.zeros_like(patch_labels, dtype=torch.bool)
    out[idx[perm]] = True
    return out


def fake_fx_paste_flat(
    gan: PretrainedInpaintingGan,
    patches: torch.Tensor,
    patch_labels: torch.Tensor,
    image_label_per_patch: torch.Tensor,
    ga_cfg,
    fraction: float,
    counterfactual_fraction: float,
    requires_grad: bool = False,
) -> Dict[str, torch.Tensor]:
    """mixed_pos paste: synthetic positives from real fx locations plus a
    smaller nofx->fx counterfactual stream. Returns aux_patches/aux_labels/
    fake_mask plus the raw generator tensors needed by the G-update path."""
    device = patches.device
    side = int(ga_cfg.gan_patch_size)

    unnorm = denormalize_tensor(patches)
    region, x0, y0, _ = crop_center_region(unnorm, side)

    positive_select = select_paste_targets(
        patch_labels, image_label_per_patch, region, "real_pos", fraction,
    )
    counter_select = select_paste_targets(
        patch_labels, image_label_per_patch, region, "neg_to_pos", counterfactual_fraction,
    ) & ~positive_select
    select = positive_select | counter_select
    stats = {
        "fake_candidates": float(select.numel()),
        "fake_used": float(select.sum().item()),
        "fake_pos_used": float(positive_select.sum().item()),
        "fake_counterfactual_pos_used": float(counter_select.sum().item()),
    }
    if not select.any():
        return {
            "aux_patches": patches,
            "aux_labels": patch_labels,
            "fake_mask": select,
            "stats": stats,
        }

    sel_idx = torch.nonzero(select, as_tuple=False).flatten()
    sel_region = region[sel_idx]

    mask = make_center_mask(sel_region.size(0), side, int(ga_cfg.mask_size),
                            device, mask_size_jitter=int(ga_cfg.mask_size_jitter))

    gan_labels = torch.ones(sel_region.size(0), dtype=torch.long, device=device)
    fake_region = gan.generate_patch(sel_region, mask, gan_labels, requires_grad=requires_grad)
    fake_region = torch.nan_to_num(fake_region, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

    alpha = feather_alpha_from_mask(mask)
    pasted_region = alpha * fake_region + (1.0 - alpha) * sel_region
    aux_unnorm = unnorm.clone()
    aux_unnorm[sel_idx, :, y0:y0 + side, x0:x0 + side] = pasted_region
    aux_patches = normalize_tensor(aux_unnorm)

    aux_labels = patch_labels.clone()
    aux_labels[select] = 1.0

    return {
        "aux_patches": aux_patches,
        "aux_labels": aux_labels,
        "fake_mask": select,
        "stats": stats,
        "_sel_region_unnorm": sel_region,
        "_fake_region_unnorm": fake_region,
        "_paste_mask": mask,
        "_sel_idx": sel_idx,
    }
