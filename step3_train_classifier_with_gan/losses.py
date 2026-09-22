"""Classifier loss."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def patch_bce(patch_logits: torch.Tensor, patch_labels: torch.Tensor,
              w_fg: float = 1.0, w_bg: float = 1.0) -> torch.Tensor:
    """BCE-with-logits split into fg/bg pools that are averaged separately
    before averaging the pools, so rare positives are not drowned out by the
    common negatives. w_fg/w_bg come from the valid-Youden controller."""
    if patch_logits.numel() == 0:
        return patch_logits.new_tensor(0.0)
    targets = patch_labels.float()
    raw = F.binary_cross_entropy_with_logits(patch_logits, targets, reduction="none")
    fg = targets > 0.5
    bg = ~fg
    pieces = []
    if fg.any():
        pieces.append(float(w_fg) * raw[fg].mean())
    if bg.any():
        pieces.append(float(w_bg) * raw[bg].mean())
    if not pieces:
        return raw.mean()
    return torch.stack(pieces).mean()
