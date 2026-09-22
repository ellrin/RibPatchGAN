"""Wrapper around the step-2 LaMa generator for patch inpainting.

GAN input range is [0, 1] (NOT classifier-normalized): callers un-normalize
before `generate_patch` and re-normalize after (see paste.py).
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from common.lama import build_lama_generator


class PretrainedInpaintingGan(nn.Module):
    """Loads the step-2 generator checkpoint. With active_g=True the train
    loop may fine-tune G cooperatively; otherwise G stays frozen."""

    def __init__(self, checkpoint_path: Path | str, lama_arch,
                 device: Optional[torch.device] = None, active_g: bool = False):
        super().__init__()
        device = device or torch.device("cpu")
        self.noise_dim = lama_arch.noise_dim
        self.G = build_lama_generator(lama_arch)
        state = torch.load(Path(checkpoint_path), map_location=device)
        self.G.load_state_dict(state)
        self.G.to(device)
        self.set_active_g(active_g)

    def set_active_g(self, active_g: bool):
        self.active_g = bool(active_g)
        for p in self.G.parameters():
            p.requires_grad_(self.active_g)
        self.G.train(self.active_g)

    def _generate_impl(self, patch: torch.Tensor, mask: torch.Tensor,
                       labels: torch.Tensor) -> torch.Tensor:
        z = torch.randn(patch.size(0), self.noise_dim, device=patch.device)
        return self.G(patch, mask, labels, z)

    def generate_patch(self, patch: torch.Tensor, mask: torch.Tensor,
                       labels: torch.Tensor, requires_grad: bool = False) -> torch.Tensor:
        if requires_grad:
            return self._generate_impl(patch, mask, labels)
        with torch.no_grad():
            return self._generate_impl(patch, mask, labels)


# --- mask / paste helpers --------------------------------------------------------

def make_center_mask(batch: int, patch_size: int, mask_size: int,
                     device: torch.device, mask_size_jitter: int = 0) -> torch.Tensor:
    """(B, 1, patch_size, patch_size) binary mask with a centered square hole."""
    mask = torch.zeros(batch, 1, patch_size, patch_size, device=device)
    for b in range(batch):
        side = mask_size
        if mask_size_jitter > 0:
            side = side + random.randint(-mask_size_jitter, mask_size_jitter)
            side = int(max(4, min(side, patch_size - 4)))
        pad = (patch_size - side) // 2
        mask[b, :, pad:pad + side, pad:pad + side] = 1.0
    return mask


def feather_alpha_from_mask(mask: torch.Tensor) -> torch.Tensor:
    """Soft alpha: 1 inside the hole, ramping to 0 at the patch boundary, so
    pasted regions blend without a hard edge."""
    alpha = torch.zeros_like(mask)
    h, w = mask.shape[-2:]
    yy = torch.arange(h, device=mask.device, dtype=mask.dtype).view(h, 1)
    xx = torch.arange(w, device=mask.device, dtype=mask.dtype).view(1, w)
    for i in range(mask.size(0)):
        coords = mask[i, 0].nonzero(as_tuple=False)
        if coords.numel() == 0:
            continue
        y_min = int(coords[:, 0].min().item())
        y_max = int(coords[:, 0].max().item())
        x_min = int(coords[:, 1].min().item())
        x_max = int(coords[:, 1].max().item())
        outside_y = torch.maximum((y_min - yy).clamp_min(0), (yy - y_max).clamp_min(0))
        outside_x = torch.maximum((x_min - xx).clamp_min(0), (xx - x_max).clamp_min(0))
        outside = torch.maximum(outside_y, outside_x)
        ramp = float(max(y_min, x_min, h - 1 - y_max, w - 1 - x_max, 1))
        alpha[i, 0] = (1.0 - outside / ramp).clamp(0.0, 1.0)
    return alpha


def crop_center_region(patches: torch.Tensor, side: int):
    """Centered (side x side) crop; returns (region, x0, y0, side) for paste-back."""
    _, _, h, w = patches.shape
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return patches[:, :, y0:y0 + side, x0:x0 + side], x0, y0, side
