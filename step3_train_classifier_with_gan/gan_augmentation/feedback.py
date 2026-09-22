"""EMA teacher and cooperative (distribution-stable) G feedback.

G receives no classifier BCE / adversarial objective. The frozen EMA teacher
encodes generated and real-positive patches; G matches their feature mean and
diagonal variance, with a one-sided variance floor against diversity collapse
and an out-of-mask L1 content guard.
"""
from __future__ import annotations

import copy
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from .generator_wrapper import PretrainedInpaintingGan
from .paste import fake_fx_paste_flat


def build_ema_teacher(model: nn.Module) -> nn.Module:
    teacher = copy.deepcopy(model)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


@torch.no_grad()
def ema_update_teacher(teacher: nn.Module, student: nn.Module, m: float):
    """theta_bar <- m * theta_bar + (1 - m) * theta"""
    for p_t, p_s in zip(teacher.parameters(), student.parameters()):
        p_t.data.mul_(m).add_(p_s.data, alpha=1.0 - m)
    for b_t, b_s in zip(teacher.buffers(), student.buffers()):
        b_t.data.copy_(b_s.data)


def cooperative_g_update(
    teacher: nn.Module,
    gan: PretrainedInpaintingGan,
    g_optimizer,
    g_scaler,
    patches: torch.Tensor,
    patch_labels: torch.Tensor,
    image_label_per_patch: torch.Tensor,
    ga_cfg,
    amp_enabled: bool,
    counterfactual_fraction: Optional[float] = None,
) -> Dict[str, torch.Tensor]:
    """One distribution-stable G step."""
    zero = patches.new_tensor(0.0)
    none_stats = {
        "loss_g": zero, "loss_g_recon": zero,
        "loss_dsr_mean": zero, "loss_dsr_var": zero,
        "loss_dsr_floor": zero, "g_n_fake": zero,
    }
    if gan is None or g_optimizer is None:
        return none_stats

    if counterfactual_fraction is None:
        counterfactual_fraction = float(ga_cfg.counterfactual_fraction)
    g_optimizer.zero_grad(set_to_none=True)
    with autocast(enabled=amp_enabled):
        fakes = fake_fx_paste_flat(
            gan, patches, patch_labels, image_label_per_patch, ga_cfg,
            fraction=float(ga_cfg.fake_fraction),
            counterfactual_fraction=float(counterfactual_fraction),
            requires_grad=True,
        )
        if not fakes["fake_mask"].any():
            return none_stats

        # Frozen teacher parameters still propagate gradients into G's pixels.
        fake_out = teacher(fakes["aux_patches"])
        with torch.no_grad():
            real_out = teacher(patches)

        fake_mask = fakes["fake_mask"]
        sel_region = fakes["_sel_region_unnorm"]
        fake_region = fakes["_fake_region_unnorm"]
        paste_mask = fakes["_paste_mask"]

        preserve_mask = (1.0 - paste_mask).expand_as(fake_region)
        loss_g_recon = (
            (torch.abs(fake_region - sel_region) * preserve_mask).sum()
            / preserve_mask.sum().clamp_min(1.0)
        )

        real_pos_mask = patch_labels.eq(1) & image_label_per_patch.eq(1)
        if not real_pos_mask.any():
            return none_stats
        fake_emb = F.normalize(fake_out["patch_embedding"].float()[fake_mask], dim=-1)
        real_emb = F.normalize(
            real_out["patch_embedding"].float()[real_pos_mask].detach(), dim=-1
        )
        # DSR: mean matching + variance matching + one-sided variance floor
        term_mean = (fake_emb.mean(dim=0) - real_emb.mean(dim=0)).pow(2).sum()
        term_var = zero
        term_floor = zero
        if fake_emb.size(0) > 1 and real_emb.size(0) > 1:
            fake_var = fake_emb.var(dim=0, unbiased=False)
            real_var = real_emb.var(dim=0, unbiased=False)
            term_var = (fake_var - real_var).pow(2).sum()
            term_floor = F.relu(real_var - fake_var).sum()
        loss_dsr = (
            float(ga_cfg.lambda_m) * term_mean
            + float(ga_cfg.lambda_v) * term_var
            + float(ga_cfg.lambda_f) * term_floor
        )
        loss_g = loss_dsr + float(ga_cfg.lambda_r) * loss_g_recon

    n_fake = patches.new_tensor(float(fake_mask.sum().item()))
    stats = {
        "loss_g": loss_g.detach(),
        "loss_g_recon": loss_g_recon.detach(),
        "loss_dsr_mean": term_mean.detach(),
        "loss_dsr_var": term_var.detach(),
        "loss_dsr_floor": term_floor.detach(),
        "g_n_fake": n_fake,
    }
    if not loss_g.requires_grad or not torch.isfinite(loss_g):
        return stats

    g_scaler.scale(loss_g).backward()
    g_scaler.unscale_(g_optimizer)
    torch.nn.utils.clip_grad_norm_(gan.G.parameters(), float(ga_cfg.g_grad_clip))
    g_scaler.step(g_optimizer)
    g_scaler.update()
    return stats


def per_patch_image_label(image_labels: torch.Tensor, sizes: List[int]) -> torch.Tensor:
    """Replicate each image's label across its patches -> (N_total,)."""
    out = []
    for lab, s in zip(image_labels.tolist(), sizes):
        out.append(torch.full((int(s),), int(lab), dtype=torch.long))
    if not out:
        return torch.empty(0, dtype=torch.long)
    return torch.cat(out, dim=0).to(image_labels.device)
