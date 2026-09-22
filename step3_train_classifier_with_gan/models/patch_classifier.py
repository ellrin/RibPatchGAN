"""ConvNeXt-Tiny patch classifier: encoder + dropout + binary head.

forward() takes a FLAT batch of patches (N_total, C, H, W) and chunks the
encoder forward to bound activation memory; callers reconstruct per-image
bags via n_patches.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class ConvNeXtTinyBackbone(nn.Module):
    """torchvision ConvNeXt-Tiny features -> AvgPool -> LayerNorm2d -> Flatten."""

    def __init__(self, pretrained: bool):
        super().__init__()
        import torchvision.models as tvm
        weights = tvm.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        base = tvm.convnext_tiny(weights=weights)
        self.net = nn.Sequential(
            base.features,
            base.avgpool,
            base.classifier[0],
            nn.Flatten(),
        )
        with torch.no_grad():
            self.out_dim = self.net(torch.zeros(1, 3, 224, 224)).shape[1]

    def forward(self, x):
        return self.net(x)


class PatchClassifier(nn.Module):
    def __init__(self, encoder: str = "convnext_tiny", pretrained: bool = True,
                 dropout: float = 0.4, patch_chunk_size: int = 32,
                 eval_chunk_size: int | None = None):
        super().__init__()
        if encoder != "convnext_tiny":
            raise ValueError(f"v5 keeps only the final encoder (convnext_tiny), got {encoder}")
        self.backbone = ConvNeXtTinyBackbone(pretrained)
        self.patch_chunk_size = patch_chunk_size
        self.eval_chunk_size = int(eval_chunk_size) if eval_chunk_size else patch_chunk_size
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.backbone.out_dim, 1),
        )

    def forward(self, patches_flat: torch.Tensor) -> Dict[str, torch.Tensor]:
        emb = self._encode_chunks(patches_flat)
        logits = self.head(emb).squeeze(-1)
        return {
            "patch_logits": logits,
            "patch_probs": torch.sigmoid(logits),
            "patch_embedding": emb,
        }

    def _encode_chunks(self, flat: torch.Tensor) -> torch.Tensor:
        chunk = self.patch_chunk_size if self.training else self.eval_chunk_size
        if chunk <= 0 or flat.size(0) <= chunk:
            return self.backbone(flat)
        outs = []
        for start in range(0, flat.size(0), chunk):
            outs.append(self.backbone(flat[start:start + chunk]))
        return torch.cat(outs, dim=0)


def build_patch_classifier(cls_cfg) -> PatchClassifier:
    return PatchClassifier(
        encoder=cls_cfg.model.encoder,
        pretrained=bool(cls_cfg.model.pretrained),
        dropout=cls_cfg.model.dropout,
        patch_chunk_size=cls_cfg.train.patch_forward_chunk_size,
        eval_chunk_size=cls_cfg.train.eval_patch_chunk_size,
    )
