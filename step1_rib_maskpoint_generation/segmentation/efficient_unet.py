"""EfficientNet-UNet rib segmentation model."""
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class EfficientUNet(nn.Module):
    def __init__(self, model_name="efficientnet_b3", weights=None, device="cpu",
                 pre_trained_path=None, num_classes=1):
        super().__init__()
        self.device = device

        match = re.search(r"efficientnet_b(\d+)", model_name)
        if not match:
            raise ValueError("Model name must be like efficientnet_b0 ~ b6")
        model_version = int(match.group(1))
        if model_version > 6:
            raise ValueError("Only b0-b6 are supported")

        backbone_fn = getattr(models, model_name)
        self.backbone = backbone_fn(weights=weights).features.to(device)
        if pre_trained_path:
            self.backbone.load_state_dict(torch.load(pre_trained_path), strict=False)

        layer_indices_map = {
            0: [2, 3, 5, 8],
            1: [2, 3, 5, 8],
            2: [2, 3, 5, 8],
            3: [2, 4, 6, 8],
            4: [2, 5, 7, 9],
            5: [2, 5, 8, 10],
            6: [2, 6, 9, 11],
        }
        indices = layer_indices_map[model_version]
        self.encoder1 = nn.Sequential(*self.backbone[:indices[0]])
        self.encoder2 = nn.Sequential(*self.backbone[indices[0]:indices[1]])
        self.encoder3 = nn.Sequential(*self.backbone[indices[1]:indices[2]])
        self.encoder4 = nn.Sequential(*self.backbone[indices[2]:indices[3]])
        self.encoder5 = nn.Sequential(*self.backbone[indices[3]:])

        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224, device=device)
            s1 = self.encoder1(dummy)
            s2 = self.encoder2(s1)
            s3 = self.encoder3(s2)
            s4 = self.encoder4(s3)
            s5 = self.encoder5(s4)
            c1, c2, c3, c4, c5 = (s.shape[1] for s in (s1, s2, s3, s4, s5))
            del dummy, s1, s2, s3, s4, s5
            torch.cuda.empty_cache()

        self.decoder1 = self.upsample_block(c5, c4).to(device)
        self.decoder2 = self.upsample_block(c4 + c4, c3).to(device)
        self.decoder3 = self.upsample_block(c3 + c3, c2).to(device)
        self.decoder4 = self.upsample_block(c2 + c2, c1).to(device)
        self.decoder5 = self.upsample_block(c1 + c1, 32).to(device)
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1).to(device)
        self.activation = nn.Sigmoid()

    def upsample_block(self, in_channels, out_channels):
        hidden_dim = max(in_channels // 16, 8)
        hidden_dim = min(hidden_dim, 128)
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)

        d1 = self.decoder1(e5)
        d1 = torch.cat([d1, F.interpolate(e4, size=d1.shape[-2:], mode="bilinear", align_corners=False)], dim=1)
        d2 = self.decoder2(d1)
        d2 = torch.cat([d2, F.interpolate(e3, size=d2.shape[-2:], mode="bilinear", align_corners=False)], dim=1)
        d3 = self.decoder3(d2)
        d3 = torch.cat([d3, F.interpolate(e2, size=d3.shape[-2:], mode="bilinear", align_corners=False)], dim=1)
        d4 = self.decoder4(d3)
        d4 = torch.cat([d4, F.interpolate(e1, size=d4.shape[-2:], mode="bilinear", align_corners=False)], dim=1)
        d5 = self.decoder5(d4)
        return self.activation(self.final_conv(d5))


def inference_with_resize_and_restore(model: torch.nn.Module, image_tensor: torch.Tensor) -> np.ndarray:
    """Run segmentation at 768x768 to bound VRAM, then restore the binary
    mask to the original resolution."""
    if image_tensor.shape[0] == 0:
        return np.array([], dtype=np.int32).reshape(0, 1, 0, 0)

    _, _, h_orig, w_orig = image_tensor.shape
    resized = F.interpolate(image_tensor, size=(768, 768), mode="bilinear", align_corners=False)
    with torch.no_grad():
        logits = model(resized)
    restored = F.interpolate(logits, size=(h_orig, w_orig), mode="bilinear", align_corners=False)
    return (restored > 0.5).int().cpu().numpy()
