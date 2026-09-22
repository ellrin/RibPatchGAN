"""Losses for non-saturating GAN training with lazy R1."""
import torch
import torch.nn as nn
import torchvision.models as tvm


def r1_penalty(real_logits, real_imgs):
    """R1 = ||grad_x D(real)||^2. real_imgs must have requires_grad=True."""
    grad = torch.autograd.grad(
        outputs=real_logits.sum(), inputs=real_imgs,
        create_graph=True, retain_graph=True, only_inputs=True,
    )[0]
    return grad.square().sum(dim=[1, 2, 3]).mean()


class VGGLoss(nn.Module):
    """VGG19 perceptual + style loss on relu1_2 / relu2_2 / relu3_3.
    Inputs in [0, 1]; returns (perceptual, style)."""
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    SLICES = [4, 9, 18]

    def __init__(self, device):
        super().__init__()
        vgg = tvm.vgg19(weights=tvm.VGG19_Weights.IMAGENET1K_V1).features.to(device).eval()
        for p in vgg.parameters():
            p.requires_grad_(False)
        self.slices = nn.ModuleList([
            nn.Sequential(*list(vgg.children())[:self.SLICES[0] + 1]),
            nn.Sequential(*list(vgg.children())[self.SLICES[0] + 1:self.SLICES[1] + 1]),
            nn.Sequential(*list(vgg.children())[self.SLICES[1] + 1:self.SLICES[2] + 1]),
        ])
        self.register_buffer("mean", torch.tensor(self.MEAN, device=device).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(self.STD, device=device).view(1, 3, 1, 1))

    def _features(self, x):
        x = (x - self.mean) / self.std
        feats = []
        for s in self.slices:
            x = s(x)
            feats.append(x)
        return feats

    @staticmethod
    def _gram(f):
        b, c, h, w = f.shape
        f = f.view(b, c, h * w)
        return torch.bmm(f, f.transpose(1, 2)) / (c * h * w)

    def forward(self, fake, real):
        ff = self._features(fake)
        rf = self._features(real)
        percep = sum(torch.mean(torch.abs(f - r)) for f, r in zip(ff, rf))
        style = sum(torch.mean(torch.abs(self._gram(f) - self._gram(r))) for f, r in zip(ff, rf))
        return percep, style
