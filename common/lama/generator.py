"""LaMa FFC-ResNet generator with AdaIN label conditioning at the bottleneck."""
import torch
import torch.nn as nn

from .ffc import FFCResnetBlock, ConcatTupleLayer, FFC_BN_ACT
from .adain import LabelAdaIN
from .blending import blend_inpaint


class LaMaGenerator(nn.Module):
    supports_noise = False

    def __init__(self, input_nc=4, output_nc=3, ngf=64, n_downsampling=3,
                 n_blocks=9, ratio_gin=0.75, ratio_gout=0.75,
                 num_classes=2, cond_dim=64, mask_feather_radius=4):
        super().__init__()
        self.mask_feather_radius = mask_feather_radius

        self.init_block = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True),
        )

        down_plain = []
        for i in range(n_downsampling - 1):
            mult = 2 ** i
            down_plain += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(ngf * mult * 2),
                nn.ReLU(True),
            ]
        self.down_plain = nn.Sequential(*down_plain)

        self.adain = LabelAdaIN(ngf * (2 ** (n_downsampling - 1)), num_classes, cond_dim)

        in_ch = ngf * (2 ** (n_downsampling - 1))
        self.down_ffc = FFC_BN_ACT(in_ch, in_ch * 2, kernel_size=3, stride=2,
                                   padding=1, ratio_gin=0, ratio_gout=ratio_gout,
                                   activation_layer=nn.ReLU)

        mult = 2 ** n_downsampling
        self.ffc_blocks = nn.Sequential(*[
            FFCResnetBlock(ngf * mult, ratio_gin=ratio_gout, ratio_gout=ratio_gout)
            for _ in range(n_blocks)
        ])
        self.concat = ConcatTupleLayer()

        up = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            up += [
                nn.ConvTranspose2d(ngf * mult, ngf * mult // 2, kernel_size=3,
                                   stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(ngf * mult // 2),
                nn.ReLU(True),
            ]
        up += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
        ]
        self.up = nn.Sequential(*up)

    def forward(self, x, mask, labels, z):
        """x: (B,3,H,W) in [0,1]; mask: (B,1,H,W) 1=hole; labels: (B,) long
        for AdaIN conditioning; z is ignored (kept for interface parity)."""
        inp = torch.cat([x * (1 - mask), mask], dim=1)
        h = self.init_block(inp)
        h = self.down_plain(h)
        h = self.adain(h, labels)
        h = self.down_ffc(h)
        h = self.ffc_blocks(h)
        h = self.concat(h)
        out = torch.sigmoid(self.up(h))
        return blend_inpaint(x, out, mask, radius=self.mask_feather_radius)
