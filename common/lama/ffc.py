"""Fast Fourier Convolution (FFC) modules for LaMa.

Adapted from LaMa, saicinpainting/training/modules/ffc.py
    https://github.com/advimman/lama
    Copyright 2021 Samsung Research
    Licensed under the Apache License, Version 2.0
LaMa in turn adapts the original FFC implementation (Chi et al., NeurIPS 2020),
    https://github.com/pkumivision/FFC

Changes from the original file: BatchNorm2d replaced with InstanceNorm2d for
WGAN-GP compatibility, and the blocks unused here (FFCSE_block,
FFCResNetGenerator, FFCNLayerDiscriminator) removed.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft


class FourierUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1,
                 spectral_pos_encoding=False, fft_norm='ortho'):
        super().__init__()
        self.groups = groups
        self.conv_layer = nn.Conv2d(
            in_channels=in_channels * 2 + (2 if spectral_pos_encoding else 0),
            out_channels=out_channels * 2,
            kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.in_norm = nn.InstanceNorm2d(in_channels * 2 + (2 if spectral_pos_encoding else 0))
        self.relu = nn.ReLU(inplace=False)
        self.spectral_pos_encoding = spectral_pos_encoding
        self.fft_norm = fft_norm

    def forward(self, x):
        batch = x.shape[0]
        r_size = x.size()
        ffted = torch.fft.rfftn(x, dim=(-2, -1), norm=self.fft_norm)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        if self.spectral_pos_encoding:
            height, width = ffted.shape[-2:]
            coords_vert = torch.linspace(0, 1, height)[None, None, :, None].expand(batch, 1, height, width).to(ffted)
            coords_hor = torch.linspace(0, 1, width)[None, None, None, :].expand(batch, 1, height, width).to(ffted)
            ffted = torch.cat((coords_vert, coords_hor, ffted), dim=1)

        ffted = self.in_norm(ffted)
        ffted = self.conv_layer(ffted)
        ffted = self.relu(ffted)

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(0, 1, 3, 4, 2).contiguous()
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        output = torch.fft.irfftn(ffted, s=x.shape[-2:], dim=(-2, -1), norm=self.fft_norm)
        return output


class SpectralTransform(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=True):
        super().__init__()
        self.enable_lfu = enable_lfu
        if stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.downsample = nn.Identity()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=1, groups=groups, bias=False),
            nn.InstanceNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.fu = FourierUnit(out_channels // 2, out_channels // 2, groups)
        if self.enable_lfu:
            self.lfu = FourierUnit(out_channels // 2, out_channels // 2, groups)
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)

    def forward(self, x):
        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)

        if self.enable_lfu:
            n, c, h, w = x.shape
            split_no = 2
            split_s = h // split_no
            if split_s > 0 and c // 4 > 0:
                xs = torch.cat(torch.split(x[:, :c // 4], split_s, dim=-2), dim=1).contiguous()
                xs = torch.cat(torch.split(xs, split_s, dim=-1), dim=1).contiguous()
                xs = self.lfu(xs)
                xs = xs.repeat(1, 1, split_no, split_no).contiguous()
            else:
                xs = 0
        else:
            xs = 0

        output = self.conv2(x + output + xs)
        return output


class FFC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin, ratio_gout, stride=1, padding=0,
                 dilation=1, groups=1, bias=False,
                 enable_lfu=True, padding_type='reflect'):
        super().__init__()
        assert stride == 1 or stride == 2

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        self.global_in_num = in_cg

        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convl2l = module(in_cl, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        self.convl2g = module(in_cl, out_cg, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
        self.convg2l = module(in_cg, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g = module(in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu)

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl, out_xg = 0, 0

        if self.ratio_gout != 1:
            out_xl = self.convl2l(x_l) + self.convg2l(x_g)
        if self.ratio_gout != 0:
            out_xg = self.convl2g(x_l) + self.convg2g(x_g)

        return out_xl, out_xg


class FFC_BN_ACT(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, ratio_gin, ratio_gout,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,
                 activation_layer=nn.ReLU, enable_lfu=True):
        super().__init__()
        self.ffc = FFC(in_channels, out_channels, kernel_size,
                       ratio_gin, ratio_gout, stride, padding,
                       dilation, groups, bias, enable_lfu)

        lnorm = int(out_channels * (1 - ratio_gout))
        gnorm = int(out_channels * ratio_gout)

        self.bn_l = nn.InstanceNorm2d(lnorm) if lnorm > 0 else nn.Identity()
        self.bn_g = nn.InstanceNorm2d(gnorm) if gnorm > 0 else nn.Identity()
        self.act_l = activation_layer(inplace=True) if lnorm > 0 else nn.Identity()
        self.act_g = activation_layer(inplace=True) if gnorm > 0 else nn.Identity()

    def forward(self, x):
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))
        return x_l, x_g


class FFCResnetBlock(nn.Module):
    def __init__(self, dim, padding_type='reflect', activation_layer=nn.ReLU,
                 dilation=1, ratio_gin=0.75, ratio_gout=0.75):
        super().__init__()
        self.conv1 = FFC_BN_ACT(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
                                ratio_gin=ratio_gin, ratio_gout=ratio_gout,
                                activation_layer=activation_layer)
        self.conv2 = FFC_BN_ACT(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
                                ratio_gin=ratio_gin, ratio_gout=ratio_gout,
                                activation_layer=activation_layer)

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        id_l, id_g = x_l, x_g
        x_l, x_g = self.conv1((x_l, x_g))
        x_l, x_g = self.conv2((x_l, x_g))
        x_l, x_g = id_l + x_l, id_g + x_g
        return x_l, x_g


class ConcatTupleLayer(nn.Module):
    def forward(self, x):
        assert isinstance(x, tuple)
        x_l, x_g = x
        assert torch.is_tensor(x_l)
        if not torch.is_tensor(x_g):
            return x_l
        return torch.cat([x_l, x_g], dim=1)
