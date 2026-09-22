"""N-layer PatchGAN discriminator (InstanceNorm). Used only during generator
pretraining (Section III-C, "Generator pretraining")."""
import numpy as np
import torch.nn as nn


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=64, n_layers=4):
        super().__init__()
        self.n_layers = n_layers
        kw, padw = 4, int(np.ceil((4 - 1.0) / 2))

        sequence = [nn.Sequential(
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, inplace=True),
        )]
        nf = ndf
        for _ in range(1, n_layers):
            nf_prev, nf = nf, min(nf * 2, 512)
            sequence.append(nn.Sequential(
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                nn.InstanceNorm2d(nf),
                nn.LeakyReLU(0.2, inplace=True),
            ))
        nf_prev, nf = nf, min(nf * 2, 512)
        sequence.append(nn.Sequential(
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            nn.InstanceNorm2d(nf),
            nn.LeakyReLU(0.2, inplace=True),
        ))
        sequence.append(nn.Sequential(
            nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw),
        ))
        for i, s in enumerate(sequence):
            setattr(self, f"model{i}", s)

    def forward(self, x):
        for i in range(self.n_layers + 2):
            x = getattr(self, f"model{i}")(x)
        return x
