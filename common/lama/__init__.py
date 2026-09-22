"""LaMa inpainting model shared by step 2 (training) and step 3 (frozen /
cooperatively fine-tuned augmentation generator)."""
from .discriminator import NLayerDiscriminator
from .generator import LaMaGenerator


def build_lama_generator(arch) -> LaMaGenerator:
    """Build the generator from config.LamaArchitectureConfig so steps 2 and 3
    always construct checkpoint-compatible models."""
    return LaMaGenerator(
        ngf=arch.ngf,
        n_downsampling=arch.n_downsampling,
        n_blocks=arch.n_blocks,
        ratio_gin=arch.ratio_g,
        ratio_gout=arch.ratio_g,
        num_classes=arch.num_classes,
        mask_feather_radius=arch.mask_feather_radius,
    )


def build_lama_discriminator(arch) -> NLayerDiscriminator:
    return NLayerDiscriminator(input_nc=3, ndf=64, n_layers=4)
