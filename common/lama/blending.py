"""Mask feathering / compositing helpers shared by the generator and losses."""
import torch
import torch.nn.functional as F


def feather_mask(mask: torch.Tensor, radius: int = 4) -> torch.Tensor:
    """Soft alpha mask with a small transition band around the hole."""
    if radius <= 0:
        return mask
    kernel = radius * 2 + 1
    return F.avg_pool2d(mask, kernel_size=kernel, stride=1, padding=radius)


def blend_inpaint(real: torch.Tensor, generated: torch.Tensor, mask: torch.Tensor,
                  radius: int = 4) -> torch.Tensor:
    alpha = feather_mask(mask, radius)
    return generated * alpha + real * (1 - alpha)


def boundary_band(mask: torch.Tensor, width: int = 4) -> torch.Tensor:
    """Pixels near the binary mask edge, covering both sides of the seam."""
    if width <= 0:
        return mask * 0.0
    kernel = width * 2 + 1
    dilated = F.max_pool2d(mask, kernel_size=kernel, stride=1, padding=width)
    eroded = -F.max_pool2d(-mask, kernel_size=kernel, stride=1, padding=width)
    return (dilated - eroded).clamp(0, 1)


def seam_loss(fake: torch.Tensor, real: torch.Tensor, mask: torch.Tensor,
              weight: float = 0.5, band_width: int = 4) -> torch.Tensor:
    """L1 intensity + finite-difference gradient match around the mask seam."""
    if weight <= 0:
        return fake.sum() * 0.0

    band = boundary_band(mask, band_width)
    denom = band.mean() + 1e-8
    pix = (torch.abs(fake - real) * band).mean() / denom

    grad_x_fake = fake[:, :, :, 1:] - fake[:, :, :, :-1]
    grad_x_real = real[:, :, :, 1:] - real[:, :, :, :-1]
    band_x = band[:, :, :, 1:].expand_as(grad_x_fake)
    grad_x = (torch.abs(grad_x_fake - grad_x_real) * band_x).mean() / (band_x.mean() + 1e-8)

    grad_y_fake = fake[:, :, 1:, :] - fake[:, :, :-1, :]
    grad_y_real = real[:, :, 1:, :] - real[:, :, :-1, :]
    band_y = band[:, :, 1:, :].expand_as(grad_y_fake)
    grad_y = (torch.abs(grad_y_fake - grad_y_real) * band_y).mean() / (band_y.mean() + 1e-8)

    return weight * (pix + 0.5 * (grad_x + grad_y))
