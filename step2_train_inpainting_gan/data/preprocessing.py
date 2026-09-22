"""Deterministic CXR preprocessing applied once at load time."""
import numpy as np


def percentile_norm(img, low=1.0, high=99.0):
    img = img.astype(np.float32)
    p_low = np.percentile(img, low)
    p_high = np.percentile(img, high)
    out = (img - p_low) / (p_high - p_low + 1e-6)
    return (np.clip(out, 0.0, 1.0) * 255).astype(np.uint8)


def cxr_preprocess(img):
    """Percentile-normalize to uint8 and replicate to 3 channels.
    CLAHE is stochastic augmentation, not preprocessing."""
    img_gray = img[:, :, 0] if img.ndim == 3 else img
    img_gray = percentile_norm(img_gray, low=1.0, high=99.0)
    return np.stack([img_gray, img_gray, img_gray], axis=-1)
