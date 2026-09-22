"""Image loading, annotation parsing and tensor normalization."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Per-site statistics (mean/std in uint8 space). When loaded, every image is
# standardized with the train-set statistics instead of ImageNet normalization.
_SITE_STATS = None


def load_site_stats(path: Path) -> None:
    global _SITE_STATS
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"[err] site stats not found: {path} (run compute_site_statistics.py first)")
    with path.open() as f:
        stats = json.load(f)
    _SITE_STATS = {
        "mean": np.array(stats["mean"], dtype=np.float32),
        "std": np.array(stats["std"], dtype=np.float32),
    }
    print(f"[data] loaded site stats from {path}", flush=True)


def percentile_norm(img: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    img = img.astype(np.float32)
    p_low = np.percentile(img, low)
    p_high = np.percentile(img, high)
    out = (img - p_low) / (p_high - p_low + 1e-6)
    return (np.clip(out, 0.0, 1.0) * 255).astype(np.uint8)


def load_cxr(path: Path) -> np.ndarray:
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(path)
    gray = percentile_norm(gray)
    return np.stack([gray, gray, gray], axis=-1)


def parse_fx_bboxes(label_path: Optional[Path]) -> List[Tuple[float, float, float, float]]:
    if label_path is None or not label_path.exists():
        return []
    with label_path.open("r") as f:
        data = json.load(f)
    boxes: List[Tuple[float, float, float, float]] = []
    for shape in data.get("shapes", []):
        if "rib fx" not in str(shape.get("label", "")).lower():
            continue
        corners = shape.get("points", [])
        if len(corners) < 2:
            continue
        x1, y1 = float(corners[0][0]), float(corners[0][1])
        x2, y2 = float(corners[1][0]), float(corners[1][1])
        xmin, xmax = min(x1, x2), max(x1, x2)
        ymin, ymax = min(y1, y2), max(y1, y2)
        if xmax > xmin and ymax > ymin:
            boxes.append((xmin, ymin, xmax, ymax))
    return boxes


def parse_rib_points(point_path: Optional[Path]) -> List[Tuple[float, float]]:
    if point_path is None or not point_path.exists():
        return []
    with point_path.open("r") as f:
        data = json.load(f)
    return [(float(x), float(y)) for x, y in data.get("points_list", [])]


def _normalization_shape(tensor: torch.Tensor):
    if tensor.ndim == 3:
        return (3, 1, 1)
    if tensor.ndim == 4:
        return (1, 3, 1, 1)
    raise ValueError(f"expected CHW or NCHW tensor, got shape {tuple(tensor.shape)}")


def normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Normalize a [0, 1] tensor for classifier consumption."""
    shape = _normalization_shape(tensor)
    if _SITE_STATS is not None:
        site_mean = tensor.new_tensor(_SITE_STATS["mean"]).view(shape)
        site_std = tensor.new_tensor(_SITE_STATS["std"]).view(shape)
        return (tensor * 255.0 - site_mean) / site_std
    mean = tensor.new_tensor(IMAGENET_MEAN).view(shape)
    std = tensor.new_tensor(IMAGENET_STD).view(shape)
    return (tensor - mean) / std


def denormalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Invert classifier normalization back to image intensity range [0, 1]."""
    shape = _normalization_shape(tensor)
    if _SITE_STATS is not None:
        site_mean = tensor.new_tensor(_SITE_STATS["mean"]).view(shape)
        site_std = tensor.new_tensor(_SITE_STATS["std"]).view(shape)
        return ((tensor * site_std + site_mean) / 255.0).clamp(0.0, 1.0)
    mean = tensor.new_tensor(IMAGENET_MEAN).view(shape)
    std = tensor.new_tensor(IMAGENET_STD).view(shape)
    return (tensor * std + mean).clamp(0.0, 1.0)
