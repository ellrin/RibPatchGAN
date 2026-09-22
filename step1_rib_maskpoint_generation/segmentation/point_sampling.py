"""Farthest-point sampling of rib-mask interior pixels."""
import cv2
import numpy as np


def _interior_mask(mask: np.ndarray, min_edge_dist: float) -> np.ndarray:
    dist = cv2.distanceTransform(mask, distanceType=cv2.DIST_L2, maskSize=3)
    return dist >= float(min_edge_dist)


def _farthest_point_sampling(coords: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = coords.shape[0]
    k = min(k, n)
    chosen = [int(rng.integers(0, n))]
    dists = np.sum((coords - coords[chosen[0]]) ** 2, axis=1)
    for _ in range(1, k):
        nxt = int(np.argmax(dists))
        chosen.append(nxt)
        d = np.sum((coords - coords[nxt]) ** 2, axis=1)
        dists = np.minimum(dists, d)
    return np.array(chosen, dtype=np.int64)


def sample_points_in_mask(mask: np.ndarray, sample_points=32, min_edge_dist=3.0, seed=42) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8)
    interior = _interior_mask(mask, min_edge_dist)
    ys, xs = np.where(interior)
    if ys.size == 0:
        return np.zeros((0, 2), dtype=np.int32)

    rng = np.random.default_rng(seed)
    idx = rng.choice(ys.size, size=min(ys.size, sample_points * 20), replace=False)
    cand = np.stack([ys[idx], xs[idx]], axis=1).astype(np.float32)

    picked = _farthest_point_sampling(cand, k=sample_points, rng=rng)
    pts = cand[picked].astype(np.int32)
    return pts[:, ::-1]  # (y, x) -> (x, y)


def sample_points_in_mask_batch(masks: np.ndarray, sample_points=32, min_edge_dist=3.0, seed=42) -> list:
    if masks.ndim == 4 and masks.shape[-1] == 1:
        masks = masks[..., 0]
    elif masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    masks = (masks > 0).astype(np.uint8)

    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2 ** 31 - 1, size=masks.shape[0])
    return [
        sample_points_in_mask(masks[i], sample_points, min_edge_dist, seed=int(seeds[i]))
        for i in range(masks.shape[0])
    ]
