"""Stochastic intensity augmentation for training patches."""
from __future__ import annotations

import random

import cv2
import numpy as np


class PatchAugmentation:
    def __init__(self, train_cfg):
        self.cfg = train_cfg

    def _random_bright(self, img, prob=0.5):
        rng = float(self.cfg.bright_aug_range)
        if np.random.uniform() < prob:
            delta = 2 * (np.random.uniform(0, rng) - rng / 2)
            img = cv2.convertScaleAbs(img, alpha=1.0, beta=int(delta * 255))
        return img

    def _random_contrast(self, img, prob=0.5):
        rng = float(self.cfg.contrast_aug_range)
        if np.random.uniform() < prob:
            delta = 2 * (np.random.uniform(0, rng) - rng / 2)
            img = cv2.convertScaleAbs(img, alpha=1.0 + delta, beta=0)
        return img

    def _random_clahe(self, img):
        if not self.cfg.use_clahe_aug or np.random.uniform() >= self.cfg.clahe_aug_prob:
            return img
        clip = self.cfg.clahe_base_clip + self.cfg.clahe_clip_jitter * (np.random.uniform() - 0.5) * 2
        clahe = cv2.createCLAHE(clipLimit=float(max(0.1, clip)),
                                tileGridSize=tuple(self.cfg.clahe_tile_size))
        out = img.copy()
        for c in range(out.shape[2]):
            out[..., c] = clahe.apply(out[..., c])
        return out

    def _pick_two_interps(self):
        interps = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC,
                   cv2.INTER_AREA, cv2.INTER_LANCZOS4]
        a = random.choice(interps)
        if not bool(self.cfg.resample_force_distinct):
            return a, random.choice(interps)
        pool = [ip for ip in interps if ip != a]
        return a, random.choice(pool) if pool else a

    def _random_resample(self, img):
        scale = float(self.cfg.resample_aug_scale)
        if np.random.rand() > float(self.cfg.resample_aug_prob):
            return img
        h, w = img.shape[:2]
        mode = random.choice(["down_up", "up_down"])
        a, b = self._pick_two_interps()
        factor = 1.0 - scale if mode == "down_up" else 1.0 + scale
        mid_w, mid_h = max(1, int(round(w * factor))), max(1, int(round(h * factor)))
        tmp = cv2.resize(img, (mid_w, mid_h), interpolation=a)
        return cv2.resize(tmp, (w, h), interpolation=b)

    def __call__(self, img):
        if not self.cfg.use_intensity_aug:
            return img
        img = self._random_bright(img)
        img = self._random_contrast(img)
        img = self._random_clahe(img)
        img = self._random_resample(img)
        return img
