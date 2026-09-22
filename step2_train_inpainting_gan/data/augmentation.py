"""Stochastic per-image augmentation for GAN training."""
import random

import cv2
import numpy as np


class RandomBright:
    def __init__(self, bright=1, random_range=0.1, aug_prob=0.5):
        self.bright = bright
        self.range = random_range
        self.aug_prob = aug_prob

    def __call__(self, img):
        if np.random.uniform() < self.aug_prob:
            delta = 2 * (np.random.uniform(0, self.range) - self.range / 2)
            img = cv2.convertScaleAbs(img, alpha=self.bright + delta, beta=0)
        return img


class RandomContrast:
    def __init__(self, contrast=1, random_range=0.1, aug_prob=0.5):
        self.contrast = contrast
        self.range = random_range
        self.aug_prob = aug_prob

    def __call__(self, img):
        if np.random.uniform() < self.aug_prob:
            delta = 2 * (np.random.uniform(0, self.range) - self.range / 2)
            img = cv2.convertScaleAbs(img, alpha=self.contrast + delta, beta=0)
        return img


class RandomClahe:
    def __init__(self, aug_prob=0.5, base_clip=3.0, clip_jitter=1.5, tile_size=(8, 8)):
        self.aug_prob = aug_prob
        self.base_clip = base_clip
        self.clip_jitter = clip_jitter
        self.tile_size = tile_size

    def __call__(self, img):
        if np.random.uniform() >= self.aug_prob:
            return img
        clip = self.base_clip + self.clip_jitter * (np.random.uniform() - 0.5) * 2
        clahe = cv2.createCLAHE(clipLimit=float(max(0.1, clip)), tileGridSize=self.tile_size)
        if img.ndim < 3:
            return clahe.apply(img)
        out = img.copy()
        for c in range(out.shape[2]):
            out[..., c] = clahe.apply(out[..., c])
        return out


class RandomInterpolationResample:
    def __init__(self, scale=0.15, p=0.5):
        self.scale = float(scale)
        self.p = float(p)
        self.interps = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC,
                        cv2.INTER_AREA, cv2.INTER_LANCZOS4]

    def __call__(self, img):
        if np.random.rand() > self.p:
            return img
        h, w = img.shape[:2]
        mode = random.choice(["down_up", "up_down"])
        interp1, interp2 = random.choice(self.interps), random.choice(self.interps)
        factor = 1.0 - self.scale if mode == "down_up" else 1.0 + self.scale
        mid_w, mid_h = max(1, int(round(w * factor))), max(1, int(round(h * factor)))
        tmp = cv2.resize(img, (mid_w, mid_h), interpolation=interp1)
        return cv2.resize(tmp, (w, h), interpolation=interp2)


class CXRAugmentation:
    def __init__(self, gan_cfg):
        self.transforms = [
            RandomBright(aug_prob=0.5),
            RandomContrast(aug_prob=0.5),
        ]
        if gan_cfg.use_clahe_aug:
            self.transforms.append(RandomClahe(
                aug_prob=gan_cfg.clahe_aug_prob,
                base_clip=gan_cfg.clahe_base_clip,
                clip_jitter=gan_cfg.clahe_clip_jitter,
                tile_size=tuple(gan_cfg.clahe_tile_size),
            ))
        self.resample = RandomInterpolationResample(scale=0.15, p=0.5)

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return self.resample(img)
