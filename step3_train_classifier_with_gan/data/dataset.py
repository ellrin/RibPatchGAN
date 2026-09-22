"""Patch dataset for classifier training/evaluation.

Train (fx image): positives at fx-bbox centers (+jitter); negatives from rib
maskpoints at least fx_exclusion_radius px from every fx bbox.
Train (nofx image): all negatives from rib maskpoints.
Eval: every rib maskpoint becomes a patch; the label is positive when the
patch/bbox overlap fraction >= pos_bbox_overlap_thresh.
"""
from __future__ import annotations

import random
from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .augmentation import PatchAugmentation
from .geometry import (
    bbox_center,
    bbox_patch_overlap_area,
    clip_to_image,
    crop_patch,
    min_dist_to_bbox,
    overlap_is_positive,
    resize_pad_groups,
    rotate_sample_groups,
)
from .items import CXRItem
from .preprocessing import load_cxr, normalize_tensor, parse_fx_bboxes, parse_rib_points


def _to_tensor(patches: List[np.ndarray]) -> torch.Tensor:
    arr = np.stack(patches, axis=0)
    tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 255.0
    return normalize_tensor(tensor)


class PatchDataset(Dataset):
    """One image -> bag of patches with per-patch binary labels."""

    def __init__(self, items: Sequence[CXRItem], cls_cfg, is_train: bool):
        self.items = list(items)
        self.cfg = cls_cfg
        self.image_size = cls_cfg.train.image_size
        self.patch_size = cls_cfg.train.patch_size
        self.is_train = is_train
        self.aug = PatchAugmentation(cls_cfg.train) if is_train else None

    def __len__(self):
        return len(self.items)

    def _train_centers(self, label: int, fx_bboxes, rib_points):
        s = self.cfg.sampling
        centers: List[Tuple[int, int]] = []
        labels: List[float] = []

        if label == 1 and fx_bboxes:
            bbox_order = list(fx_bboxes)
            random.shuffle(bbox_order)
            for i in range(s.pos_per_bag):
                bb = bbox_order[i % len(bbox_order)]
                cx, cy = bbox_center(bb)
                jx = random.uniform(-s.patch_jitter, s.patch_jitter)
                jy = random.uniform(-s.patch_jitter, s.patch_jitter)
                centers.append(clip_to_image(cx + jx, cy + jy, self.image_size))
                labels.append(1.0)

        target_neg = s.neg_per_bag if label == 1 else s.bag_size
        clean: List[Tuple[float, float]] = []
        fallback: List[Tuple[float, float]] = []
        excl = float(s.fx_exclusion_radius)
        for (x, y) in rib_points:
            if label == 1 and fx_bboxes:
                if min_dist_to_bbox(x, y, fx_bboxes) >= excl:
                    clean.append((x, y))
                else:
                    fallback.append((x, y))
            else:
                clean.append((x, y))
        random.shuffle(clean)
        for (x, y) in clean[:target_neg]:
            jx = random.uniform(-s.patch_jitter, s.patch_jitter)
            jy = random.uniform(-s.patch_jitter, s.patch_jitter)
            centers.append(clip_to_image(x + jx, y + jy, self.image_size))
            labels.append(0.0)
        target_count = s.pos_per_bag + target_neg if label == 1 else s.bag_size
        if len(centers) < target_count:
            need = target_count - len(centers)
            random.shuffle(fallback)
            for (x, y) in fallback[:need]:
                centers.append(clip_to_image(x, y, self.image_size))
                labels.append(0.0)

        order = list(range(len(centers)))
        random.shuffle(order)
        return [centers[i] for i in order], [labels[i] for i in order]

    def _eval_centers(self, fx_bboxes, rib_points):
        centers: List[Tuple[int, int]] = []
        labels: List[float] = []
        thr = float(self.cfg.sampling.pos_bbox_overlap_thresh)
        patch_area = float(self.patch_size * self.patch_size)
        for (x, y) in rib_points:
            xy = clip_to_image(x, y, self.image_size)
            overlap = bbox_patch_overlap_area(fx_bboxes, float(xy[0]), float(xy[1]), self.patch_size)
            centers.append(xy)
            labels.append(1.0 if overlap_is_positive(overlap, patch_area, thr) else 0.0)
        return centers, labels

    def __getitem__(self, idx):
        item = self.items[idx]
        img = load_cxr(item.img_path)
        fx_bboxes = parse_fx_bboxes(item.fx_label_path)
        rib_points = parse_rib_points(item.point_path)

        img, point_groups, bbox_groups = resize_pad_groups(
            img, [rib_points], self.image_size, bbox_groups=[fx_bboxes],
        )
        rib_points = point_groups[0]
        fx_bboxes = bbox_groups[0]

        t = self.cfg.train
        if self.is_train and t.use_rot_aug and random.random() < t.rot_aug_prob:
            angle = random.uniform(-t.rot_max_deg, t.rot_max_deg)
            img, point_groups, bbox_groups = rotate_sample_groups(
                img, [rib_points], angle,
                expand_crop=bool(t.use_rotate_scale_aug),
                bbox_groups=[fx_bboxes],
            )
            rib_points = point_groups[0]
            fx_bboxes = bbox_groups[0]

        if self.aug is not None:
            img = self.aug(img)

        if self.is_train:
            centers, patch_labels = self._train_centers(item.label, fx_bboxes, rib_points)
        else:
            centers, patch_labels = self._eval_centers(fx_bboxes, rib_points)

        # Eval image with zero rib points: synthesise one image-center patch
        # so the bag is never empty.
        if not centers:
            centers = [(self.image_size // 2, self.image_size // 2)]
            patch_labels = [1.0 if item.label == 1 else 0.0]

        patches = [crop_patch(img, c, self.patch_size) for c in centers]
        return {
            "patches": _to_tensor(patches),
            "patch_labels": torch.tensor(patch_labels, dtype=torch.float32),
            "image_label": torch.tensor(item.label, dtype=torch.long),
            "n_patches": torch.tensor(len(centers), dtype=torch.long),
            "site": item.site,
            "path": str(item.img_path),
        }


def collate_patch_bags(batch):
    """Variable-size bags -> list-of-tensors collation (eval bag sizes differ)."""
    return {
        "patches": [b["patches"] for b in batch],
        "patch_labels": [b["patch_labels"] for b in batch],
        "image_label": torch.stack([b["image_label"] for b in batch]),
        "n_patches": torch.stack([b["n_patches"] for b in batch]),
        "site": [b["site"] for b in batch],
        "path": [b["path"] for b in batch],
    }


class FxBalancedBatchSampler:
    """Guarantees >= fx_per_batch fx images in every training batch."""

    def __init__(self, items: Sequence[CXRItem], batch_size: int, fx_per_batch: int, seed: int = 0):
        self.batch_size = batch_size
        self.fx_per_batch = min(fx_per_batch, batch_size)
        self.fx_idx = [i for i, it in enumerate(items) if it.label == 1]
        self.nofx_idx = [i for i, it in enumerate(items) if it.label == 0]
        self.n = len(items)
        self.epoch_seed = seed

    def set_epoch(self, epoch: int):
        self.epoch_seed = epoch

    def __iter__(self):
        rng = random.Random(self.epoch_seed)
        fx = self.fx_idx[:]
        nofx = self.nofx_idx[:]
        rng.shuffle(fx)
        rng.shuffle(nofx)
        fx_cur = 0
        nofx_cur = 0
        for _ in range(len(self)):
            batch = []
            for _ in range(self.fx_per_batch):
                if not fx:
                    break
                if fx_cur >= len(fx):
                    rng.shuffle(fx)
                    fx_cur = 0
                batch.append(fx[fx_cur])
                fx_cur += 1
            while len(batch) < self.batch_size:
                if nofx and nofx_cur < len(nofx):
                    batch.append(nofx[nofx_cur])
                    nofx_cur += 1
                elif nofx:
                    rng.shuffle(nofx)
                    nofx_cur = 0
                    batch.append(nofx[nofx_cur])
                    nofx_cur += 1
                else:
                    if fx_cur >= len(fx):
                        rng.shuffle(fx)
                        fx_cur = 0
                    batch.append(fx[fx_cur])
                    fx_cur += 1
            rng.shuffle(batch)
            yield batch

    def __len__(self):
        return max(1, self.n // self.batch_size)
