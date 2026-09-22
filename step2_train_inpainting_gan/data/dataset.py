"""Patch-level dataset for inpainting-GAN training.

Each item is one CXR; it yields up to patches_per_image 128x128 patches
(fracture-centred positives + rib-point negatives) plus a per-patch centered
square inpainting mask with size jitter. The collate function balances and
caps the patch pool per batch.
"""
import json
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .augmentation import CXRAugmentation
from .preprocessing import cxr_preprocess

IMG_EXTS = (".png", ".jpg", ".jpeg")


def build_training_data_list(images_root, use_nofx_training: bool):
    """List of (img_path, fxlabel_json|None, maskpoint_json|None, label).
    fx images come from <images_root>/fx, nofx images from <images_root>/nofx."""
    images_root = str(images_root)
    fx_dir = os.path.join(images_root, "fx")
    fxlabel_dir = os.path.join(images_root, "fxlabel")
    nofx_dir = os.path.join(images_root, "nofx")
    maskpoint_dir = os.path.join(images_root, "maskpoint")

    fx_names = sorted(f for f in os.listdir(fx_dir) if f.lower().endswith(IMG_EXTS))

    data = []
    for fname in fx_names:
        stem = os.path.splitext(fname)[0]
        lbl_p = os.path.join(fxlabel_dir, stem + ".json")
        pts_p = os.path.join(maskpoint_dir, stem + ".json")
        data.append((
            os.path.join(fx_dir, fname),
            lbl_p if os.path.exists(lbl_p) else None,
            pts_p if os.path.exists(pts_p) else None,
            1,
        ))
    if not use_nofx_training:
        return data
    for fname in sorted(f for f in os.listdir(nofx_dir) if f.lower().endswith(IMG_EXTS)):
        stem = os.path.splitext(fname)[0]
        pts_p = os.path.join(maskpoint_dir, stem + ".json")
        data.append((
            os.path.join(nofx_dir, fname),
            None,
            pts_p if os.path.exists(pts_p) else None,
            0,
        ))
    return data


class BalancedBatchSampler(Sampler):
    """Half-fx / half-nofx image batches (all-fx when nofx is disabled)."""

    def __init__(self, dataset, batch_size, use_nofx_training: bool):
        self.dataset = dataset
        self.batch_size = batch_size
        self.use_nofx_training = use_nofx_training
        self.fx_indices = [i for i, x in enumerate(dataset.data_list) if x[-1] == 1]
        self.normal_indices = [i for i, x in enumerate(dataset.data_list) if x[-1] == 0]
        self.half_b = batch_size // 2
        if use_nofx_training and self.normal_indices:
            self.num_batches = len(self.fx_indices) // self.half_b
        else:
            self.num_batches = len(self.fx_indices) // self.batch_size

    def __iter__(self):
        fx = self.fx_indices.copy()
        normal = self.normal_indices.copy()
        random.shuffle(fx)
        random.shuffle(normal)
        if not self.use_nofx_training or not normal:
            for i in range(0, len(fx) - self.batch_size + 1, self.batch_size):
                yield fx[i:i + self.batch_size]
            return
        for _ in range(self.num_batches):
            batch = []
            for _ in range(self.half_b):
                batch.append(fx.pop())
            for _ in range(self.half_b):
                if len(normal) == 0:
                    normal = self.normal_indices.copy()
                    random.shuffle(normal)
                batch.append(normal.pop())
            random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches


# --- crop / geometry helpers ----------------------------------------------------

def crop_fixed_center(img, cx, cy, patch_size):
    img_h, img_w = img.shape[:2]
    half = patch_size // 2
    x1, y1 = int(cx - half), int(cy - half)
    x2, y2 = x1 + patch_size, y1 + patch_size
    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - img_w)
    pad_bottom = max(0, y2 - img_h)
    cropped = img[max(0, y1):min(img_h, y2), max(0, x1):min(img_w, x2)]
    if pad_left or pad_top or pad_right or pad_bottom:
        cropped = cv2.copyMakeBorder(cropped, pad_top, pad_bottom, pad_left, pad_right,
                                     cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return cropped


def _patch_fully_valid(valid_mask, cx, cy, patch_size):
    patch = crop_fixed_center(valid_mask, cx, cy, patch_size)
    return bool(np.all(patch > 0))


def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / float(areaA + areaB - inter + 1e-6)


def _min_dist_to_fx_bboxes(cx, cy, fx_bboxes):
    if not fx_bboxes:
        return float("inf")
    return min(
        ((cx - (b[0] + b[2]) / 2.0) ** 2 + (cy - (b[1] + b[3]) / 2.0) ** 2) ** 0.5
        for b in fx_bboxes
    )


def _patch_overlaps_fx(cx, cy, patch_size, fx_bboxes, iou_thresh=0.15):
    half = patch_size / 2.0
    boxA = [cx - half, cy - half, cx + half, cy + half]
    for bbox in fx_bboxes:
        if bb_intersection_over_union(boxA, bbox) > iou_thresh:
            return True
        fc_x = (bbox[0] + bbox[2]) / 2.0
        fc_y = (bbox[1] + bbox[3]) / 2.0
        if ((cx - fc_x) ** 2 + (cy - fc_y) ** 2) ** 0.5 < half:
            return True
    return False


def _weighted_sample_rib_point(rib_points, fx_bboxes, temperature=0.005):
    """Prefer rib points far from fracture boxes (soft-max over distance)."""
    if not fx_bboxes:
        return random.choice(rib_points)
    dists = np.array([_min_dist_to_fx_bboxes(p[0], p[1], fx_bboxes)
                      for p in rib_points], dtype=np.float64)
    dists -= dists.max()
    weights = np.exp(temperature * dists)
    weights /= weights.sum()
    return rib_points[np.random.choice(len(rib_points), p=weights)]


def _rotate_points(points, M):
    if not points:
        return []
    arr = np.asarray(points, dtype=np.float32)
    ones = np.ones((arr.shape[0], 1), dtype=np.float32)
    rot = np.concatenate([arr, ones], axis=1) @ M.T
    return [[float(x), float(y)] for x, y in rot]


def _rotate_bboxes(fx_bboxes, M):
    rotated = []
    for xmin, ymin, xmax, ymax in fx_bboxes:
        corners = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
        rc = np.asarray(_rotate_points(corners, M), dtype=np.float32)
        rxmin, rymin = float(rc[:, 0].min()), float(rc[:, 1].min())
        rxmax, rymax = float(rc[:, 0].max()), float(rc[:, 1].max())
        if rxmin < rxmax and rymin < rymax:
            rotated.append([rxmin, rymin, rxmax, rymax])
    return rotated


def rotate_sample(img, rib_points, fx_bboxes, angle_deg):
    """Rotate image and annotations together; also returns a validity mask so
    patches are only sampled from non-padded pixels."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    rotated_img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    valid = np.ones((h, w), dtype=np.uint8) * 255
    rotated_valid = cv2.warpAffine(valid, M, (w, h), flags=cv2.INTER_NEAREST,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return rotated_img, _rotate_points(rib_points, M), _rotate_bboxes(fx_bboxes, M), rotated_valid


# --- inpainting mask helpers ------------------------------------------------------

_MIN_BORDER = 2  # px of real context kept on every side of the hole


def _sample_mask_side(gan_cfg):
    if gan_cfg.mask_size_jitter <= 0:
        return gan_cfg.mask_size
    side = gan_cfg.mask_size + random.randint(-gan_cfg.mask_size_jitter, gan_cfg.mask_size_jitter)
    return int(np.clip(side, _MIN_BORDER * 2, gan_cfg.patch_size - _MIN_BORDER * 2))


def _centered_mask_np(gan_cfg, side):
    m = np.zeros((1, gan_cfg.patch_size, gan_cfg.patch_size), dtype=np.float32)
    pad = (gan_cfg.patch_size - side) // 2
    m[:, pad:pad + side, pad:pad + side] = 1.0
    return m


def make_mask(batch, device, gan_cfg, jitter=True):
    """Centered square masks; jitter=False keeps the fixed baseline size
    (used for the stable visualization panel)."""
    if not jitter or gan_cfg.mask_size_jitter <= 0:
        mask = torch.zeros(batch, 1, gan_cfg.patch_size, gan_cfg.patch_size, device=device)
        pad = (gan_cfg.patch_size - gan_cfg.mask_size) // 2
        mask[:, :, pad:pad + gan_cfg.mask_size, pad:pad + gan_cfg.mask_size] = 1.0
        return mask
    masks = torch.zeros(batch, 1, gan_cfg.patch_size, gan_cfg.patch_size, device=device)
    for b in range(batch):
        side = _sample_mask_side(gan_cfg)
        pad = (gan_cfg.patch_size - side) // 2
        masks[b, :, pad:pad + side, pad:pad + side] = 1.0
    return masks


# --- dataset --------------------------------------------------------------------

class InpaintingPatchDataset(Dataset):
    def __init__(self, data_list, gan_cfg, is_train=True):
        self.data_list = data_list
        self.cfg = gan_cfg
        self.is_train = is_train
        self.aug = CXRAugmentation(gan_cfg) if is_train else None

    def __len__(self):
        return len(self.data_list)

    def parse_annotation(self, lbl_path, pt_path):
        rib_points = []
        fx_bboxes = []
        if pt_path and os.path.exists(pt_path):
            with open(pt_path, "r") as f:
                rib_points = json.load(f).get("points_list", [])
        if lbl_path and os.path.exists(lbl_path):
            with open(lbl_path, "r") as f:
                for shape in json.load(f).get("shapes", []):
                    if shape.get("label", "") != "rib fx":
                        continue
                    pts = shape.get("points", [])
                    if len(pts) >= 2:
                        p1, p2 = pts[0], pts[1]
                        xmin, xmax = min(p1[0], p2[0]), max(p1[0], p2[0])
                        ymin, ymax = min(p1[1], p2[1]), max(p1[1], p2[1])
                        if xmin < xmax and ymin < ymax:
                            fx_bboxes.append([xmin, ymin, xmax, ymax])
        return rib_points, fx_bboxes

    def resize_and_pad(self, img, rib_points, fx_bboxes, target_size):
        h, w = img.shape[:2]
        scale = target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        img_resized = cv2.resize(img, (new_w, new_h))
        pad_y = (target_size - new_h) // 2
        pad_x = (target_size - new_w) // 2
        img_padded = cv2.copyMakeBorder(
            img_resized, pad_y, target_size - new_h - pad_y,
            pad_x, target_size - new_w - pad_x,
            cv2.BORDER_CONSTANT, value=(0, 0, 0),
        )
        new_points = [[int(x * scale) + pad_x, int(y * scale) + pad_y] for (x, y) in rib_points]
        new_bboxes = [[int(x1 * scale) + pad_x, int(y1 * scale) + pad_y,
                       int(x2 * scale) + pad_x, int(y2 * scale) + pad_y]
                      for (x1, y1, x2, y2) in fx_bboxes]
        return img_padded, new_points, new_bboxes

    def __getitem__(self, idx):
        c = self.cfg
        img_path, lbl_path, pt_path, label = self.data_list[idx]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            img = np.zeros((c.image_size, c.image_size, 3), dtype=np.uint8)

        rib_points, fx_bboxes = self.parse_annotation(lbl_path, pt_path)
        img = cxr_preprocess(img)
        img, rib_points, fx_bboxes = self.resize_and_pad(img, rib_points, fx_bboxes, c.image_size)
        valid_mask = np.ones(img.shape[:2], dtype=np.uint8) * 255

        if self.is_train and c.use_rot_aug and random.random() < c.rot_aug_prob:
            angle = random.uniform(-c.rot_max_deg, c.rot_max_deg)
            img, rib_points, fx_bboxes, valid_mask = rotate_sample(img, rib_points, fx_bboxes, angle)

        if len(rib_points) == 0:
            rib_points = [(random.randint(100, c.image_size - 100),
                           random.randint(100, c.image_size - 100)) for _ in range(20)]

        if self.aug is not None:
            img = self.aug(img)

        patches = []
        patch_labels = []
        target_fx = c.patches_per_image // 2 if label == 1 else 0
        target_total = c.patches_per_image

        if fx_bboxes and target_fx > 0:
            attempts = 0
            while len(patches) < target_fx and attempts < 200:
                attempts += 1
                bbox = random.choice(fx_bboxes)
                cx = bbox[0] + (bbox[2] - bbox[0]) / 2
                cy = bbox[1] + (bbox[3] - bbox[1]) / 2
                c_xj = np.clip(cx + random.randint(-c.patch_jitter, c.patch_jitter),
                               c.patch_size // 2, img.shape[1] - c.patch_size // 2)
                c_yj = np.clip(cy + random.randint(-c.patch_jitter, c.patch_jitter),
                               c.patch_size // 2, img.shape[0] - c.patch_size // 2)
                if not _patch_fully_valid(valid_mask, c_xj, c_yj, c.patch_size):
                    continue
                patches.append(crop_fixed_center(img, c_xj, c_yj, c.patch_size))
                patch_labels.append(1)

        attempts = 0
        while len(patches) < target_total and attempts < 200:
            attempts += 1
            cx, cy = _weighted_sample_rib_point(rib_points, fx_bboxes)
            if _patch_overlaps_fx(cx, cy, c.patch_size, fx_bboxes, iou_thresh=0.15):
                continue
            if not _patch_fully_valid(valid_mask, cx, cy, c.patch_size):
                continue
            patches.append(crop_fixed_center(img, cx, cy, c.patch_size))
            patch_labels.append(0)

        while len(patches) < target_total:
            patches.append(np.zeros((c.patch_size, c.patch_size, 3), dtype=np.uint8))
            patch_labels.append(0)

        patch_masks = [_centered_mask_np(c, _sample_mask_side(c)) for _ in patches]

        combined = list(zip(patches, patch_masks, patch_labels))
        random.shuffle(combined)
        patches, patch_masks, patch_labels = zip(*combined)

        patches_tensor = torch.stack(
            [torch.from_numpy(p).permute(2, 0, 1).float() / 255.0 for p in patches])
        masks_tensor = torch.from_numpy(np.stack(patch_masks, axis=0))
        patch_labels_tensor = torch.tensor(patch_labels, dtype=torch.float32)
        return patches_tensor, masks_tensor, patch_labels_tensor


class CollatePatchPool:
    """Flatten per-image patch bags, rebalance fx/nofx 1:1, cap the pool to
    max_patches_per_batch."""

    def __init__(self, gan_cfg):
        self.max_patches = gan_cfg.max_patches_per_batch
        self.use_nofx = gan_cfg.use_nofx_training

    def __call__(self, batch):
        patches, masks, patch_labels = zip(*batch)
        all_patches = torch.cat(patches, dim=0)
        all_masks = torch.cat(masks, dim=0)
        all_labels = torch.cat(patch_labels, dim=0)

        fx_idx = (all_labels == 1).nonzero(as_tuple=True)[0]
        norm_idx = (all_labels == 0).nonzero(as_tuple=True)[0]
        n_min = min(len(fx_idx), len(norm_idx))
        if self.use_nofx and n_min > 0:
            fx_sel = fx_idx[torch.randperm(len(fx_idx))[:n_min]]
            norm_sel = norm_idx[torch.randperm(len(norm_idx))[:n_min]]
            sel = torch.cat([fx_sel, norm_sel])
            sel = sel[torch.randperm(len(sel))]
            all_patches, all_masks, all_labels = all_patches[sel], all_masks[sel], all_labels[sel]

        if len(all_patches) > self.max_patches:
            if self.use_nofx:
                cap_half = self.max_patches // 2
                fx_idx2 = (all_labels == 1).nonzero(as_tuple=True)[0]
                norm_idx2 = (all_labels == 0).nonzero(as_tuple=True)[0]
                fx_sel2 = fx_idx2[torch.randperm(len(fx_idx2))[:cap_half]]
                norm_sel2 = norm_idx2[torch.randperm(len(norm_idx2))[:cap_half]]
                sel2 = torch.cat([fx_sel2, norm_sel2])
            else:
                sel2 = torch.randperm(len(all_patches))[:self.max_patches]
            sel2 = sel2[torch.randperm(len(sel2))]
            all_patches, all_masks, all_labels = all_patches[sel2], all_masks[sel2], all_labels[sel2]

        return all_patches, all_masks, all_labels
