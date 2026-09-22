"""Geometric transforms (resize/pad/rotate with annotations) and bbox/patch helpers."""
from __future__ import annotations

import math
from typing import Tuple

import cv2
import numpy as np
import torch


def _bbox_to_corners(bbox):
    x1, y1, x2, y2 = bbox
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def _corners_to_aabb(corners):
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    return (min(xs), min(ys), max(xs), max(ys))


def _resize_points(points, scale, px, py):
    return [(x * scale + px, y * scale + py) for x, y in points]


def _resize_bboxes(bboxes, scale, px, py):
    return [(xmin * scale + px, ymin * scale + py, xmax * scale + px, ymax * scale + py)
            for (xmin, ymin, xmax, ymax) in bboxes]


def _rotate_points(points, matrix):
    if not points:
        return []
    arr = np.asarray(points, dtype=np.float32)
    ones = np.ones((arr.shape[0], 1), dtype=np.float32)
    rot = np.concatenate([arr, ones], axis=1) @ matrix.T
    return [(float(x), float(y)) for x, y in rot]


def _rotate_bboxes(bboxes, matrix):
    return [_corners_to_aabb(_rotate_points(_bbox_to_corners(bb), matrix)) for bb in bboxes]


def resize_pad_groups(img: np.ndarray, point_groups, size: int, bbox_groups=None):
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    py = (size - nh) // 2
    px = (size - nw) // 2
    padded = cv2.copyMakeBorder(resized, py, size - nh - py, px, size - nw - px,
                                cv2.BORDER_CONSTANT, value=(0, 0, 0))
    points_out = [_resize_points(pts, scale, px, py) for pts in point_groups]
    if bbox_groups is None:
        return padded, points_out
    boxes_out = [_resize_bboxes(bbs, scale, px, py) for bbs in bbox_groups]
    return padded, points_out, boxes_out


def rotate_sample_groups(img: np.ndarray, point_groups, angle_deg: float,
                         expand_crop: bool = True, bbox_groups=None):
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    if not expand_crop:
        rotated = cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        points_out = [_rotate_points(pts, matrix) for pts in point_groups]
        if bbox_groups is None:
            return rotated, points_out
        return rotated, points_out, [_rotate_bboxes(bbs, matrix) for bbs in bbox_groups]
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    if new_w <= 0 or new_h <= 0:
        if bbox_groups is None:
            return img, point_groups
        return img, point_groups, bbox_groups
    matrix[0, 2] += (new_w / 2.0) - center[0]
    matrix[1, 2] += (new_h / 2.0) - center[1]
    rotated = cv2.warpAffine(img, matrix, (new_w, new_h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    crop_x = max(0, (new_w - w) // 2)
    crop_y = max(0, (new_h - h) // 2)
    cropped = rotated[crop_y:crop_y + h, crop_x:crop_x + w]
    crop_matrix = matrix.copy()
    crop_matrix[0, 2] -= crop_x
    crop_matrix[1, 2] -= crop_y
    points_out = [_rotate_points(pts, crop_matrix) for pts in point_groups]
    if bbox_groups is None:
        return cropped, points_out
    return cropped, points_out, [_rotate_bboxes(bbs, crop_matrix) for bbs in bbox_groups]


# --- patch sampling helpers --------------------------------------------------------

def clip_to_image(x: float, y: float, image_size: int) -> Tuple[int, int]:
    return (
        int(max(0, min(image_size - 1, round(x)))),
        int(max(0, min(image_size - 1, round(y)))),
    )


def bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return 0.5 * (x1 + x2), 0.5 * (y1 + y2)


def bbox_patch_overlap_area(bboxes, cx: float, cy: float, side: int) -> float:
    if not bboxes:
        return 0.0
    half = side / 2.0
    px1, py1, px2, py2 = cx - half, cy - half, cx + half, cy + half
    total = 0.0
    for (bx1, by1, bx2, by2) in bboxes:
        ox1 = max(bx1, px1)
        oy1 = max(by1, py1)
        ox2 = min(bx2, px2)
        oy2 = min(by2, py2)
        if ox2 > ox1 and oy2 > oy1:
            total += (ox2 - ox1) * (oy2 - oy1)
    return total


def overlap_is_positive(overlap_area: float, patch_area: float, threshold: float) -> bool:
    if threshold <= 0.0:
        return overlap_area > 0.0
    return (overlap_area / patch_area) >= threshold


def min_dist_to_bbox(x: float, y: float, bboxes) -> float:
    if not bboxes:
        return float("inf")
    best = float("inf")
    for (bx1, by1, bx2, by2) in bboxes:
        dx = max(bx1 - x, 0.0, x - bx2)
        dy = max(by1 - y, 0.0, y - by2)
        best = min(best, math.hypot(dx, dy))
    return best


def crop_patch(img: np.ndarray, center: Tuple[int, int], patch_size: int) -> np.ndarray:
    """Extract a patch centred at center, replicating missing edge pixels."""
    half = patch_size // 2
    cx, cy = center
    h, w = img.shape[:2]
    pad_top = max(0, half - cy)
    pad_bot = max(0, cy + half - h)
    pad_left = max(0, half - cx)
    pad_right = max(0, cx + half - w)
    if pad_top or pad_bot or pad_left or pad_right:
        img = cv2.copyMakeBorder(img, pad_top, pad_bot, pad_left, pad_right,
                                 cv2.BORDER_REPLICATE)
        cx += pad_left
        cy += pad_top
    return img[cy - half: cy + half, cx - half: cx + half]
