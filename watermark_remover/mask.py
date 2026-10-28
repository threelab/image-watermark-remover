"""遮罩生成：告诉程序「哪一块像素是水印」。

三种来源，可以组合使用：
  * 矩形框 ``--rect x,y,w,h``（可重复传多次）
  * 手画的遮罩图 ``--mask mask.png``（白色/不透明区域 = 要修掉的区域）
  * 颜色检测 ``--color 255,255,255 --tolerance 40``（适合纯色 logo / 白色文字）
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["build_mask", "dilate"]


def mask_from_rects(shape: Tuple[int, int], rects: Sequence[Tuple[int, int, int, int]]) -> np.ndarray:
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    for x, y, rw, rh in rects:
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(w, x + rw), min(h, y + rh)
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = True
    return mask


def mask_from_image(mask_img: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    arr = np.asarray(mask_img)
    if arr.ndim == 2:
        return arr > 10
    if arr.shape[2] == 4:
        return arr[:, :, 3] > 10
    return arr.astype(np.int16).mean(axis=2) > 10


def mask_from_color(
    img: np.ndarray,
    color: Sequence[int],
    tolerance: float,
) -> np.ndarray:
    target = np.asarray(color, dtype=np.float64).reshape(1, 1, 3)
    diff = np.abs(img.astype(np.float64) - target)
    return diff.max(axis=2) <= float(tolerance)


def dilate(mask: np.ndarray, iterations: int) -> np.ndarray:
    """把遮罩往外扩几圈，盖住水印的抗锯齿边缘。"""
    out = np.asarray(mask, dtype=bool).copy()
    for _ in range(max(0, int(iterations))):
        padded = np.pad(out, 1, mode="constant", constant_values=False)
        for dy in (0, 1, 2):
            for dx in (0, 1, 2):
                if dy == 1 and dx == 1:
                    continue
                out |= padded[dy:dy + out.shape[0], dx:dx + out.shape[1]]
    return out


def build_mask(
    img: np.ndarray,
    rects: Optional[Sequence[Tuple[int, int, int, int]]] = None,
    mask_image: Optional[np.ndarray] = None,
    color: Optional[Sequence[int]] = None,
    tolerance: float = 40.0,
    dilate_iters: int = 0,
) -> np.ndarray:
    """把各种来源合成一张 bool 遮罩（True = 待修复）。"""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=bool)

    if mask_image is not None:
        mask |= mask_from_image(mask_image, (h, w))
    if rects:
        mask |= mask_from_rects((h, w), rects)
    if color is not None:
        mask |= mask_from_color(img, list(color), tolerance)

    if dilate_iters:
        mask = dilate(mask, dilate_iters)
    return mask


def mask_summary(mask: np.ndarray) -> str:
    total = mask.size
    hit = int(mask.sum())
    return "%d px (%.2f%%)" % (hit, 100.0 * hit / max(1, total))
