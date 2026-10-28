"""图片去水印核心算法（纯 Python 实现，只依赖 numpy + pillow）。

三条思路，覆盖大多数常见场景：

1. ``inpaint`` —— 区域修复。
   先把水印做成遮罩，再用「归一化卷积」把周围的像素一层层扩散进遮罩区域。
   装了 opencv 时会自动改用 ``cv2.inpaint``（Telea），速度更快，没装也能跑。
2. ``remove_by_watermark`` —— 水印反演。
   已知水印图案（比如自己加水印时用的那张 logo）时，按叠加公式
   ``I = (1 - a) * O + a * W`` 反解出原图 ``O``，效果最干净。
3. ``estimate_background`` —— 多图中值估计。
   同一张图的多份拷贝、水印位置各不相同（很多平台是随机位置）时，
   逐像素取中值即可把水印滤掉，得到干净背景。

所有函数输入输出统一为 ``uint8`` 的 numpy 数组，形状 ``(H, W, 3)``。
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

__all__ = [
    "imread",
    "imwrite",
    "resize_like",
    "inpaint",
    "remove_by_watermark",
    "estimate_background",
    "offset_map",
]


# ---------------------------------------------------------------------------
# 图像读写
# ---------------------------------------------------------------------------
def imread(path: str, keep_alpha: bool = False) -> np.ndarray:
    """读取图片为 uint8 数组。默认转成 RGB（透明部分按白底合成）。"""
    img = Image.open(path)
    if keep_alpha:
        return np.asarray(img.convert("RGBA"))

    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        rgba = Image.alpha_composite(bg, rgba)
        img = rgba.convert("RGB")
    else:
        img = img.convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def imwrite(path: str, arr: np.ndarray) -> None:
    """把数组存成图片，自动裁剪到 0-255。"""
    data = np.clip(np.asarray(arr), 0, 255).astype(np.uint8)
    Image.fromarray(data).save(path)


def resize_like(arr: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """把 ``arr`` 缩放到与 ``reference`` 相同的宽高。"""
    h, w = reference.shape[:2]
    if arr.shape[:2] == (h, w):
        return arr
    channels = arr.shape[2] if arr.ndim == 3 else 1
    mode = "RGBA" if channels == 4 else ("L" if channels == 1 else "RGB")
    img = Image.fromarray(np.asarray(arr, dtype=np.uint8), mode=mode)
    img = img.resize((w, h), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


# ---------------------------------------------------------------------------
# 纯 numpy 的快速模糊（3 次盒式滤波近似高斯）
# ---------------------------------------------------------------------------
def _box_blur(a: np.ndarray, radius: int, axis: int) -> np.ndarray:
    """沿单个轴做盒式模糊，用前缀和实现 O(N)。要求 a 为 float64。"""
    if radius <= 0:
        return a
    n = a.shape[axis]
    pad = [(0, 0)] * a.ndim
    pad[axis] = (radius, radius)
    padded = np.pad(a, pad, mode="edge")
    cum = np.cumsum(padded, axis=axis, dtype=np.float64)

    zero_shape = list(cum.shape)
    zero_shape[axis] = 1
    cum = np.concatenate([np.zeros(zero_shape, dtype=np.float64), cum], axis=axis)

    idx = np.arange(n)
    hi = np.take(cum, idx + 2 * radius + 1, axis=axis)
    lo = np.take(cum, idx, axis=axis)
    return (hi - lo) / (2 * radius + 1)


def _blur2d(a: np.ndarray, radius: int) -> np.ndarray:
    out = a
    for _ in range(3):
        out = _box_blur(out, radius, axis=0)
        out = _box_blur(out, radius, axis=1)
    return out


def _blur_stack(a: np.ndarray, radius: int) -> np.ndarray:
    """对 (H, W, C) 逐通道模糊，通道之间不串扰。"""
    out = np.empty_like(a)
    for ch in range(a.shape[2]):
        out[:, :, ch] = _blur2d(np.ascontiguousarray(a[:, :, ch]), radius)
    return out


def _mask_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())


def auto_radius(mask: np.ndarray) -> int:
    """根据遮罩大小估算一个合适的填充半径。"""
    y0, y1, x0, x1 = _mask_bbox(mask)
    longest = max(y1 - y0 + 1, x1 - x0 + 1)
    return int(min(64, max(4, longest // 2)))


# ---------------------------------------------------------------------------
# 1. 区域修复
# ---------------------------------------------------------------------------
def _inpaint_numpy(img: np.ndarray, mask: np.ndarray, radius: int) -> np.ndarray:
    """归一化卷积修复：未知区域 = 周围已知像素的加权平均，逐层向内推进。"""
    work = img.astype(np.float64, copy=True)
    work[mask] = 0.0
    known = ~mask

    # 半径从大到小：先粗填充拿到大结构，再逐级细化
    scales: List[int] = []
    r = float(radius)
    while r >= 1.0:
        scales.append(max(1, int(round(r))))
        r /= 2.0
    scales.append(1)

    for r in scales:
        for _ in range(3):
            num = _blur_stack(work * known[:, :, None], r)
            den = _blur_stack(known.astype(np.float64)[:, :, None], r)[:, :, 0]
            filled = num / np.maximum(den[:, :, None], 1e-8)

            unknown = ~known
            if not unknown.any():
                break
            work[unknown] = filled[unknown]
            # 已经拿到足够邻域支撑的像素，下一轮当作已知继续向内推
            known = known | (den > 0.05)
        if known.all():
            break

    out = img.astype(np.float64, copy=True)
    out[mask] = work[mask]
    return np.clip(out, 0, 255)


def _inpaint_opencv(img: np.ndarray, mask: np.ndarray, radius: int) -> np.ndarray:
    import cv2  # type: ignore

    m = (mask.astype(np.uint8)) * 255
    return cv2.inpaint(img.astype(np.uint8), m, float(radius), cv2.INPAINT_TELEA).astype(np.float64)


def inpaint(
    img: np.ndarray,
    mask: np.ndarray,
    radius: Optional[int] = None,
    prefer_opencv: bool = True,
) -> np.ndarray:
    """把 ``mask``（True / 255 表示要修掉的区域）里的像素用周围内容填掉。

    ``radius`` 不给时按遮罩大小自动估计。装了 opencv 会优先用它，失败自动回退。
    """
    src = np.asarray(img, dtype=np.uint8)
    m = np.asarray(mask) > 0
    if m.ndim == 3:
        m = m.any(axis=2)
    if not m.any():
        return src.copy()
    if radius is None:
        radius = auto_radius(m)

    float_img = src.astype(np.float64)
    if prefer_opencv:
        try:
            return np.clip(_inpaint_opencv(float_img, m, radius), 0, 255).astype(np.uint8)
        except Exception:
            pass
    return np.clip(_inpaint_numpy(float_img, m, radius), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 2. 已知水印图案 -> 反演
# ---------------------------------------------------------------------------
def remove_by_watermark(
    img: np.ndarray,
    watermark: np.ndarray,
    alpha: Optional[float] = None,
    strength: float = 1.0,
) -> np.ndarray:
    """按 ``I = (1-a)*O + a*W`` 反解原图 ``O``。

    ``watermark`` 可以是 RGB（用亮度当透明度，``alpha`` 默认 0.3）
    或带透明通道的 PNG（用 alpha 通道，``alpha`` 默认 1.0）。
    结果过冲就把 ``alpha`` 或 ``strength`` 调小。
    """
    src = img.astype(np.float64)
    wm = resize_like(np.asarray(watermark, dtype=np.uint8), src)

    if wm.shape[2] == 4:
        base_alpha = 1.0 if alpha is None else float(alpha)
        a = wm[:, :, 3:4].astype(np.float64) / 255.0
        color = wm[:, :, :3].astype(np.float64)
    else:
        base_alpha = 0.3 if alpha is None else float(alpha)
        lum = wm.astype(np.float64).mean(axis=2, keepdims=True) / 255.0
        a = lum
        color = wm.astype(np.float64)

    a = np.clip(a * base_alpha * float(strength), 0.0, 0.98)
    out = (src - a * color) / (1.0 - a)
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 3. 多图中值估计
# ---------------------------------------------------------------------------
def estimate_background(images: Sequence[np.ndarray]) -> np.ndarray:
    """多张同款图（水印位置不一致）逐像素取中值，得到干净背景。"""
    if len(images) < 2:
        raise ValueError("至少需要 2 张图片才能估计背景")
    stack = np.stack([im.astype(np.float64) for im in images], axis=0)
    return np.clip(np.median(stack, axis=0), 0, 255).astype(np.uint8)


def offset_map(img: np.ndarray, background: np.ndarray) -> np.ndarray:
    """单张图相对背景的残差（含正负），用于可视化水印残留。"""
    return img.astype(np.float64) - background.astype(np.float64)
