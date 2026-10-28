"""image-watermark-remover —— 纯 Python 图片去水印小工具。"""

from .core import (
    estimate_background,
    imread,
    imwrite,
    inpaint,
    offset_map,
    remove_by_watermark,
    resize_like,
)
from .mask import build_mask, dilate, mask_from_color, mask_from_image, mask_from_rects

__version__ = "0.1.0"

__all__ = [
    "imread",
    "imwrite",
    "resize_like",
    "inpaint",
    "remove_by_watermark",
    "estimate_background",
    "offset_map",
    "build_mask",
    "dilate",
    "mask_from_color",
    "mask_from_image",
    "mask_from_rects",
    "__version__",
]
