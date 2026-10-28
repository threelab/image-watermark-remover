"""生成一组演示素材：干净底图 + 纯水印图 + 加水印后的图。

    python examples/make_sample.py

生成在 examples/samples/ 下，方便直接跑 demo.py 看效果。
"""

from __future__ import annotations

import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "samples")

W, H = 800, 500
TEXT = "SAMPLE 2026"


def build_base() -> Image.Image:
    """画一张有渐变和几何图案的底图，方便肉眼判断修复效果。"""
    x = np.linspace(0, 1, W, dtype=np.float32)
    y = np.linspace(0, 1, H, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    r = (40 + 180 * xx).astype(np.float32)
    g = (60 + 120 * yy).astype(np.float32)
    b = (120 + 100 * (1 - xx * yy)).astype(np.float32)

    # 加几道斜条纹，修复算法糊掉的话一眼能看出来
    stripes = (np.sin((xx * W + yy * H) / 28.0) > 0).astype(np.float32) * 24.0
    rgb = np.stack([r + stripes, g + stripes * 0.6, b], axis=2)
    img = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))

    d = ImageDraw.Draw(img)
    d.ellipse((120, 90, 320, 290), fill=(250, 210, 90))
    d.rectangle((430, 250, 700, 420), fill=(240, 90, 120))
    return img


def load_font(size: int) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "Arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=size)
    except Exception:
        return ImageFont.load_default()


def build_watermark() -> Image.Image:
    """白色半透明文字，带 alpha 通道，正好是 invert 模式需要的输入。"""
    layer = Image.new("RGBA", (W, H), (255, 255, 255, 0))
    d = ImageDraw.Draw(layer)
    font = load_font(46)
    d.text((430, 400), TEXT, font=font, fill=(255, 255, 255, 190))
    d.text((60, 60), TEXT, font=font, fill=(255, 255, 255, 160))
    return layer


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    base = build_base()
    wm = build_watermark()
    marked = Image.alpha_composite(base.convert("RGBA"), wm)

    base.save(os.path.join(OUT_DIR, "clean_input.png"))
    wm.save(os.path.join(OUT_DIR, "watermark.png"))
    marked.convert("RGB").save(os.path.join(OUT_DIR, "watermarked.png"))

    # 右下角水印的矩形框（x, y, w, h），inpaint 演示用
    with open(os.path.join(OUT_DIR, "rect.txt"), "w", encoding="utf-8") as fp:
        fp.write("415,390,340,70")

    print("已生成演示素材 -> %s" % OUT_DIR)
    for name in ("clean_input.png", "watermark.png", "watermarked.png"):
        print("  %s" % name)


if __name__ == "__main__":
    main()
