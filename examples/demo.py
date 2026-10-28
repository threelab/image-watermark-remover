"""三条路线各跑一遍，结果输出到 examples/out/。

    python examples/make_sample.py
    python examples/demo.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from watermark_remover import core, mask as mask_utils  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLES = os.path.join(HERE, "samples")
OUT = os.path.join(HERE, "out")


def ensure_samples() -> None:
    if not os.path.isdir(SAMPLES) or not os.path.isfile(os.path.join(SAMPLES, "watermarked.png")):
        raise SystemExit("先跑 python examples/make_sample.py 生成素材")


def demo_inpaint() -> None:
    img = core.imread(os.path.join(SAMPLES, "watermarked.png"))
    with open(os.path.join(SAMPLES, "rect.txt"), encoding="utf-8") as fp:
        from watermark_remover.utils import parse_rect

        rect = parse_rect(fp.read().strip())

    m = mask_utils.build_mask(img, rects=[rect], dilate_iters=3)
    core.imwrite(os.path.join(OUT, "inpaint_mask.png"), m.astype(np.uint8) * 255)
    out = core.inpaint(img, m)
    core.imwrite(os.path.join(OUT, "inpaint_result.png"), out)
    print("inpaint  -> out/inpaint_result.png  区域: %s" % mask_utils.mask_summary(m))


def demo_invert() -> None:
    img = core.imread(os.path.join(SAMPLES, "watermarked.png"))
    wm = core.imread(os.path.join(SAMPLES, "watermark.png"), keep_alpha=True)
    out = core.remove_by_watermark(img, wm)
    core.imwrite(os.path.join(OUT, "invert_result.png"), out)

    clean = core.imread(os.path.join(SAMPLES, "clean_input.png"))
    err = float(np.abs(out.astype(np.float64) - clean.astype(np.float64)).mean())
    print("invert   -> out/invert_result.png  与原图平均误差: %.2f/255" % err)


def demo_estimate() -> None:
    """水印位置随机时，多张取中值即可滤掉。"""
    rng = np.random.default_rng(7)
    clean = core.imread(os.path.join(SAMPLES, "clean_input.png"))
    shots = []
    h, w = clean.shape[:2]
    for _ in range(5):
        y = int(rng.integers(0, h - 70))
        x = int(rng.integers(0, w - 260))
        shot = clean.copy()
        # 模拟一块随机位置的半透明暗水印
        shot[y:y + 60, x:x + 250] = (shot[y:y + 60, x:x + 250].astype(np.float64) * 0.6).astype(np.uint8)
        shots.append(shot)

    bg = core.estimate_background(shots)
    core.imwrite(os.path.join(OUT, "estimate_result.png"), bg)
    err = float(np.abs(bg.astype(np.float64) - clean.astype(np.float64)).mean())
    print("estimate -> out/estimate_result.png 与原图平均误差: %.2f/255" % err)


def main() -> None:
    ensure_samples()
    os.makedirs(OUT, exist_ok=True)
    demo_inpaint()
    demo_invert()
    demo_estimate()
    print("全部完成，结果在 %s" % OUT)


if __name__ == "__main__":
    main()
