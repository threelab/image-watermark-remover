"""命令行入口：

    python remove_watermark.py inpaint  -i in.jpg -o out.png --rect 100,200,300,80
    python remove_watermark.py invert   -i in.jpg -o out.png -w watermark.png
    python remove_watermark.py estimate -i a.png b.png c.png -o out.png
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional, Sequence

import numpy as np

from . import core, mask as mask_utils
from .utils import ensure_dir, info, parse_color, parse_rect, setup_console, warn


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="remove_watermark.py",
        description="纯 Python 图片去水印小工具（区域修复 / 水印反演 / 多图中值）",
    )
    parser.add_argument("--version", action="version", version="image-watermark-remover 0.1.0")
    sub = parser.add_subparsers(dest="command", required=True)

    # ---------------- 区域修复 ----------------
    p1 = sub.add_parser("inpaint", help="把水印区域用周围画面填掉（通用）")
    p1.add_argument("-i", "--input", required=True, help="输入图片")
    p1.add_argument("-o", "--output", required=True, help="输出图片")
    p1.add_argument("--mask", help="遮罩图，白色/不透明区域即待修复区域")
    p1.add_argument(
        "--rect",
        action="append",
        default=[],
        metavar="x,y,w,h",
        help="矩形区域，可重复传多个（右下角水印常用）",
    )
    p1.add_argument("--color", help="按颜色自动检测水印，如 255,255,255 或 #ffffff")
    p1.add_argument("--tolerance", type=float, default=40.0, help="颜色容差，默认 40")
    p1.add_argument("--dilate", type=int, default=2, help="遮罩向外扩张的圈数，默认 2")
    p1.add_argument("--radius", type=int, default=0, help="填充半径，默认按遮罩大小自动估计")
    p1.add_argument("--no-opencv", action="store_true", help="强制使用纯 numpy 实现")
    p1.add_argument("--save-mask", help="顺手把生成的遮罩存出来，方便检查")

    # ---------------- 水印反演 ----------------
    p2 = sub.add_parser("invert", help="已知水印图案时反解原图（效果最干净）")
    p2.add_argument("-i", "--input", required=True, help="输入图片")
    p2.add_argument("-o", "--output", required=True, help="输出图片")
    p2.add_argument("-w", "--watermark", required=True, help="水印图（带透明通道的 PNG 效果最好）")
    p2.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="水印强度 0-1，默认由水印图自动判断（有 alpha 通道取 1.0，否则 0.3）",
    )
    p2.add_argument("--strength", type=float, default=1.0, help="整体力度系数，默认 1.0")

    # ---------------- 多图中值 ----------------
    p3 = sub.add_parser("estimate", help="多张同款图、水印位置不同时取中值去水印")
    p3.add_argument("-i", "--input", nargs="+", required=True, help="同一画面的多张图片，建议 >= 3 张")
    p3.add_argument("-o", "--output", required=True, help="输出图片（估计出的干净背景）")
    p3.add_argument("--save-offset", help="把第一张的残差（水印残留）存出来看看效果")

    return parser


def _cmd_inpaint(args) -> int:
    img = core.imread(args.input)
    rects = [parse_rect(r) for r in args.rect]
    color = parse_color(args.color) if args.color else None
    mask_img = None
    if args.mask:
        mask_img = core.imread(args.mask, keep_alpha=True)

    m = mask_utils.build_mask(
        img,
        rects=rects,
        mask_image=mask_img,
        color=color,
        tolerance=args.tolerance,
        dilate_iters=args.dilate,
    )
    if not m.any():
        warn("没有生成任何遮罩，请检查 --rect / --mask / --color 参数")
        return 1

    info("图片尺寸: %dx%d" % (img.shape[1], img.shape[0]))
    info("待修复区域: %s" % mask_utils.mask_summary(m))
    if args.save_mask:
        ensure_dir(args.save_mask)
        core.imwrite(args.save_mask, (m.astype(np.uint8) * 255))
        info("遮罩已保存: %s" % args.save_mask)

    radius = args.radius if args.radius and args.radius > 0 else core.auto_radius(m)
    info("填充半径: %d（%s）" % (radius, "纯 numpy" if args.no_opencv else "优先 opencv"))
    out = core.inpaint(img, m, radius=radius, prefer_opencv=not args.no_opencv)

    ensure_dir(args.output)
    core.imwrite(args.output, out)
    info("已保存: %s" % args.output)
    return 0


def _cmd_invert(args) -> int:
    img = core.imread(args.input)
    wm = core.imread(args.watermark, keep_alpha=True)
    out = core.remove_by_watermark(img, wm, alpha=args.alpha, strength=args.strength)
    ensure_dir(args.output)
    core.imwrite(args.output, out)
    info("已保存: %s" % args.output)
    info("如果画面发灰/过冲，把 --alpha 或 --strength 调小再试（例如 0.6）")
    return 0


def _cmd_estimate(args) -> int:
    paths: List[str] = list(args.input)
    if len(paths) < 2:
        warn("至少需要 2 张图片")
        return 1
    images = [core.imread(p) for p in paths]
    base_shape = images[0].shape[:2]
    images = [core.resize_like(im, images[0]) for im in images]

    bg = core.estimate_background(images)
    ensure_dir(args.output)
    core.imwrite(args.output, bg)
    info("用了 %d 张图，尺寸 %dx%d" % (len(images), base_shape[1], base_shape[0]))
    info("已保存: %s" % args.output)

    if args.save_offset:
        off = core.offset_map(images[0], bg)
        ensure_dir(args.save_offset)
        core.imwrite(args.save_offset, off + 128.0)
        info("残差图已保存（灰色 = 无差异）: %s" % args.save_offset)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    setup_console()
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "inpaint":
            return _cmd_inpaint(args)
        if args.command == "invert":
            return _cmd_invert(args)
        if args.command == "estimate":
            return _cmd_estimate(args)
    except FileNotFoundError as exc:
        warn("找不到文件：%s" % exc)
        return 1
    except ValueError as exc:
        warn(str(exc))
        return 1

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
