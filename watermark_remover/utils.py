"""命令行里用到的小工具：参数解析、目录准备、终端输出。"""

from __future__ import annotations

import os
import sys
from typing import List, Sequence, Tuple

__all__ = ["parse_rect", "parse_color", "ensure_dir", "info", "warn", "setup_console"]


def setup_console() -> None:
    """Windows 控制台默认可能是 GBK，强制用 UTF-8 输出中文提示。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass


def parse_rect(text: str) -> Tuple[int, int, int, int]:
    """解析 ``x,y,w,h``。"""
    parts = [p.strip() for p in text.replace("x", ",").split(",") if p.strip()]
    if len(parts) != 4:
        raise ValueError("矩形格式应为 x,y,w,h，例如 100,200,300,80")
    try:
        x, y, w, h = (int(float(p)) for p in parts)
    except ValueError:
        raise ValueError("矩形里的数字无法识别：%s" % text)
    if w <= 0 or h <= 0:
        raise ValueError("矩形的宽高必须大于 0：%s" % text)
    return x, y, w, h


def parse_color(text: str) -> List[int]:
    """解析 ``255,255,255`` 或 ``#ffffff``。"""
    text = text.strip()
    if text.startswith("#"):
        text = text[1:]
        if len(text) == 3:
            text = "".join(c * 2 for c in text)
        if len(text) != 6:
            raise ValueError("颜色格式应为 #rrggbb 或 r,g,b")
        try:
            return [int(text[i:i + 2], 16) for i in (0, 2, 4)]
        except ValueError:
            raise ValueError("颜色里的十六进制无法识别：%s" % text)

    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError("颜色格式应为 r,g,b，例如 255,255,255")
    try:
        vals = [int(float(p)) for p in parts]
    except ValueError:
        raise ValueError("颜色里的数字无法识别：%s" % text)
    if any(v < 0 or v > 255 for v in vals):
        raise ValueError("颜色分量必须在 0-255 之间：%s" % text)
    return vals


def ensure_dir(path: str) -> str:
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)
    return d


def info(msg: str) -> None:
    print(msg)


def warn(msg: str) -> None:
    print("[提示] %s" % msg, file=sys.stderr)
