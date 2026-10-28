#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""一行命令去水印（纯 Python，只依赖 numpy + pillow）。

    python remove_watermark.py inpaint  -i in.jpg -o out.png --rect 100,200,300,80
    python remove_watermark.py invert   -i in.jpg -o out.png -w watermark.png
    python remove_watermark.py estimate -i a.png b.png c.png -o out.png

完整说明见 README.md
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from watermark_remover.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
