"""基础自测：不需要外部素材，跑 `python -m unittest discover tests` 即可。"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from watermark_remover import core, mask as mask_utils  # noqa: E402
from watermark_remover.utils import parse_color, parse_rect  # noqa: E402


def flat(value: int, shape=(60, 80, 3)) -> np.ndarray:
    return np.full(shape, value, dtype=np.uint8)


class TestUtils(unittest.TestCase):
    def test_parse_rect(self):
        self.assertEqual(parse_rect("10,20,30,40"), (10, 20, 30, 40))

    def test_parse_color(self):
        self.assertEqual(parse_color("255,255,255"), [255, 255, 255])
        self.assertEqual(parse_color("#ffffff"), [255, 255, 255])

    def test_bad_rect(self):
        with self.assertRaises(ValueError):
            parse_rect("10,20")


class TestMask(unittest.TestCase):
    def test_rect_clipped(self):
        m = mask_utils.mask_from_rects((50, 50), [(-10, -10, 30, 30), (40, 40, 50, 50)])
        self.assertEqual(int(m.sum()), 20 * 20 + 10 * 10)

    def test_color(self):
        img = flat(200)
        img[0, 0] = (10, 10, 10)
        m = mask_utils.mask_from_color(img, (200, 200, 200), tolerance=10)
        self.assertFalse(m[0, 0])
        self.assertTrue(m[5, 5])

    def test_dilate(self):
        m = np.zeros((20, 20), dtype=bool)
        m[10, 10] = True
        self.assertEqual(int(mask_utils.dilate(m, 1).sum()), 9)
        self.assertEqual(int(mask_utils.dilate(m, 2).sum()), 25)


class TestInpaint(unittest.TestCase):
    def test_only_masked_area_changes(self):
        img = np.random.default_rng(0).integers(0, 255, (40, 40, 3), dtype=np.uint8)
        m = mask_utils.mask_from_rects(img.shape[:2], [(10, 10, 10, 10)])
        out = core.inpaint(img, m, prefer_opencv=False)
        self.assertTrue(np.array_equal(out[~m], img[~m]))
        self.assertFalse(np.array_equal(out[m], img[m]))

    def test_flat_region_restored(self):
        """纯色区域挖个洞，填回来应该还是那个颜色。"""
        img = flat(120, (60, 60, 3))
        m = mask_utils.mask_from_rects(img.shape[:2], [(20, 20, 20, 20)])
        out = core.inpaint(img, m, radius=8, prefer_opencv=False)
        diff = np.abs(out.astype(np.float64) - 120.0)
        self.assertLess(diff.max(), 2.0)

    def test_empty_mask_returns_copy(self):
        img = flat(30, (10, 10, 3))
        m = np.zeros((10, 10), dtype=bool)
        self.assertTrue(np.array_equal(core.inpaint(img, m), img))

    def test_auto_radius(self):
        m = mask_utils.mask_from_rects((200, 200), [(0, 0, 60, 10)])
        self.assertGreaterEqual(core.auto_radius(m), 4)


class TestInvert(unittest.TestCase):
    def test_roundtrip_with_alpha(self):
        """叠加 -> 反演，应当回到原图。"""
        rng = np.random.default_rng(1)
        origin = rng.integers(0, 255, (50, 50, 3), dtype=np.uint8)
        wm = np.zeros((50, 50, 4), dtype=np.uint8)
        wm[:, :, :3] = 255
        wm[10:40, 10:40, 3] = 128  # 水印区域 alpha ≈ 0.5

        a = wm[:, :, 3:4].astype(np.float64) / 255.0
        marked = (origin.astype(np.float64) * (1 - a) + 255.0 * a).astype(np.uint8)

        out = core.remove_by_watermark(marked, wm)
        err = np.abs(out.astype(np.float64) - origin.astype(np.float64)).mean()
        self.assertLess(err, 1.5)

    def test_rgb_watermark_uses_alpha_param(self):
        img = flat(200, (30, 30, 3))
        wm = flat(255, (30, 30, 3))
        out = core.remove_by_watermark(img, wm, alpha=0.5)
        expect = (200.0 - 0.5 * 255.0) / 0.5
        self.assertAlmostEqual(float(out[0, 0, 0]), float(np.clip(expect, 0, 255)), delta=1.0)


class TestEstimate(unittest.TestCase):
    def test_median_removes_random_watermark(self):
        rng = np.random.default_rng(2)
        clean = rng.integers(0, 255, (40, 40, 3), dtype=np.uint8)
        shots = []
        for _ in range(5):
            shot = clean.copy()
            y, x = int(rng.integers(0, 30)), int(rng.integers(0, 30))
            shot[y:y + 8, x:x + 8] = 0
            shots.append(shot)

        bg = core.estimate_background(shots)
        err = np.abs(bg.astype(np.float64) - clean.astype(np.float64)).mean()
        self.assertLess(err, 1.0)

    def test_needs_two_images(self):
        with self.assertRaises(ValueError):
            core.estimate_background([flat(1)])


if __name__ == "__main__":
    unittest.main(verbosity=2)
