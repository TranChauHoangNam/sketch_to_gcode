import unittest

import numpy as np
from PIL import Image, ImageDraw

import sketch_to_gcode as app
from sketch_to_gcode import _composite_rgb_and_mask, transform_to_lineart


class LineArtInputTests(unittest.TestCase):
    def test_rembg_entry_points_are_removed_from_main_pipeline(self):
        for name in (
            "AI_MODEL_NAME",
            "AI_MODEL_LABELS",
            "remove_background",
            "_build_ai_maps",
            "_decontaminate_edges",
            "transform_sketch_to_lineart",
        ):
            self.assertFalse(hasattr(app, name), name)

    def test_rgba_input_uses_soft_alpha_without_background_removal(self):
        rgba = np.zeros((9, 9, 4), dtype=np.uint8)
        rgba[:, :, :3] = 80
        rgba[3:6, 3:6, 3] = 255
        image = Image.fromarray(rgba, "RGBA")

        _, mask = _composite_rgb_and_mask(image)

        self.assertGreater(np.count_nonzero(mask), 9)
        self.assertLess(np.count_nonzero(mask), mask.size)

    def test_transform_lineart_accepts_transparent_line_art(self):
        image = Image.new("RGBA", (80, 80), (255, 255, 255, 0))
        draw = ImageDraw.Draw(image)
        draw.line((12, 40, 68, 40), fill=(0, 0, 0, 255), width=3)

        lineart = transform_to_lineart(image)
        ink = np.asarray(lineart.convert("L")) < 180

        self.assertGreater(np.count_nonzero(ink), 20)

    def test_exclude_mask_clips_hatch_but_keeps_structural_line(self):
        image = Image.new("RGB", (100, 80), (92, 92, 92))
        draw = ImageDraw.Draw(image)
        draw.line((10, 40, 90, 40), fill=(0, 0, 0), width=3)
        exclude = np.zeros((80, 100), dtype=np.uint8)
        exclude[:, :50] = 255

        lineart, hatch_lines = transform_to_lineart(
            image,
            {
                "cell_size": 10,
                "angle_deg": 0.0,
                "min_spacing": 4,
                "max_spacing": 8,
                "dark_threshold": 220.0,
                "contour_following": False,
            },
            return_hatch_vectors=True,
            exclude_mask=exclude,
        )

        left_line_region = np.asarray(lineart.convert("L"))[36:45, 10:45] < 220
        self.assertGreater(np.count_nonzero(left_line_region), 10)
        self.assertTrue(hatch_lines)
        for polyline in hatch_lines:
            for x, _y in polyline:
                self.assertGreaterEqual(x, 49.0)


if __name__ == "__main__":
    unittest.main()
