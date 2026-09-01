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

    def test_sketch_pipeline_keeps_structural_line_without_photo_hatch(self):
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
        self.assertEqual(hatch_lines, [])

    def test_clean_line_art_uses_compact_fast_path(self):
        image = Image.new("RGB", (600, 600), "white")
        draw = ImageDraw.Draw(image)
        draw.ellipse((80, 90, 500, 390), outline="black", width=9)
        for y in (210, 245, 280):
            draw.line((55, y, 170, y - 30), fill="black", width=6)
            draw.line((410, y - 30, 555, y), fill="black", width=6)
        draw.ellipse((210, 190, 250, 230), fill="black")
        draw.ellipse((345, 190, 385, 230), fill="black")
        draw.ellipse((285, 240, 320, 260), outline="black", width=6)
        draw.arc((235, 260, 365, 335), 0, 180, fill="black", width=5)

        lineart, hatch_lines = transform_to_lineart(
            image, return_hatch_vectors=True)
        candidates = app.extract_candidate_paths(lineart, reference_img=image)
        auto_budget = app.suggest_auto_detail_budget(candidates)
        plan = app.build_gcode_plan(candidates, auto_budget)

        self.assertEqual(lineart.info.get("vector_mode"), "clean_lineart")
        self.assertGreaterEqual(lineart.info.get("filled_region_count", 0), 2)
        self.assertEqual(hatch_lines, [])
        ink = np.asarray(lineart.convert("L")) < 180
        self.assertGreater(np.count_nonzero(ink[190:232, 210:252]), 55)
        self.assertGreater(np.count_nonzero(ink[190:232, 345:387]), 55)
        self.assertLess(np.count_nonzero(ink[190:232, 210:252]), 700)
        self.assertLess(np.count_nonzero(ink[190:232, 345:387]), 700)
        self.assertLess(len(candidates), 80)
        self.assertLessEqual(auto_budget, 900)
        self.assertLess(plan.stroke_count, 40)
        self.assertLess(plan.actual_segments, 1000)

    def test_overlay_quality_gate_repairs_missing_eye_detail(self):
        image = Image.new("RGB", (420, 320), "white")
        draw = ImageDraw.Draw(image)
        draw.ellipse((55, 55, 365, 245), outline="black", width=7)
        draw.ellipse((140, 130, 170, 160), fill="black")
        draw.ellipse((250, 130, 280, 160), fill="black")
        draw.ellipse((195, 170, 225, 187), outline="black", width=5)

        lineart, _ = transform_to_lineart(image, return_hatch_vectors=True)
        candidates = app.extract_candidate_paths(lineart, reference_img=image)

        def overlaps_eye(candidate):
            hits = 0
            for point in candidate.points:
                x, y = app._fit_page_to_pixel(
                    point[0], point[1], lineart.size[0], lineart.size[1])
                hits += int(130 <= y <= 162 and (138 <= x <= 172 or 248 <= x <= 282))
            return hits > 0

        broken = [candidate for candidate in candidates if not overlaps_eye(candidate)]
        plan = app.build_gcode_plan(broken, app.suggest_auto_detail_budget(broken))
        before = app._overlay_missing_mask(plan, lineart)
        _repaired_candidates, repaired_plan, stats = app.repair_missing_details_with_overlay(
            broken,
            plan,
            plan.target_segments,
            lineart,
        )
        after = app._overlay_missing_mask(repaired_plan, lineart)

        self.assertGreater(before["missing_pixels"], app.QA_OVERLAY_MIN_MISSING_PIXELS)
        self.assertTrue(stats["accepted"])
        self.assertGreater(stats["added_candidates"], 0)
        self.assertLess(after["missing_pixels"], before["missing_pixels"])


if __name__ == "__main__":
    unittest.main()
