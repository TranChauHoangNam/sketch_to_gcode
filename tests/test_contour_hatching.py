import math
import io
import unittest
from contextlib import redirect_stdout

import numpy as np

from hatching import generate_hatching
from sketch_to_gcode import (
    CandidatePath,
    DEFAULT_HATCH_DARK_THRESHOLD,
    HATCH_CANDIDATE_DETAIL_TIER,
    HATCH_PRESETS,
    STITCH_THRESHOLD_MM,
    _adaptive_hatch_threshold,
    _create_hatching_vectors,
    _estimate_path_pen_lifts,
    _local_hatch_orientation_map,
    _select_candidates_for_budget,
    _stitch_paths,
)


def _axial_distance(angle, expected):
    return abs((float(angle) - float(expected) + 90.0) % 180.0 - 90.0)


class ContourHatchingTests(unittest.TestCase):
    def test_dark_threshold_defaults_are_strict(self):
        self.assertEqual(DEFAULT_HATCH_DARK_THRESHOLD, 160.0)
        self.assertEqual(HATCH_PRESETS["Balanced"]["dark_threshold"], 160.0)
        self.assertTrue(all(
            140.0 <= settings["dark_threshold"] <= 170.0
            for settings in HATCH_PRESETS.values()
        ))
        self.assertEqual(HATCH_CANDIDATE_DETAIL_TIER, 2)
        self.assertEqual(STITCH_THRESHOLD_MM, 3.2)

    def test_mm_path_stitching_draws_through_nearby_aligned_gap(self):
        paths = [
            np.asarray([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32),
            np.asarray([[5.0, 0.0], [8.0, 0.0]], dtype=np.float32),
        ]
        self.assertEqual(_estimate_path_pen_lifts(paths), 2)
        stitched, count = _stitch_paths(paths)
        self.assertEqual(count, 1)
        self.assertEqual(len(stitched), 1)
        self.assertEqual(_estimate_path_pen_lifts(stitched), 1)

    def test_fine_detail_budget_prefers_continuous_long_stroke(self):
        short = CandidatePath(
            points=np.asarray([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32),
            importance=0.5, length_mm=2.0, detail_tier=2, saliency=0.5)
        long = CandidatePath(
            points=np.asarray([[0.0, 0.0], [80.0, 0.0]], dtype=np.float32),
            importance=0.5, length_mm=80.0, detail_tier=2, saliency=0.5)

        selected = _select_candidates_for_budget([short, long], target=1)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].length_mm, 80.0)

    def test_adaptive_threshold_targets_shadow_percentile(self):
        row = np.linspace(40, 220, 200, dtype=np.uint8)
        tone = np.tile(row, (80, 1))
        foreground = np.full(tone.shape, 255, dtype=np.uint8)

        threshold, eligible, foreground_count = _adaptive_hatch_threshold(
            tone, foreground, requested_threshold=160.0, cell_size=1)
        coverage = 100.0 * np.count_nonzero(eligible) / foreground_count

        self.assertLessEqual(threshold, 160.0)
        self.assertGreaterEqual(coverage, 20.0)
        self.assertLessEqual(coverage, 32.0)

    def test_bright_foreground_is_not_hatched_and_coverage_is_logged(self):
        tone = np.full((60, 80), 215, dtype=np.uint8)
        foreground = np.full(tone.shape, 255, dtype=np.uint8)
        threshold, eligible, _ = _adaptive_hatch_threshold(
            tone, foreground, requested_threshold=160.0, cell_size=18)
        self.assertEqual(threshold, 160.0)
        self.assertEqual(np.count_nonzero(eligible), 0)

        output = io.StringIO()
        with redirect_stdout(output):
            lines = _create_hatching_vectors(tone, foreground)
        self.assertEqual(lines, [])
        self.assertIn("Hatch coverage: 0.0% foreground", output.getvalue())

    def test_none_orientation_preserves_fixed_angle_output(self):
        image = np.full((60, 80), 110, dtype=np.uint8)
        kwargs = dict(
            cell_size=18,
            angle_deg=35.0,
            min_spacing=3,
            max_spacing=12,
            dark_threshold=235.0,
        )
        legacy = generate_hatching(image, **kwargs)
        explicit_none = generate_hatching(image, orientation_map=None, **kwargs)
        self.assertEqual(legacy, explicit_none)

    def test_orientation_map_changes_hatch_direction_by_region(self):
        image = np.full((80, 120), 100, dtype=np.uint8)
        orientation = np.zeros(image.shape, dtype=np.float32)
        orientation[:, 60:] = 90.0
        output = io.StringIO()
        with redirect_stdout(output):
            lines = generate_hatching(
                image,
                cell_size=20,
                angle_deg=35.0,
                min_spacing=4,
                max_spacing=10,
                dark_threshold=235.0,
                orientation_map=orientation,
            )
        self.assertLess(len(lines), 40)
        self.assertIn("Oriented hatch stitching: 66 ->", output.getvalue())

        measured = []
        for line in lines:
            if len(line) < 2:
                continue
            x_mid = sum(point[0] for point in line) / len(line)
            dx = line[-1][0] - line[0][0]
            dy = line[-1][1] - line[0][1]
            measured.append((x_mid, math.degrees(math.atan2(dy, dx)) % 180.0))

        left = [angle for x_mid, angle in measured if x_mid < 45.0]
        right = [angle for x_mid, angle in measured if x_mid > 75.0]
        self.assertTrue(left)
        self.assertTrue(right)
        self.assertLess(np.median([_axial_distance(angle, 0.0) for angle in left]), 8.0)
        self.assertLess(np.median([_axial_distance(angle, 90.0) for angle in right]), 8.0)

    def test_gradient_orientation_is_tangent_to_edge(self):
        image = np.zeros((80, 80), dtype=np.uint8)
        image[:, 40:] = 255
        orientation = _local_hatch_orientation_map(
            image, fallback_angle_deg=35.0, cell_size=20)

        edge_angles = orientation[12:68, 38:42]
        distances = np.vectorize(_axial_distance)(edge_angles, 90.0)
        self.assertLess(float(np.median(distances)), 12.0)


if __name__ == "__main__":
    unittest.main()
