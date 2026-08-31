import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import numpy as np

import sketch_to_gcode as app


class RouteAndRdpScalingTests(unittest.TestCase):
    @staticmethod
    def _path(x, y, length=2.0):
        return np.asarray([[x, y], [x + length, y]], dtype=np.float32)

    @staticmethod
    def _candidate(x, y, points=12):
        xs = np.linspace(x, x + 5.0, points, dtype=np.float32)
        ys = y + np.where(np.arange(points) % 2, 2.0, -2.0).astype(np.float32)
        polyline = np.column_stack([xs, ys]).astype(np.float32)
        return app.CandidatePath(
            points=polyline,
            importance=0.7,
            length_mm=float(np.linalg.norm(np.diff(polyline, axis=0), axis=1).sum()),
            detail_tier=2,
            saliency=0.7,
        )

    def test_large_route_uses_chunked_optimizers_and_logs_coverage(self):
        paths = [
            self._path(float(index % 50) * 4.0, float(index // 50) * 4.0)
            for index in range(app.OR_OPT_MAX_PATHS + 50)
        ]

        def unchanged(route, time_budget_s, **kwargs):
            return list(route), 0.0

        output = io.StringIO()
        with patch.object(app, "_two_opt_route_improve", side_effect=unchanged) as two_opt, \
                patch.object(app, "_or_opt_route_improve", side_effect=unchanged) as or_opt, \
                redirect_stdout(output):
            result, _, _ = app._postprocess_route_improve(paths)

        self.assertEqual(len(result), len(paths))
        self.assertGreater(two_opt.call_count, 0)
        self.assertGreater(or_opt.call_count, 0)
        self.assertIn(
            f"{len(paths)}/{len(paths)} paths (100.0%)",
            output.getvalue(),
        )

    def test_dense_micro_texture_is_reduced_before_budgeting(self):
        candidates = [
            app.CandidatePath(
                points=self._path(1.0 + index * 0.05, 2.0),
                importance=0.6,
                length_mm=2.0,
                detail_tier=2,
                saliency=0.6,
            )
            for index in range(20)
        ]
        output = io.StringIO()
        with redirect_stdout(output):
            consolidated = app._consolidate_dense_micro_candidates(candidates)

        self.assertLess(len(consolidated), len(candidates))
        self.assertGreaterEqual(len(consolidated), 8)
        self.assertIn("Dense micro-texture: 20 ->", output.getvalue())

    def test_rdp_pressure_reselects_candidates_before_exceeding_warning_cap(self):
        candidates = [
            self._candidate(float(index % 40) * 3.0, float(index // 40) * 3.0)
            for index in range(400)
        ]

        # Model a detail-heavy source where every retained path still needs ten
        # segments at the warning epsilon. This isolates the rebalance policy.
        def detail_heavy_simplify(selected, epsilon, curvatures=None):
            point_count = 11 if epsilon <= app.RDP_EPS_WARNING_MM else 2
            return [candidate.points[:point_count] for candidate in selected]

        output = io.StringIO()
        with patch.object(app, "_simplify_all", side_effect=detail_heavy_simplify), \
                redirect_stdout(output):
            paths, epsilon, _, _, warning = app._fit_paths_to_budget(
                candidates, app.MIN_STROKE_BUDGET)

        self.assertTrue(warning)
        self.assertLessEqual(epsilon, app.RDP_EPS_WARNING_MM)
        self.assertLessEqual(app._count_segments(paths), app.MIN_STROKE_BUDGET)
        self.assertIn("RDP rebalance pass", output.getvalue())


if __name__ == "__main__":
    unittest.main()
