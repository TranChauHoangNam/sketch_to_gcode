import io
import unittest
from contextlib import redirect_stdout

import numpy as np

from sketch_to_gcode import (
    BUDGET_GRID_MM,
    CandidatePath,
    _candidate_center,
    _candidate_min_segment_cost,
    _grid_key,
    _select_candidates_for_budget,
)


class BudgetCoverageTests(unittest.TestCase):
    @staticmethod
    def _candidate(x, y, saliency):
        points = np.asarray([[x, y], [x + 1.0, y]], dtype=np.float32)
        return CandidatePath(
            points=points,
            importance=saliency,
            length_mm=1.0,
            detail_tier=0,
            saliency=saliency,
        )

    def test_coverage_pass_selects_one_winner_per_cell_first(self):
        # Six high-value strokes compete in one noisy cell, while other cells
        # contain lower-value base contours that still need representation.
        candidates = [
            self._candidate(2.0 + offset, 2.0, 1.0 - offset * 0.01)
            for offset in range(6)
        ]
        candidates.extend([
            self._candidate(40.0, 2.0, 0.70),
            self._candidate(75.0, 2.0, 0.60),
            self._candidate(110.0, 2.0, 0.50),
            self._candidate(145.0, 2.0, 0.40),
        ])

        output = io.StringIO()
        with redirect_stdout(output):
            selected = _select_candidates_for_budget(candidates, target=10)

        coverage_keys = {
            _grid_key(_candidate_center(candidate), BUDGET_GRID_MM)
            for candidate in selected[:3]
        }
        self.assertEqual(len(coverage_keys), 3)
        noisy_cell = _grid_key(_candidate_center(candidates[0]), BUDGET_GRID_MM)
        noisy_winners = [
            candidate for candidate in selected[:3]
            if _grid_key(_candidate_center(candidate), BUDGET_GRID_MM) == noisy_cell
        ]
        self.assertEqual(len(noisy_winners), 1)
        self.assertEqual(noisy_winners[0].saliency, 1.0)
        self.assertIn("Budget coverage: 3/5 grid cells covered", output.getvalue())
        self.assertIn("cost 3/3", output.getvalue())

        selected_paths = {
            tuple(sorted(map(tuple, candidate.points.tolist())))
            for candidate in selected
        }
        self.assertEqual(len(selected_paths), len(selected))
        self.assertLessEqual(
            sum(_candidate_min_segment_cost(candidate) for candidate in selected),
            10,
        )


if __name__ == "__main__":
    unittest.main()
