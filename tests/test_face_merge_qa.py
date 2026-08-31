import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

import sketch_to_gcode as app


class FaceMergeQATests(unittest.TestCase):
    def test_classifier_marks_face_strokes_protected(self):
        lineart = Image.new("L", (100, 100), 255)
        face_mask = np.zeros((100, 100), dtype=np.uint8)
        face_mask[34:66, 34:66] = 255
        candidates = [
            app.CandidatePath(
                points=np.asarray([[132.0, 200.0], [168.0, 200.0]], dtype=np.float32),
                importance=0.35,
                length_mm=36.0,
                detail_tier=1,
                saliency=0.30,
            ),
            app.CandidatePath(
                points=np.asarray([[50.0, 90.0], [250.0, 90.0]], dtype=np.float32),
                importance=0.45,
                length_mm=200.0,
                detail_tier=1,
                saliency=0.40,
            ),
        ]

        classified = app.classify_strokes(
            candidates,
            app.StrokeClassificationContext(
                canvas_size=lineart.size,
                lineart_img=lineart,
                face_mask=face_mask,
            ),
        )

        self.assertEqual(classified[0].region, "face")
        self.assertTrue(classified[0].protected)
        self.assertEqual(classified[0].detail_tier, 0)

    def test_semantic_budget_keeps_face_and_outline_before_texture(self):
        face = app.CandidatePath(
            points=np.asarray([[132.0, 200.0], [168.0, 200.0]], dtype=np.float32),
            importance=0.45,
            length_mm=36.0,
            detail_tier=0,
            saliency=0.50,
            region="face",
            protected=True,
        )
        outline = app.CandidatePath(
            points=np.asarray([[30.0, 110.0], [270.0, 110.0]], dtype=np.float32),
            importance=0.35,
            length_mm=240.0,
            detail_tier=0,
            saliency=0.42,
            region="garment_outline",
        )
        texture = app.CandidatePath(
            points=np.asarray([[40.0, 95.0], [45.0, 96.0]], dtype=np.float32),
            importance=0.99,
            length_mm=5.1,
            detail_tier=2,
            saliency=0.99,
            region="fine_detail",
        )

        selected = app._select_candidates_for_budget(
            [texture, outline, face], target=2)
        regions = {candidate.region for candidate in selected}

        self.assertIn("face", regions)
        self.assertIn("garment_outline", regions)
        self.assertNotIn("fine_detail", regions)

    def test_manual_reduce_protects_face_without_confirmation(self):
        face_mask = np.zeros((100, 100), dtype=np.uint8)
        face_mask[34:66, 34:66] = 255
        brush_mask = app.make_brush_mask([(150.0, 200.0)], 80.0, (100, 100))
        self.assertTrue(app.brush_overlaps_face(brush_mask, face_mask))

        face = app.CandidatePath(
            points=np.asarray([[132.0, 200.0], [168.0, 200.0]], dtype=np.float32),
            importance=0.90,
            length_mm=36.0,
            detail_tier=0,
            saliency=0.86,
            region="face",
            protected=True,
        )
        hatch = app.CandidatePath(
            points=np.asarray([[144.0, 125.0], [156.0, 125.0]], dtype=np.float32),
            importance=0.30,
            length_mm=12.0,
            detail_tier=2,
            saliency=0.25,
            region="hatch",
            source="hatch",
        )

        updated, stats = app.apply_manual_brush_adjustment(
            [face, hatch],
            Image.new("RGB", (100, 100), "white"),
            brush_mask,
            "reduce",
            face_mask=face_mask,
            allow_face_reduce=False,
        )

        self.assertEqual(stats["removed"], 1)
        self.assertEqual(len(updated), 1)
        self.assertEqual(updated[0].region, "face")
        self.assertTrue(updated[0].protected)

    def test_micro_filter_keeps_short_protected_face_strokes(self):
        face_dot = app.CandidatePath(
            points=np.asarray([[148.0, 200.0], [151.0, 200.0]], dtype=np.float32),
            importance=0.90,
            length_mm=3.0,
            detail_tier=0,
            saliency=0.90,
            region="face",
            protected=True,
        )
        isolated_noise = app.CandidatePath(
            points=np.asarray([[20.0, 20.0], [20.2, 20.0]], dtype=np.float32),
            importance=0.10,
            length_mm=0.2,
            detail_tier=2,
            saliency=0.10,
            region="fine_detail",
        )

        filtered = app._filter_micro_strokes([face_dot, isolated_noise])

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].region, "face")
        self.assertTrue(filtered[0].protected)

    def test_hatch_exclude_mask_removes_face_region(self):
        row = np.linspace(20, 120, 100, dtype=np.uint8)
        tone = np.tile(row, (80, 1))
        fg = np.full(tone.shape, 255, dtype=np.uint8)
        exclude = np.zeros(tone.shape, dtype=np.uint8)
        exclude[:, :50] = 255

        lines = app._create_hatching_vectors(
            tone,
            fg,
            cell_size=10,
            angle_deg=0.0,
            min_spacing=4,
            max_spacing=8,
            dark_threshold=220.0,
            contour_following=False,
            exclude_mask=exclude,
        )

        self.assertTrue(lines)
        for polyline in lines:
            for x, _y in polyline:
                self.assertGreaterEqual(x, 49.0)

    def test_merge_collinear_chain_and_remove_overlap(self):
        strokes = [
            np.asarray([[0.0, 0.0], [5.0, 0.0]], dtype=np.float32),
            np.asarray([[5.3, 0.0], [10.0, 0.0]], dtype=np.float32),
            np.asarray([[10.4, 0.0], [15.0, 0.0]], dtype=np.float32),
            np.asarray([[0.0, 0.08], [4.7, 0.08]], dtype=np.float32),
        ]

        merged, stats = app.merge_collinear_and_touching_strokes(
            strokes, endpoint_gap_mm=0.6, return_stats=True)

        self.assertLess(len(merged), len(strokes))
        self.assertGreaterEqual(stats["merged_count"], 2)
        self.assertGreaterEqual(stats["overlap_removed_count"], 1)

    def test_quality_gate_runs_three_iterations(self):
        reference = Image.new("L", (80, 80), 255)
        candidates = [
            app.CandidatePath(
                points=np.asarray([[20.0, 20.0], [80.0, 20.0]], dtype=np.float32),
                importance=1.0,
                length_mm=60.0,
                detail_tier=0,
                saliency=1.0,
            )
        ]

        with patch.object(app, "_call_anthropic_vision_qa", return_value={
            "status": "ok",
            "fidelity_score": 82,
            "redundancy_notes": "",
            "missing_detail_notes": "",
        }):
            plan, report = app.run_quality_gate(
                candidates,
                app.MIN_STROKE_BUDGET,
                reference,
                original_img=reference,
                timeout_s=30.0,
            )

        self.assertIs(plan.qa_report, report)
        self.assertEqual(len(report.iterations), app.QA_MIN_ITERATIONS)
        self.assertEqual(report.final_model_score, 82.0)


if __name__ == "__main__":
    unittest.main()
