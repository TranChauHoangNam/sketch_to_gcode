import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from reporting import ReportConfig, generate_report, generate_pdf_report
from reporting.analyzer import analyze_report
from reporting.metrics import build_report_data
from reporting.models import NOT_AVAILABLE
from reporting.rules import RuleConfig, analyze_rules


class ReportingTests(unittest.TestCase):
    def make_plan(self, **overrides):
        values = dict(
            paths=[[(0, 0), (10, 0)]], target_segments=100, actual_segments=90,
            stroke_count=1, pen_lifts=1, command_count=10, raw_segment_count=100,
            stitched_count=0, travel_distance_mm=5.0, draw_distance_mm=10.0,
            estimated_time_s=20.0, route_improvement_mm=2.0, route_improvement_pct=20.0,
            clipped_path_count=0, deduplicated_path_count=0, island_count=1,
            gcode_postprocess_removed_count=1, compressed_segment_count=9,
            resampled_point_count=0, stroke_direction_flip_count=0,
            dry_run_warning_count=0, pen_lift_bridge_count=0, planning_time_s=0.1,
            gcode_size_bytes=1000, ai_backend="classical", ai_elapsed_s=0.0,
            hatch_candidate_count=3, validation_errors=[],
        )
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_missing_values_are_not_available(self):
        data = build_report_data(SimpleNamespace(paths=[], validation_errors=[]))
        self.assertEqual(data.input["filename"], NOT_AVAILABLE)
        self.assertEqual(data.gcode["total_commands"], NOT_AVAILABLE)

    def test_rule_detects_validation_and_travel(self):
        plan = self.make_plan(validation_errors=["bad coordinate"], travel_distance_mm=100)
        data = build_report_data(plan)
        result = analyze_rules(data, RuleConfig(max_travel_draw_ratio=2.0))
        self.assertTrue(result["problems"])
        self.assertTrue(result["warnings"])

    def test_generator_writes_pdf(self):
        with tempfile.TemporaryDirectory() as directory:
            result = generate_report(
                pipeline_result=self.make_plan(),
                report_config=ReportConfig(output_dir=directory),
                stem="run",
            )
            self.assertEqual(set(result["paths"]), {"pdf"})
            self.assertTrue(Path(result["paths"]["pdf"]).exists())

    def test_pdf_api_writes_single_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = generate_pdf_report(pipeline_result=self.make_plan(),
                                       output_path=str(Path(directory) / "run_report.pdf"))
            self.assertTrue(Path(path).exists())

    def test_llm_failure_falls_back(self):
        class BrokenLLM:
            def analyze(self, report_data):
                raise RuntimeError("offline")

        with tempfile.TemporaryDirectory() as directory:
            result = generate_report(
                pipeline_result=self.make_plan(),
                report_config=ReportConfig(output_dir=directory, llm=BrokenLLM()),
            )
            self.assertEqual(result["analysis"]["llm_status"], "LLM analysis unavailable")

    def test_analysis_schema_has_required_keys(self):
        analysis = analyze_report(build_report_data(self.make_plan())).to_dict()
        required = {"summary", "overall_score", "quality_level", "performance",
                    "path_quality", "gcode_quality", "optimization", "problems",
                    "warnings", "recommendations", "important_metrics", "confidence"}
        self.assertTrue(required.issubset(analysis))


if __name__ == "__main__":
    unittest.main()
