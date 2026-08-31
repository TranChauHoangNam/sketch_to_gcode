from __future__ import annotations

from typing import Any

from .models import AnalysisResult, NOT_AVAILABLE, ReportData
from .rules import RuleConfig, analyze_rules


def quality_level(score: Any) -> str:
    if not isinstance(score, (int, float)):
        return NOT_AVAILABLE
    return "Excellent" if score >= 90 else "Good" if score >= 75 else "Fair" if score >= 55 else "Poor"


def analyze_report(report: ReportData, rule_config: RuleConfig | None = None) -> AnalysisResult:
    rules = analyze_rules(report, rule_config)
    score = rules["score"]
    available = sum(v != NOT_AVAILABLE for section in report.to_dict().values() if isinstance(section, dict) for v in section.values())
    confidence = min(100, round(available / max(1, len(report.raw_metrics)) * 100))
    return AnalysisResult(
        summary=(rules["problems"][0] if rules["problems"] else "Rule-based analysis completed."),
        overall_score=score,
        quality_level=quality_level(score),
        performance={"score": score, "analysis": "Based only on available distance, lift, timing, and validation metrics."},
        path_quality={"score": score, "analysis": "Based only on available path and validation metrics."},
        gcode_quality={"score": score, "analysis": "Based only on available G-code counts, size, and validation metrics."},
        optimization={"score": report.optimization.get("improvement_pct", NOT_AVAILABLE),
                      "analysis": "Route improvement is reported directly from the pipeline."},
        problems=rules["problems"], warnings=rules["warnings"], recommendations=rules["recommendations"],
        important_metrics={k: report.raw_metrics.get(k, NOT_AVAILABLE) for k in (
            "actual_segments", "stroke_count", "pen_lifts", "travel_distance_mm",
            "draw_distance_mm", "estimated_time_s", "route_improvement_pct", "gcode_size_bytes")},
        confidence=confidence,
    )
