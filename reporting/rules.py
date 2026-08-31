from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .models import NOT_AVAILABLE, ReportData


@dataclass(frozen=True)
class RuleConfig:
    max_pen_lift_ratio: float = 0.45
    max_travel_draw_ratio: float = 2.5
    good_route_improvement_pct: float = 10.0
    high_micro_stroke_removal_ratio: float = 0.35
    large_gcode_size_bytes: int = 2_000_000
    long_estimated_time_s: float = 3600.0
    max_direction_flip_ratio: float = 0.8


def _num(data: dict[str, Any], key: str) -> float | None:
    value = data.get(key)
    return value if isinstance(value, (int, float)) else None


def analyze_rules(report: ReportData, config: RuleConfig | None = None) -> dict[str, Any]:
    cfg = config or RuleConfig()
    g, v, vec, opt = report.gcode, report.validation, report.vectorization, report.optimization
    problems, warnings, recommendations = [], [], []
    errors = _num(v, "error_count") or 0
    if errors > 0:
        problems.append(f"Critical: validation produced {int(errors)} error(s).")
        recommendations.append("Inspect validation errors before sending G-code to the machine.")

    strokes, lifts = _num(g, "pen_lifts"), _num(vec, "total_paths")
    if strokes is not None and lifts is not None and strokes > 0 and lifts / strokes > cfg.max_pen_lift_ratio:
        warnings.append("Pen-lift ratio is high relative to the number of paths.")
        recommendations.append("Review path stitching and ordering to reduce unnecessary pen lifts.")

    travel, draw = _num(g, "travel_distance_mm"), _num(g, "drawing_distance_mm")
    if travel is not None and draw is not None and draw > 0 and travel / draw > cfg.max_travel_draw_ratio:
        warnings.append("Travel distance is high relative to drawing distance.")
        recommendations.append("Review path ordering, island grouping, and stitching thresholds.")

    improvement = _num(opt, "improvement_pct")
    if improvement is not None and improvement >= cfg.good_route_improvement_pct:
        warnings.append(f"Route optimization saved {improvement:.1f}% of the measured route.")

    removed = _num(report.raw_metrics, "removed_micro_strokes")
    raw_segments = _num(report.raw_metrics, "raw_segment_count")
    if removed is not None and raw_segments and removed / raw_segments > cfg.high_micro_stroke_removal_ratio:
        warnings.append("A large fraction of micro-strokes was removed, indicating possible input noise.")
        recommendations.append("Check source image quality and preprocessing thresholds.")

    size = _num(g, "file_size_bytes")
    if size is not None and size > cfg.large_gcode_size_bytes:
        warnings.append("G-code file size is high and may indicate excessive complexity.")
        recommendations.append("Consider a lower detail budget or stronger collinear compression.")

    duration = _num(g, "estimated_total_time_s")
    if duration is not None and duration > cfg.long_estimated_time_s:
        warnings.append("Estimated machine time is high.")
        recommendations.append("Review detail budget, travel routing, and feed-rate settings.")

    score_parts = []
    if errors == 0 and "error_count" in v:
        score_parts.append(100.0)
    if travel is not None and draw is not None and draw > 0:
        score_parts.append(max(0.0, min(100.0, 100.0 * cfg.max_travel_draw_ratio / max(travel / draw, 1e-6))))
    if lifts is not None and strokes is not None and strokes > 0:
        score_parts.append(max(0.0, min(100.0, 100.0 * cfg.max_pen_lift_ratio / max(lifts / strokes, 1e-6))))
    if improvement is not None:
        score_parts.append(max(0.0, min(100.0, 70.0 + improvement)))
    score = round(sum(score_parts) / len(score_parts)) if score_parts else NOT_AVAILABLE
    return {"score": score, "problems": problems, "warnings": warnings, "recommendations": recommendations}
