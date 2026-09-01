from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .models import NOT_AVAILABLE, ReportData


def _get(obj: Any, name: str, default: Any = NOT_AVAILABLE) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _finite(value: Any) -> Any:
    return value if value is not None else NOT_AVAILABLE


def _map_stats(ai_maps: Any) -> dict[str, Any]:
    if ai_maps is None:
        return {"foreground": NOT_AVAILABLE, "saliency": NOT_AVAILABLE, "hatch_weight": NOT_AVAILABLE}
    result = {}
    for name in ("foreground", "saliency", "hatch_weight"):
        arr = getattr(ai_maps, name, None)
        if arr is None:
            result[name] = NOT_AVAILABLE
            continue
        try:
            import numpy as np
            values = np.asarray(arr, dtype=np.float32)
            result[name] = {
                "mean": float(values.mean()),
                "min": float(values.min()),
                "max": float(values.max()),
                "p50": float(np.percentile(values, 50)),
                "p90": float(np.percentile(values, 90)),
            }
        except Exception:
            result[name] = NOT_AVAILABLE
    return result


def build_report_data(pipeline_result: Any, *, input_path: str | None = None,
                      original_image: Any = None, processing_mode: str | None = None,
                      selected_ai_model: str | None = None, ai_maps: Any = None,
                      hatch_settings: dict[str, Any] | None = None,
                      run_context: dict[str, Any] | None = None) -> ReportData:
    plan = pipeline_result
    paths = _get(plan, "paths", [])
    validation_errors = list(_get(plan, "validation_errors", []) or [])
    width = getattr(original_image, "width", NOT_AVAILABLE) if original_image is not None else NOT_AVAILABLE
    height = getattr(original_image, "height", NOT_AVAILABLE) if original_image is not None else NOT_AVAILABLE
    filename = os.path.basename(input_path) if input_path else NOT_AVAILABLE
    image_format = NOT_AVAILABLE
    if input_path:
        try:
            from PIL import Image
            with Image.open(input_path) as img:
                image_format = img.format or NOT_AVAILABLE
                if width == NOT_AVAILABLE:
                    width, height = img.size
        except Exception:
            pass

    raw = {}
    for name in ("target_segments", "actual_segments", "stroke_count", "pen_lifts",
                 "command_count", "raw_segment_count", "stitched_count",
                 "travel_distance_mm", "draw_distance_mm", "estimated_time_s",
                 "command_count_before_compression", "compressed_segment_count",
                 "compression_removed_count", "route_improvement_mm",
                 "route_improvement_pct", "clipped_path_count",
                 "deduplicated_path_count", "island_count", "gcode_postprocess_removed_count",
                 "resampled_point_count", "stroke_direction_flip_count",
                 "dry_run_warning_count", "pen_lift_bridge_count", "planning_time_s",
                 "gcode_size_bytes", "ai_backend", "ai_elapsed_s", "hatch_candidate_count"):
        raw[name] = _finite(_get(plan, name))
    total_points = sum(len(p) for p in paths) if paths else 0
    raw["total_points"] = total_points
    raw["total_paths"] = len(paths)
    context = run_context or {}
    return ReportData(
        input={"filename": filename, "image_width": width, "image_height": height,
               "image_format": image_format, "processing_mode": processing_mode or NOT_AVAILABLE,
               "selected_AI_model": selected_ai_model or NOT_AVAILABLE},
        ai_processing={"backend": raw["ai_backend"], "inference_time_s": raw["ai_elapsed_s"],
                       "map_statistics": _map_stats(ai_maps)},
        image_processing=context.get("image_processing", {}),
        vectorization={"number_of_contours": context.get("number_of_contours", NOT_AVAILABLE),
                       "number_of_hatch_paths": raw["hatch_candidate_count"],
                       "total_paths": total_points and len(paths) or 0,
                       "total_points": total_points,
                       "removed_micro_strokes": context.get("removed_micro_strokes", NOT_AVAILABLE),
                       "stitched_paths": raw["stitched_count"], "clipped_paths": raw["clipped_path_count"],
                       "duplicated_paths": raw["deduplicated_path_count"]},
        hatching={"candidate_count": raw["hatch_candidate_count"], "settings": hatch_settings or NOT_AVAILABLE},
        gcode={"total_commands": raw["command_count"], "drawing_distance_mm": raw["draw_distance_mm"],
               "travel_distance_mm": raw["travel_distance_mm"], "pen_lifts": raw["pen_lifts"],
               "estimated_total_time_s": raw["estimated_time_s"],
               "estimated_drawing_time_s": context.get("estimated_drawing_time_s", NOT_AVAILABLE),
               "estimated_travel_time_s": context.get("estimated_travel_time_s", NOT_AVAILABLE),
               "file_size_bytes": raw["gcode_size_bytes"]},
        optimization={"route_distance_before_mm": context.get("route_distance_before_mm", NOT_AVAILABLE),
                      "route_distance_after_mm": context.get("route_distance_after_mm", NOT_AVAILABLE),
                      "improvement_mm": raw["route_improvement_mm"], "improvement_pct": raw["route_improvement_pct"],
                      "removed_commands": raw["gcode_postprocess_removed_count"],
                      "compressed_segments": raw["compressed_segment_count"]},
        validation={"status": "FAIL" if validation_errors else "PASS", "errors": validation_errors,
                    "error_count": len(validation_errors), "warnings": context.get("warnings", []),
                    "dry_run_warnings": raw["dry_run_warning_count"],
                    "bounds_violations": context.get("bounds_violations", NOT_AVAILABLE),
                    "invalid_coordinates": context.get("invalid_coordinates", NOT_AVAILABLE),
                    "suspicious_paths": context.get("suspicious_paths", NOT_AVAILABLE)},
        raw_metrics={**raw, **context.get("raw_metrics", {})},
    )
