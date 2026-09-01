from __future__ import annotations

import json
from typing import Any

from .models import AnalysisResult, ReportData


REPORT_ANALYST_SYSTEM_PROMPT = """You are a technical report analyst for a sketch-to-G-code pipeline.
Use only facts present in the supplied ReportData. Never invent, repair, reinterpret, or recalculate metrics.
Clearly distinguish fact from inference. If data is missing, write exactly 'Insufficient data'.
Prioritize problems that directly affect machine safety and G-code quality. Give practical recommendations,
but do not propose code changes without evidence in the metrics. Return only valid JSON with this exact shape:
{"summary":"...","overall_score":0,"quality_level":"...","performance":{"score":0,"analysis":"..."},
"path_quality":{"score":0,"analysis":"..."},"gcode_quality":{"score":0,"analysis":"..."},
"optimization":{"score":0,"analysis":"..."},"problems":[],"warnings":[],"recommendations":[],
"important_metrics":{},"confidence":0}. Do not add metrics not present in ReportData.
"""


class ReportLLM:
    def analyze(self, report_data: ReportData) -> dict[str, Any]:
        raise NotImplementedError


class LocalReportLLM(ReportLLM):
    """Provider-neutral fallback. It deliberately performs no invented AI analysis."""
    def analyze(self, report_data: ReportData) -> dict[str, Any]:
        raise RuntimeError("LLM provider is not configured")


def analyze_with_llm(report_data: ReportData, llm: ReportLLM | None,
                     fallback: AnalysisResult) -> dict[str, Any]:
    if llm is None:
        result = fallback.to_dict()
        result["llm_status"] = "LLM analysis unavailable"
        return result
    try:
        result = llm.analyze(report_data)
        json.dumps(result)
        defaults = fallback.to_dict()
        result = {key: result.get(key, defaults[key]) for key in defaults}
        result["llm_status"] = "available"
        return result
    except Exception as exc:
        result = fallback.to_dict()
        result["llm_status"] = "LLM analysis unavailable"
        result["llm_error"] = type(exc).__name__
        return result
