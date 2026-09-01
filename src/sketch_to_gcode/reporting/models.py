from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


NOT_AVAILABLE = "Not available"


@dataclass
class ReportData:
    input: dict[str, Any] = field(default_factory=dict)
    ai_processing: dict[str, Any] = field(default_factory=dict)
    image_processing: dict[str, Any] = field(default_factory=dict)
    vectorization: dict[str, Any] = field(default_factory=dict)
    hatching: dict[str, Any] = field(default_factory=dict)
    gcode: dict[str, Any] = field(default_factory=dict)
    optimization: dict[str, Any] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)
    quality_score: dict[str, Any] = field(default_factory=dict)
    raw_metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "input": self.input,
            "ai_processing": self.ai_processing,
            "image_processing": self.image_processing,
            "vectorization": self.vectorization,
            "hatching": self.hatching,
            "gcode": self.gcode,
            "optimization": self.optimization,
            "validation": self.validation,
            "quality_score": self.quality_score,
            "raw_metrics": self.raw_metrics,
        }


@dataclass
class AnalysisResult:
    summary: str = "Insufficient data"
    overall_score: int | str = NOT_AVAILABLE
    quality_level: str = NOT_AVAILABLE
    performance: dict[str, Any] = field(default_factory=dict)
    path_quality: dict[str, Any] = field(default_factory=dict)
    gcode_quality: dict[str, Any] = field(default_factory=dict)
    optimization: dict[str, Any] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    important_metrics: dict[str, Any] = field(default_factory=dict)
    confidence: int | str = NOT_AVAILABLE

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()
