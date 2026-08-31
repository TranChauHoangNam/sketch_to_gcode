from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .analyzer import analyze_report
from .llm_analyzer import ReportLLM, analyze_with_llm
from .metrics import build_report_data
from .models import ReportData
from .rules import RuleConfig


@dataclass
class ReportConfig:
    output_dir: str | None = None
    output_format: str = "pdf"
    write_pdf: bool = True
    llm: ReportLLM | None = None
    rule_config: RuleConfig = field(default_factory=RuleConfig)


class ReportGenerator:
    def __init__(self, config: ReportConfig | None = None):
        self.config = config or ReportConfig()

    def generate(self, report_data: ReportData, *, stem: str = "pipeline_report") -> dict[str, Any]:
        rule_analysis = analyze_report(report_data, self.config.rule_config)
        analysis = analyze_with_llm(report_data, self.config.llm, rule_analysis)
        report_data.quality_score = {
            "overall_score": analysis.get("overall_score", "Not available"),
            "quality_level": analysis.get("quality_level", "Not available"),
            "confidence": analysis.get("confidence", "Not available"),
        }
        payload = {"report_data": report_data.to_dict(), "analysis": analysis}
        output = {"pdf": None, "analysis": analysis}
        if self.config.output_dir:
            directory = Path(self.config.output_dir)
            directory.mkdir(parents=True, exist_ok=True)
            paths = {"pdf": self._write_pdf(directory / f"{stem}.pdf", report_data, analysis)}
            output["paths"] = paths
        return output

    @staticmethod
    def _write_pdf(path: Path, data: ReportData, analysis: dict[str, Any]) -> str:
        try:
            from reportlab.lib import colors
            from reportlab.lib.enums import TA_CENTER
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import mm
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

            styles = getSampleStyleSheet()
            styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], alignment=TA_CENTER,
                                      fontSize=18, leading=22, textColor=colors.HexColor("#16324F"), spaceAfter=8))
            styles.add(ParagraphStyle(name="Section", parent=styles["Heading2"], fontSize=12, leading=15,
                                      textColor=colors.HexColor("#16324F"), spaceBefore=10, spaceAfter=5))
            styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8.5, leading=11))

            def safe(value: Any) -> str:
                return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

            def section(title: str, values: dict[str, Any]) -> list[Any]:
                rows = [[Paragraph("Metric", styles["Small"]), Paragraph("Value", styles["Small"])]]
                for key, value in values.items():
                    rows.append([Paragraph(safe(key), styles["Small"]), Paragraph(safe(value), styles["Small"])])
                table = Table(rows, colWidths=[68 * mm, 112 * mm], repeatRows=1)
                table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCEAF5")),
                    ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#B8C7D1")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]))
                return [Paragraph(safe(title), styles["Section"]), table]

            story = [Paragraph("Sketch-to-G-Code Pipeline Report", styles["ReportTitle"]),
                     Paragraph("Technical run report generated from pipeline metrics", styles["Small"]), Spacer(1, 8)]
            story += section("1. Executive Summary", {"Summary": analysis.get("summary", "Insufficient data"),
                                                        "Overall score": analysis.get("overall_score", "Not available"),
                                                        "Quality level": analysis.get("quality_level", "Not available"),
                                                        "Confidence": analysis.get("confidence", "Not available")})
            for title, values in (("2. Input Information", data.input), ("3. AI Processing", data.ai_processing),
                                  ("4. Image Processing", data.image_processing), ("5. Vectorization", data.vectorization),
                                  ("6. Hatching", data.hatching), ("7. Path Optimization", data.optimization),
                                  ("8. G-code Statistics", data.gcode), ("9. Validation", data.validation),
                                  ("10. Quality Score", data.quality_score)):
                story += section(title, values or {"status": "Not available"})
            story += [Paragraph("11. Problems Detected", styles["Section"])]
            problems = analysis.get("problems", []) or ["None reported"]
            story += [Paragraph("<br/>".join(f"- {safe(x)}" for x in problems), styles["Small"])]
            story += [Paragraph("12. Recommendations", styles["Section"])]
            recommendations = analysis.get("recommendations", []) or ["None reported"]
            story += [Paragraph("<br/>".join(f"- {safe(x)}" for x in recommendations), styles["Small"])]
            story += [Paragraph("13. Raw Metrics", styles["Section"]),
                      Paragraph(safe(json.dumps(data.raw_metrics, indent=2, ensure_ascii=False)), styles["Small"])]
            doc = SimpleDocTemplate(str(path), pagesize=A4, rightMargin=15 * mm, leftMargin=15 * mm,
                                    topMargin=15 * mm, bottomMargin=15 * mm, title="Sketch-to-G-Code Pipeline Report")
            doc.build(story)
            return str(path)
        except Exception:
            return "Not available"

    @staticmethod
    def _markdown(data: ReportData, analysis: dict[str, Any]) -> str:
        d = data.to_dict()
        lines = ["# Sketch-to-G-Code Pipeline Report", "", "## 1. Executive Summary",
                 analysis.get("summary", "Insufficient data"), "",
                 "## 2. Input Information", "```json", json.dumps(d["input"], indent=2, ensure_ascii=False), "```",
                 "", "## 3. AI Processing", "```json", json.dumps(d["ai_processing"], indent=2, ensure_ascii=False), "```",
                 "", "## 4. Image Processing", "```json", json.dumps(d["image_processing"], indent=2, ensure_ascii=False), "```",
                 "", "## 5. Vectorization", "```json", json.dumps(d["vectorization"], indent=2, ensure_ascii=False), "```",
                 "", "## 6. Hatching", "```json", json.dumps(d["hatching"], indent=2, ensure_ascii=False), "```",
                 "", "## 7. Path Optimization", "```json", json.dumps(d["optimization"], indent=2, ensure_ascii=False), "```",
                 "", "## 8. G-code Statistics", "```json", json.dumps(d["gcode"], indent=2, ensure_ascii=False), "```",
                 "", "## 9. Validation", "```json", json.dumps(d["validation"], indent=2, ensure_ascii=False), "```",
                 "", "## 10. Quality Score", f"- Overall score: {analysis.get('overall_score', 'Not available')}",
                 f"- Quality level: {analysis.get('quality_level', 'Not available')}", "",
                 "## 11. Problems Detected"]
        lines += [f"- {x}" for x in analysis.get("problems", [])] or ["- None reported"]
        lines += ["", "## 12. Recommendations"] + ([f"- {x}" for x in analysis.get("recommendations", [])] or ["- None reported"])
        lines += ["", "## 13. Raw Metrics", "```json", json.dumps(d["raw_metrics"], indent=2, ensure_ascii=False), "```", ""]
        return "\n".join(lines)

    @staticmethod
    def _html(data: ReportData, analysis: dict[str, Any]) -> str:
        body = ReportGenerator._markdown(data, analysis).replace("&", "&amp;").replace("<", "&lt;")
        return f"<!doctype html><html><head><meta charset='utf-8'><title>Pipeline Report</title><style>body{{font-family:Arial;max-width:1000px;margin:2rem auto;line-height:1.5}}pre{{background:#f4f4f4;padding:1rem;overflow:auto}}h1{{color:#16324f}}</style></head><body><pre style='white-space:pre-wrap'>{body}</pre></body></html>"

def generate_report(*, pipeline_result: Any, report_config: ReportConfig | None = None,
                    input_path: str | None = None, original_image: Any = None,
                    processing_mode: str | None = None, selected_ai_model: str | None = None,
                    ai_maps: Any = None, hatch_settings: dict[str, Any] | None = None,
                    run_context: dict[str, Any] | None = None, stem: str = "pipeline_report") -> dict[str, Any]:
    config = report_config or ReportConfig()
    data = build_report_data(pipeline_result, input_path=input_path, original_image=original_image,
                             processing_mode=processing_mode, selected_ai_model=selected_ai_model,
                             ai_maps=ai_maps, hatch_settings=hatch_settings, run_context=run_context)
    return ReportGenerator(config).generate(data, stem=stem)


def generate_pdf_report(*, pipeline_result: Any, output_path: str,
                        input_path: str | None = None, original_image: Any = None,
                        processing_mode: str | None = None, selected_ai_model: str | None = None,
                        ai_maps: Any = None, hatch_settings: dict[str, Any] | None = None,
                        run_context: dict[str, Any] | None = None) -> str:
    """Create the single user-facing PDF artifact for one completed run."""
    data = build_report_data(pipeline_result, input_path=input_path, original_image=original_image,
                             processing_mode=processing_mode, selected_ai_model=selected_ai_model,
                             ai_maps=ai_maps, hatch_settings=hatch_settings, run_context=run_context)
    generator = ReportGenerator(ReportConfig(output_dir=str(Path(output_path).parent)))
    result = generator.generate(data, stem=Path(output_path).stem)
    return result.get("paths", {}).get("pdf", "Not available")
