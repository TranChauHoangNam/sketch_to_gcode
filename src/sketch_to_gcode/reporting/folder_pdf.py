from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any


def create_folder_analysis_pdf(root: str | Path, output_path: str | Path) -> str:
    """Create a Vietnamese explanatory PDF for the project folder."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    root = Path(root).resolve()
    output = Path(output_path).resolve()
    excluded = {".git", ".venv", "__pycache__", ".codex", ".agents"}
    files, directories = [], set()
    for item in root.rglob("*"):
        relative = item.relative_to(root)
        if set(relative.parts) & excluded:
            continue
        if item.is_dir():
            directories.add(str(relative))
        elif item.is_file():
            files.append((str(relative), item.stat().st_size))
    files.sort()

    font_path = Path("C:/Windows/Fonts/arial.ttf")
    bold_path = Path("C:/Windows/Fonts/arialbd.ttf")
    if font_path.exists():
        pdfmetrics.registerFont(TTFont("ReportArial", str(font_path)))
        pdfmetrics.registerFont(TTFont("ReportArial-Bold", str(bold_path)))
        font, bold = "ReportArial", "ReportArial-Bold"
    else:
        font, bold = "Helvetica", "Helvetica-Bold"

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="VNTitle", fontName=bold, fontSize=19, leading=23,
                              alignment=TA_CENTER, textColor=colors.HexColor("#16324F"), spaceAfter=8))
    styles.add(ParagraphStyle(name="VNSection", fontName=bold, fontSize=13, leading=16,
                              textColor=colors.HexColor("#16324F"), spaceBefore=11, spaceAfter=5))
    styles.add(ParagraphStyle(name="VNBody", fontName=font, fontSize=9.5, leading=13, spaceAfter=5))
    styles.add(ParagraphStyle(name="VNSmall", fontName=font, fontSize=8, leading=10))

    def esc(value: Any) -> str:
        return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def table(rows, widths):
        converted = [[Paragraph(esc(cell), styles["VNSmall"]) for cell in row] for row in rows]
        result = Table(converted, colWidths=widths, repeatRows=1)
        result.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCEAF5")),
            ("GRID", (0, 0), (-1, -1), .35, colors.HexColor("#AAB8C2")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        return result

    story = [Paragraph("BÁO CÁO PHÂN TÍCH THƯ MỤC PROJECT", styles["VNTitle"]),
             Paragraph("Sketch-to-G-Code 2.0", styles["VNSection"]),
             Paragraph(f"Thời gian tạo: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}", styles["VNSmall"]), Spacer(1, 5)]
    story += [Paragraph("1. Phạm vi phân tích", styles["VNSection"]),
              Paragraph(f"Thư mục gốc được phân tích: {esc(root)}", styles["VNBody"]),
              Paragraph("Báo cáo tập trung vào mã nguồn và các module của project. Các thư mục môi trường hoặc cache gồm .git, .venv, __pycache__, .codex và .agents được bỏ qua để không làm nhiễu kết quả.", styles["VNBody"]),
              Paragraph(f"Kết quả: {len(files)} file và {len(directories)} thư mục được đưa vào phân tích.", styles["VNBody"])]

    story += [Paragraph("2. Project này dùng để làm gì?", styles["VNSection"]),
              Paragraph("Project nhận một ảnh hoặc sketch, xử lý ảnh để tách nền và làm nổi bật nét vẽ, chuyển các nét đó thành các đường vector, sau đó sinh mã G-code để pen plotter hoặc CNC có thể vẽ lại hình.", styles["VNBody"]),
              Paragraph("Nói ngắn gọn: Ảnh đầu vào → xử lý ảnh/AI → line-art và hatch → vector path → tối ưu đường chạy → G-code → kiểm tra và export.", styles["VNBody"])]

    story += [Paragraph("3. Cấu trúc thư mục", styles["VNSection"]),
              table([["Thư mục", "Giải thích"],
                     ["(root)", "Chứa hai module lõi của ứng dụng và các package bổ sung."],
                     ["reporting", "Package tạo dữ liệu, phân tích chất lượng và sinh báo cáo PDF."],
                     ["tests", "Unit tests cho reporting pipeline."]], [35*mm, 145*mm])]

    story += [Paragraph("4. Giải thích từng file", styles["VNSection"]),
              table([["File", "Vai trò trong hệ thống"],
                     ["sketch_to_gcode.py", "Module chính và hiện vẫn là monolith. Chứa giao diện Tkinter, đọc ảnh, rembg/AI maps, line-art, vectorization, hatching integration, path ordering, stitching, validation, sinh G-code và export."],
                     ["hatching.py", "Sinh hatch dựa trên grayscale/darkness. Vùng tối được hatch dày hơn, vùng sáng thưa hơn; có clipping theo mask và nối các hatch segment."],
                     ["reporting/models.py", "Định nghĩa ReportData và AnalysisResult."],
                     ["reporting/metrics.py", "Chuyển GCodePlan và context của lần chạy thành ReportData. Metric thiếu được ghi Not available."],
                     ["reporting/rules.py", "Rule-based analyzer và các threshold đánh giá pen lift, travel, validation, kích thước G-code và thời gian."],
                     ["reporting/analyzer.py", "Tổng hợp rule analysis thành quality score, quality level và confidence."],
                     ["reporting/llm_analyzer.py", "Định nghĩa interface ReportLLM và cơ chế fallback khi LLM không sẵn sàng hoặc API lỗi."],
                     ["reporting/generator.py", "Sinh PDF báo cáo kỹ thuật từ ReportData. Có API generate_pdf_report."],
                     ["reporting/folder_pdf.py", "Tạo báo cáo PDF giải thích cấu trúc thư mục project bằng tiếng Việt."],
                     ["tests/test_reporting.py", "Kiểm tra metric mapping, rule detection, PDF generation, schema và LLM fallback."]], [48*mm, 132*mm])]

    story += [Paragraph("5. Pipeline runtime hiện tại", styles["VNSection"]),
              Paragraph("1) Người dùng chọn ảnh trong giao diện Tkinter. 2) Ảnh được chuẩn hóa và tùy chọn tách nền bằng rembg. 3) AIMaps gồm foreground, saliency và hatch_weight được tạo nếu AI hoạt động. 4) Ảnh được chuyển thành line-art; hatch được sinh riêng. 5) Các nét được chuyển thành CandidatePath. 6) build_gcode_plan lọc micro-stroke, deduplicate, simplify, clip, smooth, order, stitch, bridge và compress. 7) paths_to_gcode sinh lệnh G-code và chạy validation/dry-run. 8) Khi export, G-code và PDF report được ghi ra cùng thư mục.", styles["VNBody"])]

    story += [Paragraph("6. Điểm mạnh hiện tại", styles["VNSection"]),
              Paragraph("Project đã có pipeline xử lý tương đối đầy đủ: hỗ trợ nhiều model rembg, có fallback classical, hatch theo darkness, clipping foreground, micro-stroke filtering, path stitching, route optimization, compression, dry-run validation và nhiều G-code metrics.", styles["VNBody"]),
              Paragraph("Package reporting mới tận dụng GCodePlan thay vì viết lại thuật toán xử lý ảnh hoặc G-code.", styles["VNBody"])]

    story += [Paragraph("7. Điểm cần lưu ý", styles["VNSection"]),
              Paragraph("sketch_to_gcode.py hiện rất lớn và gom nhiều trách nhiệm trong một file. Một số thông tin như foreground quality có ground truth, thời gian draw/travel tách riêng, số contour thuần túy và phân loại validation error vẫn chưa được pipeline gốc lưu độc lập; vì vậy báo cáo phải ghi Not available thay vì tự suy đoán.", styles["VNBody"]),
              Paragraph("Báo cáo PDF được tạo sau khi người dùng bấm Export G-Code. Nếu chưa export thì chưa có file report của một lần chạy thực tế.", styles["VNBody"])]

    story += [Paragraph("8. Tổng kết", styles["VNSection"]),
              Paragraph("Đây là một ứng dụng desktop chuyển ảnh thành đường chạy cho máy vẽ. Hai file lõi là sketch_to_gcode.py và hatching.py; reporting/ là lớp bổ sung để phân tích chất lượng sau mỗi lần chạy, còn tests/ dùng để kiểm tra tính ổn định của lớp reporting.", styles["VNBody"]),
              Paragraph("File PDF này là báo cáo phân tích cấu trúc project, không phải báo cáo metrics của một ảnh cụ thể.", styles["VNBody"])]

    output.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(output), pagesize=A4, rightMargin=15*mm, leftMargin=15*mm,
                            topMargin=15*mm, bottomMargin=15*mm, title="Bao cao phan tich thu muc project")
    doc.build(story)
    return str(output)
