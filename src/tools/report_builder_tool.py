"""
report_builder_tool.py — Structured Report Generation Tool

CrewAI tool that generates a structured analysis report (Markdown and/or PDF)
from the combined classification, extraction, and summarization results.

Team: Benmouma Salma, Gassi Oumaima
"""

import os
import time
import json
from datetime import datetime
from typing import Type, Optional
from pathlib import Path

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from src.utils.logger import AgentLogger


class ReportBuilderInput(BaseModel):
    """Input schema for the report builder tool."""
    classification: str = Field(..., description="JSON string with classification results")
    extraction: str = Field(..., description="JSON string with extracted text")
    summary: str = Field(..., description="JSON string with summarization results")
    file_path: str = Field(default="", description="Original document file path")


class ReportBuilderTool(BaseTool):
    """Generate a structured Markdown/PDF report from all analysis results."""

    name: str = "report_builder_tool"
    description: str = (
        "Generate a professional structured report combining document classification, "
        "extracted text, and summary into a Markdown file. Input: classification results, "
        "extracted text, and summary as JSON strings."
    )
    args_schema: Type[BaseModel] = ReportBuilderInput
    logger: Optional[AgentLogger] = None
    output_dir: str = "outputs/reports"

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, output_dir: str = "outputs/reports", logger: Optional[AgentLogger] = None, **kwargs):
        super().__init__(**kwargs)
        self.output_dir = output_dir
        self.logger = logger or AgentLogger()
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def _run(self, classification: str, extraction: str, summary: str, file_path: str = "") -> str:
        start_time = time.time()

        try:
            cls_data = json.loads(classification) if isinstance(classification, str) else classification
            ext_data = json.loads(extraction) if isinstance(extraction, str) else extraction
            sum_data = json.loads(summary) if isinstance(summary, str) else summary
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"Invalid JSON input: {e}"})

        try:
            # Generate Markdown report
            report_md = self._build_markdown_report(cls_data, ext_data, sum_data, file_path)

            # Save report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            doc_name = Path(file_path).stem if file_path else "document"
            report_filename = f"report_{doc_name}_{timestamp}.md"
            report_path = os.path.join(self.output_dir, report_filename)

            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report_md)

            # Try PDF generation
            pdf_path = None
            try:
                pdf_path = self._generate_pdf(report_md, report_path.replace(".md", ".pdf"))
            except Exception:
                pass  # PDF generation is optional

            duration_ms = (time.time() - start_time) * 1000
            result = {
                "report_path": report_path,
                "pdf_path": pdf_path,
                "format": "markdown" + (" + pdf" if pdf_path else ""),
                "report_length": len(report_md)
            }

            self.logger.log_action(
                agent="ReporterAgent", action="report_builder_tool",
                input_data={"file_path": file_path},
                output_data=result, status="success", duration_ms=duration_ms
            )
            return json.dumps(result, indent=2)

        except Exception as e:
            self.logger.log_action(
                agent="ReporterAgent", action="report_builder_tool",
                input_data={"file_path": file_path}, status="error", error=str(e)
            )
            return json.dumps({"error": str(e)})

    def _build_markdown_report(self, cls_data: dict, ext_data: dict, sum_data: dict, file_path: str) -> str:
        """Build a professional Markdown report."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        doc_class = cls_data.get("class", "Unknown")
        confidence = cls_data.get("confidence", 0)

        # Header
        report = f"""# 📄 Document Analysis Report

**Generated:** {now}
**Analyzed by:** Smart Document Analyst — Multi-Agent AI System
**Team:** Benmouma Salma, Gassi Oumaima

---

## 1. Document Information

| Field | Value |
|-------|-------|
| **File** | `{file_path or 'N/A'}` |
| **Classification** | **{doc_class.upper()}** |
| **Confidence** | {confidence:.1%} |
| **Pages** | {ext_data.get('num_pages', 'N/A')} |
| **Extraction Method** | {ext_data.get('method', 'N/A')} |

"""

        # Top predictions
        top_3 = cls_data.get("top_3", [])
        if top_3:
            report += "### Classification Details\n\n"
            report += "| Rank | Class | Confidence |\n|------|-------|------------|\n"
            for i, pred in enumerate(top_3, 1):
                marker = "→" if i == 1 else " "
                report += f"| {marker} {i} | {pred.get('class', 'N/A')} | {pred.get('confidence', 0):.1%} |\n"
            report += "\n"

        # Summary section
        summary_text = sum_data.get("summary", "No summary available.")
        key_points = sum_data.get("key_points", [])

        report += f"""---

## 2. Document Summary

{summary_text}

"""
        if key_points:
            report += "### Key Points\n\n"
            for point in key_points:
                report += f"- {point}\n"
            report += "\n"

        # Extracted text (truncated)
        full_text = ext_data.get("text", "")
        display_text = full_text[:3000] + "\n\n[... truncated ...]" if len(full_text) > 3000 else full_text

        report += f"""---

## 3. Extracted Text

```
{display_text}
```

---

## 4. Pipeline Metadata

| Metric | Value |
|--------|-------|
| **Summarization Method** | {sum_data.get('method', 'N/A')} |
| **Summary Word Count** | {sum_data.get('word_count', 'N/A')} |
| **Extracted Text Length** | {ext_data.get('text_length', len(full_text))} chars |

---

*Report generated by Smart Document Analyst — UIR S8 Multi-Agent AI Project*
*Benmouma Salma & Gassi Oumaima — 2025–2026*
"""
        return report

    def _generate_pdf(self, markdown_text: str, pdf_path: str) -> Optional[str]:
        """Attempt to generate a PDF from the Markdown report."""
        try:
            from fpdf import FPDF

            pdf = FPDF()
            pdf.add_page()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.set_font("Helvetica", size=10)

            for line in markdown_text.split("\n"):
                line = line.strip()
                if line.startswith("# "):
                    pdf.set_font("Helvetica", "B", 16)
                    pdf.cell(0, 10, line[2:], ln=True)
                    pdf.set_font("Helvetica", size=10)
                elif line.startswith("## "):
                    pdf.set_font("Helvetica", "B", 13)
                    pdf.cell(0, 8, line[3:], ln=True)
                    pdf.set_font("Helvetica", size=10)
                elif line.startswith("### "):
                    pdf.set_font("Helvetica", "B", 11)
                    pdf.cell(0, 7, line[4:], ln=True)
                    pdf.set_font("Helvetica", size=10)
                elif line.startswith("---"):
                    pdf.ln(3)
                elif line:
                    # Handle encoding for PDF
                    safe_line = line.encode("latin-1", errors="replace").decode("latin-1")
                    pdf.multi_cell(0, 5, safe_line)
                else:
                    pdf.ln(3)

            pdf.output(pdf_path)
            return pdf_path
        except Exception:
            return None
