"""
test_tools.py — Unit Tests for Agent Tools

Tests each tool independently with mock data to verify:
- Correct input/output schemas
- Error handling for invalid inputs
- Graceful failure modes

Team: Benmouma Salma, Gassi Oumaima
"""

import json
import os
import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestCNNClassifyTool:
    """Tests for the CNN classification tool."""

    def test_file_not_found(self):
        """Tool should return error JSON when file doesn't exist."""
        from src.tools.cnn_classify_tool import CNNClassifyTool
        tool = CNNClassifyTool.__new__(CNNClassifyTool)
        # Manually initialize minimal attributes
        tool.model = None
        tool.preprocessor = None
        tool.logger = MagicMock()
        tool.model_path = "fake.pt"
        tool.device = "cpu"
        tool.class_names = ["letter", "invoice"]

        result = json.loads(tool._run("nonexistent_file.pdf"))
        assert "error" in result
        assert "not found" in result["error"].lower() or "not loaded" in result["error"].lower()

    def test_model_not_loaded(self):
        """Tool should return error when model is not loaded."""
        from src.tools.cnn_classify_tool import CNNClassifyTool
        tool = CNNClassifyTool.__new__(CNNClassifyTool)
        tool.model = None
        tool.preprocessor = MagicMock()
        tool.logger = MagicMock()
        tool.model_path = "fake.pt"
        tool.device = "cpu"
        tool.class_names = ["letter"]

        # Create a temp file to pass the file check
        temp_file = Path("test_temp_doc.txt")
        temp_file.write_text("test")
        try:
            result = json.loads(tool._run(str(temp_file)))
            assert "error" in result
        finally:
            temp_file.unlink()


class TestOCRExtractTool:
    """Tests for the OCR extraction tool."""

    def test_file_not_found(self):
        """Tool should return error when file doesn't exist."""
        from src.tools.ocr_extract_tool import OCRExtractTool
        tool = OCRExtractTool(logger=MagicMock())
        result = json.loads(tool._run("nonexistent.pdf"))
        assert "error" in result

    def test_unsupported_format(self):
        """Tool should reject unsupported file formats."""
        from src.tools.ocr_extract_tool import OCRExtractTool
        tool = OCRExtractTool(logger=MagicMock())
        # Create a temp file with unsupported extension
        temp_file = Path("test_temp.xyz")
        temp_file.write_text("test")
        try:
            result = json.loads(tool._run(str(temp_file)))
            assert "error" in result
            assert "unsupported" in result["error"].lower()
        finally:
            temp_file.unlink()


class TestLLMSummarizeTool:
    """Tests for the LLM summarization tool."""

    def test_short_text_rejection(self):
        """Tool should reject text that's too short to summarize."""
        from src.tools.llm_summarize_tool import LLMSummarizeTool
        tool = LLMSummarizeTool(logger=MagicMock())
        result = json.loads(tool._run("short", "unknown", 300))
        assert "error" in result

    def test_fallback_summary(self):
        """Fallback summary should work without LLM."""
        from src.tools.llm_summarize_tool import LLMSummarizeTool
        tool = LLMSummarizeTool(logger=MagicMock())
        long_text = "This is a test document with enough content. " * 20
        tool._gemini_model = None  # Force fallback
        result = json.loads(tool._run(long_text, "letter", 50))
        assert "summary" in result
        assert result["method"] == "extractive_fallback"


class TestReportBuilderTool:
    """Tests for the report builder tool."""

    def test_report_generation(self):
        """Report builder should create a valid Markdown report."""
        from src.tools.report_builder_tool import ReportBuilderTool

        output_dir = "test_outputs"
        tool = ReportBuilderTool(output_dir=output_dir, logger=MagicMock())

        classification = json.dumps({
            "class": "invoice",
            "confidence": 0.95,
            "top_3": [
                {"class": "invoice", "confidence": 0.95},
                {"class": "form", "confidence": 0.03},
                {"class": "letter", "confidence": 0.02}
            ]
        })
        extraction = json.dumps({
            "text": "Sample extracted text from document.",
            "num_pages": 1,
            "method": "pymupdf_native",
            "text_length": 38
        })
        summary = json.dumps({
            "summary": "This is a test invoice summary.",
            "key_points": ["Point 1", "Point 2"],
            "word_count": 6,
            "method": "gemini"
        })

        result = json.loads(tool._run(classification, extraction, summary, "test.pdf"))

        assert "report_path" in result
        assert Path(result["report_path"]).exists()

        # Cleanup
        import shutil
        if Path(output_dir).exists():
            shutil.rmtree(output_dir)

    def test_invalid_json_input(self):
        """Tool should handle invalid JSON gracefully."""
        from src.tools.report_builder_tool import ReportBuilderTool
        tool = ReportBuilderTool(output_dir="test_outputs", logger=MagicMock())
        result = json.loads(tool._run("not json", "{}", "{}", "test.pdf"))
        assert "error" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
