"""
ocr_extract_tool.py — OCR Text Extraction Tool

CrewAI tool that extracts text from documents using PyMuPDF (native PDFs)
with Tesseract OCR fallback (scanned documents).

Team: Benmouma Salma, Gassi Oumaima
"""

import time
from typing import Type, Optional
from pathlib import Path

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from src.utils.logger import AgentLogger

try:
    import fitz
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

try:
    import pytesseract
    from PIL import Image
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False


class OCRExtractInput(BaseModel):
    """Input schema for the OCR extraction tool."""
    file_path: str = Field(..., description="Path to document file (PDF, PNG, JPG, TIFF)")


class OCRExtractTool(BaseTool):
    """Extract text from a document using PyMuPDF or Tesseract OCR."""

    name: str = "ocr_extract_tool"
    description: str = (
        "Extract text content from a document file (PDF, PNG, JPG, TIFF). "
        "Uses native PDF text extraction when possible, with OCR fallback."
    )
    args_schema: Type[BaseModel] = OCRExtractInput
    logger: Optional[AgentLogger] = None
    min_text_threshold: int = 50

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, logger: Optional[AgentLogger] = None, **kwargs):
        super().__init__(**kwargs)
        self.logger = logger or AgentLogger()

    def _run(self, file_path: str) -> str:
        import json
        start_time = time.time()
        path = Path(file_path)

        if not path.exists():
            return json.dumps({"error": f"File not found: {file_path}"})

        ext = path.suffix.lower()
        try:
            if ext == ".pdf":
                result = self._extract_from_pdf(file_path)
            elif ext in (".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"):
                result = self._extract_from_image(file_path)
            else:
                return json.dumps({"error": f"Unsupported format: {ext}"})

            duration_ms = (time.time() - start_time) * 1000
            self.logger.log_action(
                agent="ExtractorAgent", action="ocr_extract_tool",
                input_data={"file_path": file_path},
                output_data={"text_length": len(result["text"]), "method": result["method"]},
                status="success", duration_ms=duration_ms
            )
            return json.dumps(result, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.log_action(
                agent="ExtractorAgent", action="ocr_extract_tool",
                input_data={"file_path": file_path}, status="error", error=str(e)
            )
            return json.dumps({"error": str(e)})

    def _extract_from_pdf(self, file_path: str) -> dict:
        if not HAS_PYMUPDF:
            raise ImportError("PyMuPDF required: pip install pymupdf")
        doc = fitz.open(file_path)
        num_pages = len(doc)
        text_parts = [page.get_text() for page in doc]
        native_text = "\n".join(text_parts).strip()
        doc.close()

        if len(native_text) >= self.min_text_threshold:
            return {"text": native_text, "num_pages": num_pages, "method": "pymupdf_native", "text_length": len(native_text)}

        if HAS_TESSERACT:
            ocr_text = self._ocr_pdf_pages(file_path)
            if ocr_text:
                return {"text": ocr_text, "num_pages": num_pages, "method": "tesseract_ocr", "text_length": len(ocr_text)}

        return {"text": native_text or "[No text extracted]", "num_pages": num_pages, "method": "pymupdf_native (limited)", "text_length": len(native_text)}

    def _extract_from_image(self, file_path: str) -> dict:
        if not HAS_TESSERACT:
            raise ImportError("pytesseract required: pip install pytesseract")
        img = Image.open(file_path)
        text = pytesseract.image_to_string(img).strip()
        return {"text": text or "[No text detected]", "num_pages": 1, "method": "tesseract_ocr", "text_length": len(text)}

    def _ocr_pdf_pages(self, file_path: str) -> str:
        if not HAS_PYMUPDF or not HAS_TESSERACT:
            return ""
        doc = fitz.open(file_path)
        ocr_texts = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            zoom = 200 / 72
            pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            img = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
            text = pytesseract.image_to_string(img).strip()
            if text:
                ocr_texts.append(f"--- Page {page_num + 1} ---\n{text}")
        doc.close()
        return "\n\n".join(ocr_texts)
