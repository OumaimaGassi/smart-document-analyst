"""
llm_summarize_tool.py — LLM Document Summarization Tool

CrewAI tool that summarizes extracted document text using the Gemini API,
adapting the summary style based on the document type.

Team: Benmouma Salma, Gassi Oumaima
"""

import os
import time
import json
from typing import Type, Optional

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from src.utils.logger import AgentLogger

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False


class LLMSummarizeInput(BaseModel):
    """Input schema for the LLM summarization tool."""
    text: str = Field(..., description="The extracted document text to summarize")
    doc_type: str = Field(default="unknown", description="Type of document (invoice, letter, resume, etc.)")
    max_length: int = Field(default=300, description="Maximum summary length in words")


class LLMSummarizeTool(BaseTool):
    """Summarize document text using Gemini LLM, adapting style to document type."""

    name: str = "llm_summarize_tool"
    description: str = (
        "Summarize document text using an LLM (Gemini). Adapts the summary style "
        "based on document type. Returns a concise summary with key points."
    )
    args_schema: Type[BaseModel] = LLMSummarizeInput
    logger: Optional[AgentLogger] = None
    _gemini_model: Optional[object] = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, logger: Optional[AgentLogger] = None, **kwargs):
        super().__init__(**kwargs)
        self.logger = logger or AgentLogger()
        self._setup_gemini()

    def _setup_gemini(self):
        """Configure the Gemini API client."""
        if not HAS_GEMINI:
            return
        api_key = os.getenv("GEMINI_API_KEY", "")
        if api_key:
            genai.configure(api_key=api_key)
            self._gemini_model = genai.GenerativeModel("gemini-1.5-flash")

    def _run(self, text: str, doc_type: str = "unknown", max_length: int = 300) -> str:
        start_time = time.time()

        if not text or len(text.strip()) < 20:
            return json.dumps({"error": "Text too short to summarize"})

        # Truncate very long texts to stay within API limits
        if len(text) > 15000:
            text = text[:15000] + "\n[... text truncated for summarization ...]"

        prompt = self._build_prompt(text, doc_type, max_length)

        try:
            if self._gemini_model:
                summary_text = self._call_gemini(prompt)
            else:
                summary_text = self._fallback_summary(text, max_length)

            # Extract key points
            key_points = self._extract_key_points(summary_text)

            result = {
                "summary": summary_text,
                "key_points": key_points,
                "word_count": len(summary_text.split()),
                "doc_type": doc_type,
                "method": "gemini" if self._gemini_model else "extractive_fallback"
            }

            duration_ms = (time.time() - start_time) * 1000
            self.logger.log_action(
                agent="ExtractorAgent", action="llm_summarize_tool",
                input_data={"text_length": len(text), "doc_type": doc_type},
                output_data={"summary_length": len(summary_text), "method": result["method"]},
                status="success", duration_ms=duration_ms
            )
            return json.dumps(result, indent=2, ensure_ascii=False)

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.logger.log_action(
                agent="ExtractorAgent", action="llm_summarize_tool",
                input_data={"text_length": len(text)}, status="error",
                error=str(e), duration_ms=duration_ms
            )
            # Fallback to extractive summary on error
            fallback = self._fallback_summary(text, max_length)
            return json.dumps({
                "summary": fallback,
                "key_points": [],
                "word_count": len(fallback.split()),
                "doc_type": doc_type,
                "method": "extractive_fallback",
                "warning": f"LLM unavailable, used fallback: {str(e)}"
            }, indent=2)

    def _build_prompt(self, text: str, doc_type: str, max_length: int) -> str:
        """Build a context-aware summarization prompt."""
        type_instructions = {
            "invoice": "Focus on: vendor, amounts, dates, items, payment terms.",
            "letter": "Focus on: sender, recipient, purpose, key requests or information.",
            "resume": "Focus on: candidate name, experience level, key skills, education.",
            "form": "Focus on: form purpose, key fields, required information.",
            "scientific_publication": "Focus on: research question, methods, key findings, conclusions.",
            "email": "Focus on: sender, subject, key action items, deadlines.",
            "memo": "Focus on: purpose, key decisions, action items.",
            "advertisement": "Focus on: product/service, target audience, key selling points.",
        }

        type_hint = type_instructions.get(doc_type, "Provide a comprehensive summary.")

        return f"""You are a document analysis expert. Summarize the following {doc_type} document.

{type_hint}

Provide your response in this exact format:
SUMMARY: [Your concise summary in {max_length} words or less]
KEY POINTS:
- [Point 1]
- [Point 2]
- [Point 3]
- [Point 4]
- [Point 5]

DOCUMENT TEXT:
{text}"""

    def _call_gemini(self, prompt: str, max_retries: int = 3) -> str:
        """Call the Gemini API with retry logic."""
        for attempt in range(max_retries):
            try:
                response = self._gemini_model.generate_content(prompt)
                return response.text
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise e

    def _fallback_summary(self, text: str, max_words: int) -> str:
        """Simple extractive summary when LLM is unavailable."""
        sentences = text.replace("\n", " ").split(".")
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        summary_sentences = sentences[:min(5, len(sentences))]
        summary = ". ".join(summary_sentences) + "."
        words = summary.split()
        if len(words) > max_words:
            summary = " ".join(words[:max_words]) + "..."
        return summary

    def _extract_key_points(self, summary_text: str) -> list:
        """Extract key points from the summary text."""
        points = []
        for line in summary_text.split("\n"):
            line = line.strip()
            if line.startswith("- ") or line.startswith("• "):
                points.append(line[2:].strip())
        return points[:5]
