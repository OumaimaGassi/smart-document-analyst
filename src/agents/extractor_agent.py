"""
extractor_agent.py — Text Extractor & Summarizer Agent

Specialist agent responsible for extracting text from documents
and generating concise summaries using LLM.

Team: Benmouma Salma, Gassi Oumaima
"""

from crewai import Agent
from typing import Optional

from src.tools.ocr_extract_tool import OCRExtractTool
from src.tools.llm_summarize_tool import LLMSummarizeTool
from src.utils.logger import AgentLogger


def create_extractor_agent(
    logger: Optional[AgentLogger] = None,
    verbose: bool = True
) -> Agent:
    """
    Create the Text Extractor & Summarizer specialist agent.

    This agent uses OCR to extract text from documents and then
    summarizes the content using an LLM, adapting the summary
    style based on the document type.

    Args:
        logger: AgentLogger instance for action logging.
        verbose: Whether to enable verbose agent output.

    Returns:
        Configured CrewAI Agent instance.
    """
    logger = logger or AgentLogger()

    # Initialize tools
    ocr_tool = OCRExtractTool(logger=logger)
    summarize_tool = LLMSummarizeTool(logger=logger)

    agent = Agent(
        role="Text Extractor and Summarizer",
        goal=(
            "Extract all text content from the document using the best available "
            "method (native PDF extraction or OCR), then produce a concise and "
            "informative summary with key points. Adapt the summary style based "
            "on the document type provided by the Classifier Agent."
        ),
        backstory=(
            "You are a specialist in natural language processing and document analysis. "
            "You excel at extracting text from various document formats, including "
            "scanned documents that require OCR. You are also skilled at creating "
            "clear, concise summaries that capture the essential information. "
            "You always adapt your summarization approach based on the document type — "
            "for invoices you focus on financial details, for resumes on qualifications, "
            "and for scientific papers on findings and methodology."
        ),
        tools=[ocr_tool, summarize_tool],
        verbose=verbose,
        allow_delegation=False,
        max_iter=5,
    )

    return agent
