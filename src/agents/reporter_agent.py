"""
reporter_agent.py — Report Generator Agent

Specialist agent responsible for producing structured analysis reports
from the combined classification, extraction, and summarization results.

Team: Benmouma Salma, Gassi Oumaima
"""

from crewai import Agent
from typing import Optional

from src.tools.report_builder_tool import ReportBuilderTool
from src.utils.logger import AgentLogger


def create_reporter_agent(
    output_dir: str = "outputs/reports",
    logger: Optional[AgentLogger] = None,
    verbose: bool = True
) -> Agent:
    """
    Create the Report Generator specialist agent.

    This agent takes the outputs from the Classifier and Extractor agents
    and produces a professional structured report in Markdown/PDF format.

    Args:
        output_dir: Directory to save generated reports.
        logger: AgentLogger instance for action logging.
        verbose: Whether to enable verbose agent output.

    Returns:
        Configured CrewAI Agent instance.
    """
    logger = logger or AgentLogger()

    # Initialize the report builder tool
    report_tool = ReportBuilderTool(output_dir=output_dir, logger=logger)

    agent = Agent(
        role="Report Generator",
        goal=(
            "Produce a professional, well-structured analysis report that combines "
            "the document classification, extracted text, and summary into a clear "
            "and informative document. The report should be suitable for stakeholders "
            "and include all relevant metadata."
        ),
        backstory=(
            "You are a technical writer with expertise in creating clear, professional "
            "analysis reports. You excel at organizing information from multiple sources "
            "into coherent, well-formatted documents. You always ensure reports include "
            "proper headers, tables, and structured sections for easy reading. "
            "Your reports are used by decision-makers who rely on accurate and "
            "well-presented document analysis results."
        ),
        tools=[report_tool],
        verbose=verbose,
        allow_delegation=False,
        max_iter=3,
    )

    return agent
