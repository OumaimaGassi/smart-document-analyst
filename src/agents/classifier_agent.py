"""
classifier_agent.py — Document Classifier Agent

Specialist agent responsible for classifying documents using the CNN model.

Team: Benmouma Salma, Gassi Oumaima
"""

from crewai import Agent
from typing import Optional

from src.tools.cnn_classify_tool import CNNClassifyTool
from src.utils.logger import AgentLogger


def create_classifier_agent(
    model_path: str = "model/document_classifier.pt",
    logger: Optional[AgentLogger] = None,
    verbose: bool = True
) -> Agent:
    """
    Create the Document Classifier specialist agent.

    This agent uses the trained CNN model to classify documents into
    predefined categories (invoice, letter, resume, form, etc.).

    Args:
        model_path: Path to the trained CNN model weights.
        logger: AgentLogger instance for action logging.
        verbose: Whether to enable verbose agent output.

    Returns:
        Configured CrewAI Agent instance.
    """
    logger = logger or AgentLogger()

    # Initialize the CNN classification tool
    cnn_tool = CNNClassifyTool(model_path=model_path, logger=logger)

    agent = Agent(
        role="Document Classifier",
        goal=(
            "Accurately classify the type of a given document using the CNN "
            "deep learning model. Determine whether the document is an invoice, "
            "letter, resume, form, scientific publication, email, memo, or advertisement."
        ),
        backstory=(
            "You are an expert document analyst with deep expertise in computer vision "
            "and document image analysis. You have been trained on thousands of documents "
            "and can quickly identify document types with high accuracy using your CNN "
            "classification model. Your classifications are critical as they determine "
            "how downstream agents process the document."
        ),
        tools=[cnn_tool],
        verbose=verbose,
        allow_delegation=False,
        max_iter=3,
    )

    return agent
