"""
cnn_classify_tool.py — CNN Document Classification Tool

CrewAI-compatible tool that uses the trained PyTorch CNN model
to classify document images into predefined categories.

Team: Benmouma Salma, Gassi Oumaima
"""

import os
import time
from typing import Type, Optional
from pathlib import Path

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

import torch

from src.models.document_classifier import DocumentClassifierCNN
from src.utils.preprocessing import DocumentPreprocessor, DOCUMENT_CLASSES_SUBSET
from src.utils.logger import AgentLogger


# ──────────────────────────────────────────────
# Tool Input Schema
# ──────────────────────────────────────────────

class CNNClassifyInput(BaseModel):
    """Input schema for the CNN classification tool."""
    file_path: str = Field(
        ...,
        description="Absolute or relative path to the document file (PDF, PNG, JPG, TIFF, etc.)"
    )


# ──────────────────────────────────────────────
# CNN Classification Tool
# ──────────────────────────────────────────────

class CNNClassifyTool(BaseTool):
    """
    Classify a document image using a trained CNN model.
    
    This tool loads the PyTorch document classifier (fine-tuned ResNet-18),
    preprocesses the input document (converting PDFs to images if needed),
    and returns the predicted class with confidence scores.
    
    Input:  file_path (str) — path to document image or PDF
    Output: JSON string with class, confidence, and top 3 predictions
    """

    name: str = "cnn_classify_tool"
    description: str = (
        "Classify a document into one of the predefined categories "
        "(letter, form, invoice, resume, scientific_publication, email, memo, advertisement) "
        "using a trained CNN deep learning model. "
        "Input: file path to a document (PDF, PNG, JPG). "
        "Output: predicted class, confidence score, and top 3 predictions."
    )
    args_schema: Type[BaseModel] = CNNClassifyInput

    # Internal attributes (not exposed as tool arguments)
    model: Optional[DocumentClassifierCNN] = None
    preprocessor: Optional[DocumentPreprocessor] = None
    logger: Optional[AgentLogger] = None
    model_path: str = "model/document_classifier.pt"
    device: str = "cpu"
    class_names: list = DOCUMENT_CLASSES_SUBSET

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        model_path: str = "model/document_classifier.pt",
        logger: Optional[AgentLogger] = None,
        **kwargs
    ):
        """
        Initialize the CNN classification tool.

        Args:
            model_path: Path to the trained model weights file.
            logger: Optional AgentLogger for logging actions.
        """
        super().__init__(**kwargs)
        self.model_path = model_path
        self.logger = logger or AgentLogger()
        self.preprocessor = DocumentPreprocessor()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()

    def _load_model(self):
        """Load the trained CNN model from disk."""
        try:
            self.model = DocumentClassifierCNN.load_model(
                model_path=self.model_path,
                num_classes=len(self.class_names),
                device=self.device
            )
            self.logger.log_action(
                agent="ClassifierAgent",
                action="load_model",
                input_data={"model_path": self.model_path, "device": self.device},
                output_data={"status": "loaded", "num_classes": len(self.class_names)},
                status="success"
            )
        except FileNotFoundError as e:
            self.logger.log_action(
                agent="ClassifierAgent",
                action="load_model",
                input_data={"model_path": self.model_path},
                status="error",
                error=str(e)
            )
            # Model will be None — tool will return an error when called
            self.model = None

    def _run(self, file_path: str) -> str:
        """
        Run CNN classification on a document.

        Args:
            file_path: Path to the document file.

        Returns:
            JSON string with classification results.
        """
        import json
        start_time = time.time()

        # Validate file exists
        if not Path(file_path).exists():
            error_msg = f"File not found: {file_path}"
            self.logger.log_action(
                agent="ClassifierAgent",
                action="cnn_classify_tool",
                input_data={"file_path": file_path},
                status="error",
                error=error_msg
            )
            return json.dumps({"error": error_msg})

        # Check model is loaded
        if self.model is None:
            error_msg = (
                "CNN model not loaded. Please train the model first using "
                "notebooks/training.ipynb or provide a valid model file."
            )
            self.logger.log_action(
                agent="ClassifierAgent",
                action="cnn_classify_tool",
                input_data={"file_path": file_path},
                status="error",
                error=error_msg
            )
            return json.dumps({"error": error_msg})

        # Preprocess document
        try:
            tensor = self.preprocessor.load_and_preprocess(file_path, mode="inference")
            if tensor is None:
                error_msg = f"Failed to preprocess document: {file_path}"
                self.logger.log_action(
                    agent="ClassifierAgent",
                    action="cnn_classify_tool",
                    input_data={"file_path": file_path},
                    status="error",
                    error=error_msg
                )
                return json.dumps({"error": error_msg})

            # Add batch dimension: (3, 224, 224) → (1, 3, 224, 224)
            tensor = tensor.unsqueeze(0).to(self.device)

        except Exception as e:
            error_msg = f"Preprocessing error: {str(e)}"
            self.logger.log_action(
                agent="ClassifierAgent",
                action="cnn_classify_tool",
                input_data={"file_path": file_path},
                status="error",
                error=error_msg
            )
            return json.dumps({"error": error_msg})

        # Run inference
        try:
            top_3 = self.model.predict_top_k(
                tensor, k=3, class_names=self.class_names
            )

            predicted_class = top_3[0]["class"]
            confidence = top_3[0]["confidence"]

            duration_ms = (time.time() - start_time) * 1000

            result = {
                "class": predicted_class,
                "confidence": confidence,
                "top_3": top_3,
                "file_path": file_path,
                "device": self.device
            }

            self.logger.log_action(
                agent="ClassifierAgent",
                action="cnn_classify_tool",
                input_data={"file_path": file_path},
                output_data=result,
                status="success",
                duration_ms=duration_ms
            )

            return json.dumps(result, indent=2)

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            error_msg = f"Inference error: {str(e)}"
            self.logger.log_action(
                agent="ClassifierAgent",
                action="cnn_classify_tool",
                input_data={"file_path": file_path},
                status="error",
                error=error_msg,
                duration_ms=duration_ms
            )
            return json.dumps({"error": error_msg})
