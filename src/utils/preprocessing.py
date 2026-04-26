"""
preprocessing.py — Document & Image Preprocessing Utilities

Handles document-to-image conversion, image preprocessing for CNN,
and text extraction preparation.

Team: Benmouma Salma, Gassi Oumaima
"""

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

try:
    import torch
    from torchvision import transforms
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

# RVL-CDIP class labels (16 classes)
DOCUMENT_CLASSES = [
    "letter",
    "memo",
    "email",
    "file_folder",
    "form",
    "handwritten",
    "invoice",
    "advertisement",
    "budget",
    "news_article",
    "presentation",
    "scientific_publication",
    "questionnaire",
    "resume",
    "scientific_report",
    "specification"
]

# Subset of classes (recommended for feasibility)
DOCUMENT_CLASSES_SUBSET = [
    "letter",
    "form",
    "invoice",
    "resume",
    "scientific_publication",
    "email",
    "memo",
    "advertisement"
]

# Image size for CNN input
CNN_INPUT_SIZE = 224


class DocumentPreprocessor:
    """
    Preprocessing pipeline for documents.
    
    Handles:
        - PDF to image conversion (first page)
        - Image resizing and normalization for CNN
        - Grayscale to RGB conversion
        - Data augmentation transforms for training
    """

    def __init__(self, input_size: int = CNN_INPUT_SIZE):
        """
        Initialize the preprocessor.

        Args:
            input_size: Target image size for CNN input (square).
        """
        self.input_size = input_size

        if HAS_TORCH:
            # Inference transform (no augmentation)
            self.inference_transform = transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

            # Training transform (with augmentation)
            self.train_transform = transforms.Compose([
                transforms.Resize((input_size + 32, input_size + 32)),
                transforms.RandomCrop(input_size),
                transforms.Grayscale(num_output_channels=3),
                transforms.RandomRotation(5),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.05, 0.05),
                    scale=(0.95, 1.05)
                ),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

            # Validation/test transform (same as inference)
            self.val_transform = self.inference_transform

    def pdf_to_image(
        self,
        pdf_path: str,
        page_num: int = 0,
        dpi: int = 150
    ) -> Optional[Image.Image]:
        """
        Convert a PDF page to a PIL Image.

        Args:
            pdf_path: Path to the PDF file.
            page_num: Page number to convert (0-indexed).
            dpi: Resolution for rendering.

        Returns:
            PIL Image of the page, or None if conversion fails.
        """
        if not HAS_PYMUPDF:
            raise ImportError(
                "PyMuPDF (fitz) is required for PDF conversion. "
                "Install with: pip install pymupdf"
            )

        try:
            doc = fitz.open(pdf_path)
            if page_num >= len(doc):
                page_num = 0  # Fallback to first page

            page = doc[page_num]
            # Render page to pixmap
            zoom = dpi / 72  # Default PDF resolution is 72 DPI
            matrix = fitz.Matrix(zoom, zoom)
            pixmap = page.get_pixmap(matrix=matrix)

            # Convert to PIL Image
            img = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
            doc.close()
            return img

        except Exception as e:
            print(f"Error converting PDF to image: {e}")
            return None

    def load_and_preprocess(
        self,
        file_path: str,
        mode: str = "inference"
    ) -> Optional["torch.Tensor"]:
        """
        Load a document (PDF or image) and preprocess for CNN input.

        Args:
            file_path: Path to document file (PDF, PNG, JPG, TIFF, etc.)
            mode: "inference", "train", or "val" — selects the transform.

        Returns:
            Preprocessed tensor ready for CNN, or None if loading fails.
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is required. Install with: pip install torch torchvision")

        img = self._load_as_image(file_path)
        if img is None:
            return None

        # Select transform
        if mode == "train":
            transform = self.train_transform
        elif mode == "val":
            transform = self.val_transform
        else:
            transform = self.inference_transform

        try:
            tensor = transform(img)
            return tensor
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None

    def _load_as_image(self, file_path: str) -> Optional[Image.Image]:
        """
        Load a file as a PIL Image, handling PDFs and image files.

        Args:
            file_path: Path to the file.

        Returns:
            PIL Image, or None if loading fails.
        """
        path = Path(file_path)
        if not path.exists():
            print(f"File not found: {file_path}")
            return None

        ext = path.suffix.lower()

        if ext == ".pdf":
            return self.pdf_to_image(file_path)
        elif ext in (".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".gif"):
            try:
                return Image.open(file_path).convert("RGB")
            except Exception as e:
                print(f"Error loading image: {e}")
                return None
        else:
            print(f"Unsupported file format: {ext}")
            return None

    @staticmethod
    def get_supported_extensions() -> list[str]:
        """Return list of supported file extensions."""
        return [".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".gif"]

    @staticmethod
    def get_class_labels(subset: bool = True) -> list[str]:
        """
        Get the document class labels.

        Args:
            subset: If True, return the recommended subset of classes.

        Returns:
            List of class label strings.
        """
        return DOCUMENT_CLASSES_SUBSET if subset else DOCUMENT_CLASSES
