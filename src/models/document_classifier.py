"""
document_classifier.py — PyTorch CNN Document Classifier

Fine-tuned ResNet-18 model for classifying document images.
Trained on a subset of the RVL-CDIP dataset.

Team: Benmouma Salma, Gassi Oumaima
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional
from pathlib import Path


class DocumentClassifierCNN(nn.Module):
    """
    Document image classifier based on fine-tuned ResNet-18.
    
    Architecture:
        - ResNet-18 backbone (pretrained on ImageNet)
        - Custom classification head for document types
        - Dropout for regularization
    
    The model accepts 224×224 RGB images (grayscale documents are
    converted to 3-channel in preprocessing).
    """

    def __init__(
        self,
        num_classes: int = 8,
        pretrained: bool = True,
        dropout_rate: float = 0.3
    ):
        """
        Initialize the document classifier.

        Args:
            num_classes: Number of document classes to classify.
            pretrained: Whether to use ImageNet pretrained weights.
            dropout_rate: Dropout rate for the classification head.
        """
        super(DocumentClassifierCNN, self).__init__()

        self.num_classes = num_classes

        # Load pretrained ResNet-18
        if pretrained:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            self.backbone = models.resnet18(weights=weights)
        else:
            self.backbone = models.resnet18(weights=None)

        # Freeze early layers (conv1, bn1, layer1, layer2)
        # Only train layer3, layer4, and the classification head
        for name, param in self.backbone.named_parameters():
            if "layer3" not in name and "layer4" not in name and "fc" not in name:
                param.requires_grad = False

        # Replace the final fully-connected layer with our custom head
        in_features = self.backbone.fc.in_features  # 512 for ResNet-18
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, 3, 224, 224).

        Returns:
            Logits tensor of shape (batch, num_classes).
        """
        return self.backbone(x)

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict class and confidence for input images.

        Args:
            x: Input tensor of shape (batch, 3, 224, 224).

        Returns:
            Tuple of (predicted_classes, confidence_scores).
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted = torch.max(probabilities, dim=1)
        return predicted, confidence

    def predict_top_k(
        self,
        x: torch.Tensor,
        k: int = 3,
        class_names: Optional[list[str]] = None
    ) -> list[dict]:
        """
        Get top-k predictions with class names and confidence scores.

        Args:
            x: Input tensor of shape (1, 3, 224, 224) — single image.
            k: Number of top predictions to return.
            class_names: Optional list of class name strings.

        Returns:
            List of dicts with 'class', 'class_idx', and 'confidence' keys.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.softmax(logits, dim=1)
            top_conf, top_idx = torch.topk(probabilities, k, dim=1)

        results = []
        for i in range(k):
            idx = top_idx[0, i].item()
            conf = top_conf[0, i].item()
            name = class_names[idx] if class_names and idx < len(class_names) else f"class_{idx}"
            results.append({
                "class": name,
                "class_idx": idx,
                "confidence": round(conf, 4)
            })
        return results

    def get_trainable_params(self) -> int:
        """Count the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self) -> int:
        """Count the total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    @staticmethod
    def load_model(
        model_path: str,
        num_classes: int = 8,
        device: Optional[str] = None
    ) -> "DocumentClassifierCNN":
        """
        Load a trained model from disk.

        Args:
            model_path: Path to the saved .pt file.
            num_classes: Number of classes the model was trained on.
            device: Device to load model on ('cpu', 'cuda', or None for auto).

        Returns:
            Loaded DocumentClassifierCNN instance.

        Raises:
            FileNotFoundError: If model file doesn't exist.
        """
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                f"Please train the model first using notebooks/training.ipynb "
                f"or download a pre-trained model."
            )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        model = DocumentClassifierCNN(num_classes=num_classes, pretrained=False)
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model

    def save_model(self, model_path: str):
        """
        Save model weights to disk.

        Args:
            model_path: Path to save the .pt file.
        """
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), model_path)
        print(f"Model saved to: {model_path}")


# ──────────────────────────────────────────────
# Custom lightweight CNN (Option B — Alternative)
# ──────────────────────────────────────────────

class LightweightDocumentCNN(nn.Module):
    """
    Custom lightweight CNN for document classification.
    
    Use this if you want a from-scratch model instead of fine-tuning.
    Simpler architecture, but may require more training data/epochs.
    """

    def __init__(self, num_classes: int = 8):
        super(LightweightDocumentCNN, self).__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x
