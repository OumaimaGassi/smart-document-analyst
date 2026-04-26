"""
training.py — CNN Document Classifier Training Script

This script trains a ResNet-18 based CNN for document image classification
on a subset of the RVL-CDIP dataset. Can be run as a script or converted
to a Jupyter notebook with: jupytext --to notebook training.py

Team: Benmouma Salma, Gassi Oumaima
UIR S8 — Multi-Agent AI Project, Prof. Hakim Hafidi

Usage:
    python notebooks/training.py
    # Or in Colab: upload and run cells
"""

# %% [markdown]
# # 📄 Document Classifier — CNN Training
# **Team:** Benmouma Salma, Gassi Oumaima
# **Project:** Smart Document Analyst — Multi-Agent AI System
# **UIR S8** — AI & Big Data 2025–2026

# %% Imports
import os
import sys
import time
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support
)
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.document_classifier import DocumentClassifierCNN
from src.utils.preprocessing import DOCUMENT_CLASSES_SUBSET, CNN_INPUT_SIZE

# %% Configuration
# ============================================
# Training Configuration
# ============================================

CONFIG = {
    "dataset_dir": "data/dataset",       # Directory with class subfolders
    "model_save_path": "model/document_classifier.pt",
    "num_classes": len(DOCUMENT_CLASSES_SUBSET),
    "class_names": DOCUMENT_CLASSES_SUBSET,
    "input_size": CNN_INPUT_SIZE,          # 224
    "batch_size": 32,
    "num_epochs": 20,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "patience": 5,                         # Early stopping patience
    "train_split": 0.8,
    "val_split": 0.1,
    "test_split": 0.1,
    "seed": 42,
    "pretrained": True,
    "dropout_rate": 0.3,
}

print("=" * 60)
print("📄 Document Classifier — Training Configuration")
print("=" * 60)
for key, value in CONFIG.items():
    if key != "class_names":
        print(f"  {key}: {value}")
print(f"  classes: {CONFIG['class_names']}")
print(f"  device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
print("=" * 60)

# %% Set seeds for reproducibility
def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(CONFIG["seed"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n🖥️  Using device: {device}")

# %% Data Transforms
# ============================================
# Data Augmentation & Preprocessing
# ============================================

train_transform = transforms.Compose([
    transforms.Resize((CONFIG["input_size"] + 32, CONFIG["input_size"] + 32)),
    transforms.RandomCrop(CONFIG["input_size"]),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomRotation(5),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomHorizontalFlip(p=0.1),  # Documents rarely flipped, low prob
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((CONFIG["input_size"], CONFIG["input_size"])),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

print("\n✅ Data transforms defined")

# %% Load Dataset
# ============================================
# Dataset Loading
# ============================================
# Expected structure:
#   data/dataset/
#   ├── letter/
#   │   ├── img001.png
#   │   └── ...
#   ├── invoice/
#   │   ├── img001.png
#   │   └── ...
#   └── ...

def load_dataset(data_dir, transform):
    """Load dataset from directory with class subfolders."""
    if not Path(data_dir).exists():
        print(f"⚠️  Dataset directory not found: {data_dir}")
        print("   Please download and organize the RVL-CDIP subset first.")
        print("   Expected structure: data/dataset/<class_name>/<image_files>")
        return None
    
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    print(f"\n📊 Dataset loaded: {len(dataset)} images")
    print(f"   Classes: {dataset.classes}")
    print(f"   Class distribution:")
    class_counts = {}
    for _, label in dataset.samples:
        class_name = dataset.classes[label]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    for cls, count in sorted(class_counts.items()):
        print(f"     {cls}: {count}")
    return dataset


def split_dataset(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """Split dataset into train/val/test sets."""
    total = len(dataset)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    test_size = total - train_size - val_size
    
    train_set, val_set, test_set = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(CONFIG["seed"])
    )
    
    print(f"\n📂 Dataset split:")
    print(f"   Train: {len(train_set)}")
    print(f"   Val:   {len(val_set)}")
    print(f"   Test:  {len(test_set)}")
    
    return train_set, val_set, test_set

# Load data
full_dataset = load_dataset(CONFIG["dataset_dir"], train_transform)

if full_dataset is not None:
    train_set, val_set, test_set = split_dataset(full_dataset)
    
    # Apply different transforms to val/test
    val_set.dataset.transform = val_transform
    
    train_loader = DataLoader(train_set, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2)
    
    print("\n✅ Data loaders created")
else:
    print("\n⚠️  Skipping dataset loading — will demonstrate with synthetic data")
    train_loader = val_loader = test_loader = None

# %% Model Initialization
# ============================================
# Model Setup
# ============================================

model = DocumentClassifierCNN(
    num_classes=CONFIG["num_classes"],
    pretrained=CONFIG["pretrained"],
    dropout_rate=CONFIG["dropout_rate"]
)
model = model.to(device)

print(f"\n🧠 Model: DocumentClassifierCNN (ResNet-18 based)")
print(f"   Total parameters:     {model.get_total_params():,}")
print(f"   Trainable parameters: {model.get_trainable_params():,}")
print(f"   Frozen parameters:    {model.get_total_params() - model.get_trainable_params():,}")

# %% Loss, Optimizer, Scheduler

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=CONFIG["learning_rate"],
    weight_decay=CONFIG["weight_decay"]
)
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=CONFIG["num_epochs"], eta_min=1e-6
)

print(f"\n⚙️  Loss: CrossEntropyLoss")
print(f"   Optimizer: AdamW (lr={CONFIG['learning_rate']})")
print(f"   Scheduler: CosineAnnealingLR")

# %% Training Loop
# ============================================
# Training
# ============================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(loader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Validating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                num_epochs, patience, device, save_path):
    """Full training loop with early stopping."""
    
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    patience_counter = 0
    
    print(f"\n{'='*60}")
    print(f"🚀 Starting training for {num_epochs} epochs...")
    print(f"{'='*60}\n")
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        # Record history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        elapsed = time.time() - start_time
        lr = optimizer.param_groups[0]["lr"]
        
        print(f"Epoch {epoch+1:3d}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
              f"LR: {lr:.2e} | {elapsed:.1f}s")
        
        # Early stopping check
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model
            model.save_model(save_path)
            print(f"  💾 New best model saved (val_acc: {val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⏹️  Early stopping at epoch {epoch+1} (patience={patience})")
                break
    
    print(f"\n{'='*60}")
    print(f"✅ Training complete! Best val accuracy: {best_val_acc:.4f}")
    print(f"{'='*60}")
    
    return history


# Run training if data is available
if train_loader is not None:
    history = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        CONFIG["num_epochs"], CONFIG["patience"], device, CONFIG["model_save_path"]
    )
else:
    print("\n⚠️  No training data available. Skipping training.")
    history = None

# %% Evaluation
# ============================================
# Test Set Evaluation
# ============================================

def evaluate_model(model, test_loader, class_names, device):
    """Full evaluation on test set with metrics and confusion matrix."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Classification report
    print("\n" + "=" * 60)
    print("📊 CLASSIFICATION REPORT")
    print("=" * 60)
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print(report)
    
    # Accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    
    print(f"\n📈 Overall Metrics:")
    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1 Score:  {f1:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "all_preds": all_preds,
        "all_labels": all_labels
    }


if test_loader is not None:
    # Load best model
    best_model = DocumentClassifierCNN.load_model(
        CONFIG["model_save_path"], CONFIG["num_classes"], device
    )
    eval_results = evaluate_model(best_model, test_loader, CONFIG["class_names"], device)
else:
    print("\n⚠️  No test data. Skipping evaluation.")
    eval_results = None

# %% Visualization
# ============================================
# Training Curves & Confusion Matrix Plots
# ============================================

def plot_training_curves(history, save_path="outputs/training_curves.png"):
    """Plot training and validation loss/accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history["train_loss"]) + 1)
    
    # Loss curves
    ax1.plot(epochs, history["train_loss"], "b-o", label="Train Loss", markersize=3)
    ax1.plot(epochs, history["val_loss"], "r-o", label="Val Loss", markersize=3)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curves
    ax2.plot(epochs, history["train_acc"], "b-o", label="Train Acc", markersize=3)
    ax2.plot(epochs, history["val_acc"], "r-o", label="Val Acc", markersize=3)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training & Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle("Document Classifier — Training Curves\nBenmouma Salma & Gassi Oumaima", fontsize=12)
    plt.tight_layout()
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"📊 Training curves saved to: {save_path}")


def plot_confusion_matrix(cm, class_names, save_path="outputs/confusion_matrix.png"):
    """Plot the confusion matrix as a heatmap."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix — Document Classifier\nBenmouma Salma & Gassi Oumaima")
    plt.tight_layout()
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"📊 Confusion matrix saved to: {save_path}")


if history is not None:
    plot_training_curves(history)

if eval_results is not None:
    plot_confusion_matrix(eval_results["confusion_matrix"], CONFIG["class_names"])

# %% Save training metadata
# ============================================
# Save Training Metadata
# ============================================

training_meta = {
    "timestamp": datetime.now().isoformat(),
    "team": "Benmouma Salma, Gassi Oumaima",
    "project": "Smart Document Analyst — UIR S8",
    "config": {k: v for k, v in CONFIG.items() if k != "class_names"},
    "class_names": CONFIG["class_names"],
    "device": str(device),
    "pytorch_version": torch.__version__,
}

if eval_results:
    training_meta["evaluation"] = {
        "accuracy": eval_results["accuracy"],
        "precision": eval_results["precision"],
        "recall": eval_results["recall"],
        "f1": eval_results["f1"],
    }

meta_path = "model/training_metadata.json"
Path(meta_path).parent.mkdir(parents=True, exist_ok=True)
with open(meta_path, "w") as f:
    json.dump(training_meta, f, indent=2, default=str)
print(f"\n📋 Training metadata saved to: {meta_path}")

print("\n" + "=" * 60)
print("🎉 Training pipeline complete!")
print("   Model: model/document_classifier.pt")
print("   Metadata: model/training_metadata.json")
print("=" * 60)
