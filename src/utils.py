"""Utility functions for dataset, training, and evaluation."""

import copy
import json
import os
import random
import time
from typing import Dict, Tuple

import kagglehub
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src import config


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Return CUDA device if available, else CPU."""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# -------------------------------------------------------------------
# Dataset download / preparation
# -------------------------------------------------------------------

def download_and_prepare_dataset(
    dataset_handle: str = config.DATASET_HANDLE,
    extraction_dir: str = config.DATA_ROOT,
) -> str:
    """Download the Kaggle dataset using kagglehub and extract it."""
    os.makedirs(extraction_dir, exist_ok=True)

    print(f"Downloading dataset via kagglehub: {dataset_handle}")
    downloaded_path = kagglehub.dataset_download(dataset_handle)
    print(f"Dataset downloaded to: {downloaded_path}")

    zip_file_path = None
    if os.path.isdir(downloaded_path):
        for root, _, files in os.walk(downloaded_path):
            for file in files:
                if file.endswith(".zip"):
                    zip_file_path = os.path.join(root, file)
                    break
            if zip_file_path:
                break
    elif os.path.isfile(downloaded_path) and str(downloaded_path).endswith(".zip"):
        zip_file_path = downloaded_path

    if zip_file_path:
        print(f"Extracting {zip_file_path} into {extraction_dir}")
        import zipfile

        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(extraction_dir)
        print("Extraction complete.")
    else:
        print("No zip file found; assuming dataset is already extracted.")

    return extraction_dir


def create_dataloaders(
    data_root: str,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
) -> Tuple[DataLoader, DataLoader, DataLoader, list]:
    """Create train, validation, and test DataLoaders from an ImageFolder dataset."""
    train_dir = os.path.join(data_root, config.TRAIN_DIR)
    val_dir = os.path.join(data_root, config.VAL_DIR)
    test_dir = os.path.join(data_root, config.TEST_DIR)

    mean = config.MEAN
    std = config.STD

    train_transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    val_test_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    print(f"Loading training data from: {train_dir}")
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)

    print(f"Loading validation data from: {val_dir}")
    val_dataset = datasets.ImageFolder(val_dir, transform=val_test_transforms)

    print(f"Loading test data from: {test_dir}")
    test_dataset = datasets.ImageFolder(test_dir, transform=val_test_transforms)

    class_names = train_dataset.classes
    print(f"Detected {len(class_names)} classes.")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    print(
        f"Dataset sizes -> train: {len(train_dataset)}, "
        f"val: {len(val_dataset)}, test: {len(test_dataset)}"
    )

    return train_loader, val_loader, test_loader, class_names


# -------------------------------------------------------------------
# Training and evaluation
# -------------------------------------------------------------------

def train_model(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloaders: Dict[str, DataLoader],
    device: torch.device,
    num_epochs: int = config.NUM_EPOCHS,
    checkpoint_path: str = config.BEST_MODEL_PATH,
):
    """Train the model and save the best checkpoint by validation accuracy."""

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    print(f"Starting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        start_time = time.time()
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 10)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            dataloader = dataloaders[phase]
            running_loss = 0.0
            running_corrects = 0
            total_samples = 0

            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                batch_size = inputs.size(0)
                running_loss += loss.item() * batch_size
                running_corrects += torch.sum(preds == labels).item()
                total_samples += batch_size

            epoch_loss = running_loss / total_samples
            epoch_acc = running_corrects / total_samples

            if phase == "train":
                history["train_loss"].append(epoch_loss)
            else:
                history["val_loss"].append(epoch_loss)
                history["val_acc"].append(epoch_acc)

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "val" and epoch_acc > best_val_acc:
                best_val_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), checkpoint_path)
                print(f"New best model saved to {checkpoint_path} (Acc: {best_val_acc:.4f})")

        elapsed = time.time() - start_time
        print(f"Epoch time: {elapsed // 60:.0f}m {elapsed % 60:.0f}s\n")

    print(f"Training complete. Best val accuracy: {best_val_acc:.4f}")
    model.load_state_dict(best_model_wts)
    return model, history


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
):
    """Evaluate a model on a dataloader and return metrics."""
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0
    )

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }

    print("Evaluation metrics:")
    for k, v in metrics.items():
        print(f"{k.capitalize()}: {v:.4f}")

    return metrics


# -------------------------------------------------------------------
# I/O helpers
# -------------------------------------------------------------------

def save_history(history, path: str = config.TRAINING_HISTORY_FILE) -> None:
    """Save training history to a JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {path}")


def load_checkpoint(model: nn.Module, checkpoint_path: str, device: torch.device) -> nn.Module:
    """Load model weights from a checkpoint path."""
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    return model
