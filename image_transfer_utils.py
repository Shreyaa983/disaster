from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
from torchvision.models import VGG16_Weights, ResNet18_Weights, vgg16, resnet18


IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transform(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_eval_transform(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_image_datasets(train_dir: str, test_dir: str, image_size: int = IMAGE_SIZE):
    train_for_split = datasets.ImageFolder(train_dir, transform=build_eval_transform(image_size))
    train_for_training = datasets.ImageFolder(train_dir, transform=build_train_transform(image_size))
    train_for_eval = datasets.ImageFolder(train_dir, transform=build_eval_transform(image_size))
    test_dataset = datasets.ImageFolder(test_dir, transform=build_eval_transform(image_size))

    return train_for_split, train_for_training, train_for_eval, test_dataset


def verify_class_layout(train_dataset: datasets.ImageFolder, test_dataset: datasets.ImageFolder) -> None:
    train_classes = [class_name.lower() for class_name in train_dataset.classes]
    test_classes = [class_name.lower() for class_name in test_dataset.classes]
    if train_classes != test_classes:
        raise ValueError(
            "Train and test folders must contain the same class names in compatible order. "
            f"Train: {train_dataset.classes} | Test: {test_dataset.classes}"
        )


def build_split_loaders(
    train_for_split: datasets.ImageFolder,
    train_for_training: datasets.ImageFolder,
    train_for_eval: datasets.ImageFolder,
    batch_size: int,
    val_split: float,
    seed: int,
) -> tuple[DataLoader, DataLoader, list[str], list[int], list[int]]:
    total_size = len(train_for_split)
    val_size = max(1, int(total_size * val_split))
    train_size = total_size - val_size

    generator = torch.Generator().manual_seed(seed)
    train_subset_split, val_subset_split = random_split(
        train_for_split,
        [train_size, val_size],
        generator=generator,
    )

    train_indices = train_subset_split.indices
    val_indices = val_subset_split.indices

    train_subset = Subset(train_for_training, train_indices)
    val_subset = Subset(train_for_eval, val_indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, train_for_split.classes, train_indices, val_indices


def build_test_loader(test_dataset: datasets.ImageFolder, batch_size: int) -> DataLoader:
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def compute_class_weights(dataset: datasets.ImageFolder, sample_indices: list[int], device: torch.device) -> torch.Tensor:
    labels = torch.tensor([dataset.targets[index] for index in sample_indices], dtype=torch.long)
    class_counts = torch.bincount(labels, minlength=len(dataset.classes)).float()
    weights = 1.0 / class_counts.clamp_min(1.0)
    weights = weights / weights.sum()
    return weights.to(device)


def _count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"total_parameters": int(total), "trainable_parameters": int(trainable)}


def build_vgg16_model(num_classes: int, pretrained: bool = True, freeze_base: bool = True):
    weights = VGG16_Weights.DEFAULT if pretrained else None
    try:
        model = vgg16(weights=weights)
    except Exception:
        model = vgg16(weights=None)
        pretrained = False

    if freeze_base and pretrained:
        for parameter in model.features.parameters():
            parameter.requires_grad = False

    classifier_input = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(classifier_input, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(4096, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes),
    )

    metadata = {
        "arch": "vgg16",
        "pretrained": bool(pretrained),
        "freeze_base": bool(freeze_base and pretrained),
        **_count_parameters(model),
    }
    return model, metadata


def build_resnet18_model(num_classes: int):
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    metadata = {"arch": "resnet18", **_count_parameters(model)}
    return model, metadata


def load_model_checkpoint(model: nn.Module, checkpoint_path: str, device: torch.device) -> nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint

    if any(key.startswith("module.") for key in state_dict.keys()):
        state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def train_epoch(model: nn.Module, loader: DataLoader, criterion, optimizer, device: torch.device) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    running_correct = 0
    sample_count = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        running_correct += (outputs.argmax(dim=1) == labels).sum().item()
        sample_count += batch_size

    avg_loss = running_loss / max(sample_count, 1)
    accuracy = running_correct / max(sample_count, 1)
    return float(avg_loss), float(accuracy)


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, criterion, device: torch.device, class_names: list[str]) -> dict[str, Any]:
    model.eval()
    running_loss = 0.0
    predictions: list[int] = []
    labels: list[int] = []
    sample_count = 0

    for images, batch_labels in loader:
        images = images.to(device)
        batch_labels = batch_labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, batch_labels)

        batch_size = batch_labels.size(0)
        running_loss += loss.item() * batch_size
        sample_count += batch_size

        predictions.extend(outputs.argmax(dim=1).cpu().tolist())
        labels.extend(batch_labels.cpu().tolist())

    avg_loss = running_loss / max(sample_count, 1)
    accuracy = accuracy_score(labels, predictions) if labels else 0.0
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="weighted",
        zero_division=0,
    ) if labels else (0.0, 0.0, 0.0, None)

    return {
        "loss": float(avg_loss),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "classification_report": classification_report(
            labels,
            predictions,
            target_names=class_names,
            zero_division=0,
        ) if labels else "",
        "confusion_matrix": confusion_matrix(labels, predictions).tolist() if labels else [],
        "predictions": predictions,
        "labels": labels,
    }


def best_state_from_history(model: nn.Module):
    return deepcopy(model.state_dict())