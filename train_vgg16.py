from __future__ import annotations

import argparse
import json
import os
import time
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from image_transfer_utils import (
    build_image_datasets,
    build_resnet18_model,
    build_split_loaders,
    build_test_loader,
    build_vgg16_model,
    compute_class_weights,
    evaluate_model,
    load_model_checkpoint,
    train_epoch,
    verify_class_layout,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a transfer-learning VGG16 model on the disaster image dataset")
    parser.add_argument("--train-dir", default="data/Dataset_Images/Train", help="Training image directory")
    parser.add_argument("--test-dir", default="data/Dataset_Images/Test", help="Test image directory")
    parser.add_argument("--epochs", type=int, default=15, help="Number of fine-tuning epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split from training folder")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--pretrained", action="store_true", default=True, help="Use ImageNet pretrained weights")
    parser.add_argument("--no-pretrained", action="store_false", dest="pretrained", help="Disable ImageNet pretrained weights")
    parser.add_argument("--freeze-base", action="store_true", default=True, help="Freeze the convolutional backbone")
    parser.add_argument("--no-freeze-base", action="store_false", dest="freeze_base", help="Fine-tune the convolutional backbone")
    parser.add_argument("--save-path", default="models/vgg16_disaster.pth", help="Path to save the VGG16 checkpoint")
    parser.add_argument("--metrics-path", default="models/vgg16_metrics.json", help="Path to save training metrics")
    parser.add_argument(
        "--compare-resnet-checkpoint",
        default=os.environ.get("RESNET_MODEL_PATH", ""),
        help="Optional ResNet checkpoint to evaluate on the same test split after VGG training",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_for_split, train_for_training, train_for_eval, test_dataset = build_image_datasets(args.train_dir, args.test_dir)
    verify_class_layout(train_for_split, test_dataset)

    train_loader, val_loader, class_names, train_indices, _ = build_split_loaders(
        train_for_split=train_for_split,
        train_for_training=train_for_training,
        train_for_eval=train_for_eval,
        batch_size=args.batch_size,
        val_split=args.val_split,
        seed=args.seed,
    )
    test_loader = build_test_loader(test_dataset, args.batch_size)

    model, model_metadata = build_vgg16_model(
        num_classes=len(class_names),
        pretrained=args.pretrained,
        freeze_base=args.freeze_base,
    )
    model = model.to(device)

    class_weights = compute_class_weights(train_for_split, train_indices, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)

    print(f"Training VGG16 on {device}")
    print(f"Classes: {class_names}")
    print(f"Samples: {len(train_for_split)} | Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_dataset)}")
    print(f"Model metadata: {model_metadata}")

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }
    best_val_loss = float("inf")
    best_state = deepcopy(model.state_dict())

    start_time = time.time()
    for epoch in range(args.epochs):
        train_loss, train_accuracy = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate_model(model, val_loader, criterion, device, class_names)
        scheduler.step(val_metrics["loss"])

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(val_metrics["loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = deepcopy(model.state_dict())

        print(
            f"Epoch {epoch + 1}/{args.epochs} - "
            f"train loss: {train_loss:.4f} train acc: {train_accuracy:.4f} | "
            f"val loss: {val_metrics['loss']:.4f} val acc: {val_metrics['accuracy']:.4f}"
        )

    training_seconds = time.time() - start_time
    model.load_state_dict(best_state)

    test_metrics = evaluate_model(model, test_loader, criterion, device, class_names)

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "model_metadata": model_metadata,
            "history": history,
            "test_metrics": test_metrics,
        },
        save_path,
    )

    metrics_path = Path(args.metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    metrics_payload = {
        "model": model_metadata,
        "history": history,
        "test_metrics": {
            "loss": test_metrics["loss"],
            "accuracy": test_metrics["accuracy"],
            "precision": test_metrics["precision"],
            "recall": test_metrics["recall"],
            "f1": test_metrics["f1"],
            "classification_report": test_metrics["classification_report"],
            "confusion_matrix": test_metrics["confusion_matrix"],
        },
        "training_seconds": float(training_seconds),
        "best_val_loss": float(best_val_loss),
        "class_names": class_names,
        "class_count": len(class_names),
        "image_size": 224,
        "train_dir": args.train_dir,
        "test_dir": args.test_dir,
    }

    with metrics_path.open("w", encoding="utf-8") as file_handle:
        json.dump(metrics_payload, file_handle, indent=2)

    print("\n===== VGG16 Test Metrics =====")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Loss: {test_metrics['loss']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    print(f"F1-score: {test_metrics['f1']:.4f}")
    print("\nClassification Report:\n", test_metrics["classification_report"])
    print("\nConfusion Matrix:\n", test_metrics["confusion_matrix"])
    print(f"\nSaved checkpoint to: {save_path}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Training time: {training_seconds:.2f}s")

    if args.compare_resnet_checkpoint:
        try:
            resnet_model, resnet_metadata = build_resnet18_model(len(class_names))
            resnet_model = load_model_checkpoint(resnet_model, args.compare_resnet_checkpoint, device)
            resnet_metrics = evaluate_model(resnet_model, test_loader, criterion, device, class_names)
            print("\n===== ResNet18 Comparison =====")
            print(f"Checkpoint: {args.compare_resnet_checkpoint}")
            print(f"Accuracy: {resnet_metrics['accuracy']:.4f}")
            print(f"Loss: {resnet_metrics['loss']:.4f}")
            print(f"Precision: {resnet_metrics['precision']:.4f}")
            print(f"Recall: {resnet_metrics['recall']:.4f}")
            print(f"F1-score: {resnet_metrics['f1']:.4f}")
            print(f"Parameters: {resnet_metadata['total_parameters']:,}")
            print(f"Trainable parameters: {resnet_metadata['trainable_parameters']:,}")
            print("\nUse the VGG16 and ResNet18 metrics above to compare accuracy, loss, and model complexity on the same split.")
        except Exception as error:
            print(f"\nCould not evaluate ResNet checkpoint: {error}")


if __name__ == "__main__":
    main()