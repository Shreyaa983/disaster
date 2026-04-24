from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from image_transfer_utils import (
    build_image_datasets,
    build_resnet18_model,
    build_test_loader,
    build_vgg16_model,
    evaluate_model,
    load_model_checkpoint,
    verify_class_layout,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare VGG16 and ResNet18 checkpoints on the disaster test set")
    parser.add_argument("--train-dir", default="data/Dataset_Images/Train", help="Training image directory")
    parser.add_argument("--test-dir", default="data/Dataset_Images/Test", help="Test image directory")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--vgg-checkpoint", default="models/vgg16_disaster.pth", help="VGG16 checkpoint path")
    parser.add_argument(
        "--resnet-checkpoint",
        default="",
        help="ResNet18 checkpoint path. If omitted, uses RESNET_MODEL_PATH from the environment when present.",
    )
    parser.add_argument("--output-path", default="models/image_model_comparison.json", help="Where to save the comparison summary")
    return parser.parse_args()


def resolve_checkpoint_path(explicit_path: str, env_path: str | None) -> str:
    if explicit_path:
        return explicit_path
    if env_path:
        return env_path
    return ""


def format_score(value: float) -> str:
    return f"{value:.4f}"


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_for_split, train_for_training, train_for_eval, test_dataset = build_image_datasets(args.train_dir, args.test_dir)
    verify_class_layout(train_for_split, test_dataset)
    test_loader = build_test_loader(test_dataset, args.batch_size)

    _, vgg_metadata = build_vgg16_model(num_classes=len(train_for_split.classes), pretrained=False, freeze_base=True)
    vgg_model, _ = build_vgg16_model(num_classes=len(train_for_split.classes), pretrained=False, freeze_base=True)

    vgg_checkpoint = Path(args.vgg_checkpoint)
    if not vgg_checkpoint.exists():
        raise FileNotFoundError(f"VGG16 checkpoint not found at {vgg_checkpoint}")

    vgg_model = load_model_checkpoint(vgg_model, str(vgg_checkpoint), device)
    vgg_metrics = evaluate_model(vgg_model, test_loader, torch.nn.CrossEntropyLoss(), device, train_for_split.classes)

    resnet_checkpoint = resolve_checkpoint_path(args.resnet_checkpoint, os.environ.get("RESNET_MODEL_PATH"))
    comparison = {
        "classes": train_for_split.classes,
        "vgg16": {
            "checkpoint": str(vgg_checkpoint),
            "total_parameters": vgg_metadata["total_parameters"],
            "trainable_parameters": vgg_metadata["trainable_parameters"],
            "metrics": {
                "accuracy": vgg_metrics["accuracy"],
                "loss": vgg_metrics["loss"],
                "precision": vgg_metrics["precision"],
                "recall": vgg_metrics["recall"],
                "f1": vgg_metrics["f1"],
            },
        },
    }

    print("===== VGG16 =====")
    print(f"Checkpoint: {vgg_checkpoint}")
    print(f"Accuracy: {format_score(vgg_metrics['accuracy'])}")
    print(f"Loss: {format_score(vgg_metrics['loss'])}")
    print(f"Precision: {format_score(vgg_metrics['precision'])}")
    print(f"Recall: {format_score(vgg_metrics['recall'])}")
    print(f"F1-score: {format_score(vgg_metrics['f1'])}")
    print(f"Parameters: {vgg_metadata['total_parameters']:,}")
    print(f"Trainable parameters: {vgg_metadata['trainable_parameters']:,}")

    if resnet_checkpoint:
        resnet_model, resnet_metadata = build_resnet18_model(len(train_for_split.classes))
        resnet_model = load_model_checkpoint(resnet_model, resnet_checkpoint, device)
        resnet_metrics = evaluate_model(resnet_model, test_loader, torch.nn.CrossEntropyLoss(), device, train_for_split.classes)

        comparison["resnet18"] = {
            "checkpoint": resnet_checkpoint,
            "total_parameters": resnet_metadata["total_parameters"],
            "trainable_parameters": resnet_metadata["trainable_parameters"],
            "metrics": {
                "accuracy": resnet_metrics["accuracy"],
                "loss": resnet_metrics["loss"],
                "precision": resnet_metrics["precision"],
                "recall": resnet_metrics["recall"],
                "f1": resnet_metrics["f1"],
            },
        }

        print("\n===== ResNet18 =====")
        print(f"Checkpoint: {resnet_checkpoint}")
        print(f"Accuracy: {format_score(resnet_metrics['accuracy'])}")
        print(f"Loss: {format_score(resnet_metrics['loss'])}")
        print(f"Precision: {format_score(resnet_metrics['precision'])}")
        print(f"Recall: {format_score(resnet_metrics['recall'])}")
        print(f"F1-score: {format_score(resnet_metrics['f1'])}")
        print(f"Parameters: {resnet_metadata['total_parameters']:,}")
        print(f"Trainable parameters: {resnet_metadata['trainable_parameters']:,}")

        winner = "VGG16" if vgg_metrics["accuracy"] >= resnet_metrics["accuracy"] else "ResNet18"
        print(f"\nWinner on accuracy: {winner}")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_handle:
        json.dump(comparison, file_handle, indent=2)
    print(f"\nSaved comparison summary to: {output_path}")


if __name__ == "__main__":
    main()