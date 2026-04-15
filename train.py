from pathlib import Path
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageFile
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import CNN


# Some scraped datasets contain partially truncated images.
ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train base CNN model")
    parser.add_argument("--data-dir", default="data/Dataset_Images/Train", help="Training data directory")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--save-path", default="models/model.pth", help="Output weights file")
    parser.add_argument(
        "--skip-image-check",
        action="store_true",
        help="Skip upfront image integrity scan",
    )
    return parser.parse_args()


def verify_and_filter_images(train_data: datasets.ImageFolder) -> list[str]:
    valid_samples = []
    invalid_paths = []

    for path, class_idx in train_data.samples:
        try:
            with Image.open(path) as img:
                img.verify()
            valid_samples.append((path, class_idx))
        except Exception:
            invalid_paths.append(path)

    train_data.samples = valid_samples
    train_data.imgs = valid_samples
    train_data.targets = [class_idx for _, class_idx in valid_samples]
    return invalid_paths


def compute_class_weights(train_data: datasets.ImageFolder, device: torch.device) -> torch.Tensor:
    labels = torch.tensor(train_data.targets, dtype=torch.long)
    class_counts = torch.bincount(labels, minlength=len(train_data.classes)).float()

    weights = 1.0 / class_counts.clamp_min(1.0)
    weights = weights / weights.sum()
    return weights.to(device)


def main() -> None:
    args = parse_args()

    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_data = datasets.ImageFolder(args.data_dir, transform=transform)

    if not args.skip_image_check:
        invalid_paths = verify_and_filter_images(train_data)
        if invalid_paths:
            print(f"Skipped unreadable images: {len(invalid_paths)}")
            for bad_path in invalid_paths[:10]:
                print(f"  - {bad_path}")
            if len(invalid_paths) > 10:
                print("  - ...")

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)

    class_weights = compute_class_weights(train_data, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print(f"Training on {device}")
    print(f"Classes: {train_data.classes}")
    print(f"Samples: {len(train_data)}")

    try:
        for epoch in range(args.epochs):
            total_loss = 0.0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch: {epoch + 1} Loss: {total_loss / len(train_loader):.4f}")
    except KeyboardInterrupt:
        print("Training interrupted by user. Saving latest checkpoint...")

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Saved model to: {save_path}")


if __name__ == "__main__":
    main()