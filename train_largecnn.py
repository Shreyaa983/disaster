from pathlib import Path
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from largecnn import LargeCNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LargeCNN on calamity dataset")
    parser.add_argument("--data-dir", default="data/Dataset_Images/Train", help="Path to training dataset")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--save-path", default="models/largecnn.pth", help="Path to save model weights")
    parser.add_argument(
        "--disable-class-weights",
        action="store_true",
        help="Disable inverse-frequency class weights in loss",
    )
    return parser.parse_args()


def build_dataloader(data_dir: str, batch_size: int) -> tuple[DataLoader, datasets.ImageFolder]:
    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_data = datasets.ImageFolder(data_dir, transform=transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    return train_loader, train_data


def compute_class_weights(train_data: datasets.ImageFolder, device: torch.device) -> torch.Tensor:
    labels = torch.tensor(train_data.targets, dtype=torch.long)
    class_counts = torch.bincount(labels, minlength=len(train_data.classes)).float()

    # Inverse-frequency weighting improves minority-class learning.
    weights = 1.0 / class_counts.clamp_min(1.0)
    weights = weights / weights.sum()
    return weights.to(device)


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, train_data = build_dataloader(args.data_dir, args.batch_size)

    model = LargeCNN().to(device)

    if args.disable_class_weights:
        criterion = nn.CrossEntropyLoss()
    else:
        class_weights = compute_class_weights(train_data, device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print(f"Training LargeCNN on {device}")
    print(f"Classes: {train_data.classes}")
    print(f"Epochs: {args.epochs} | Batch size: {args.batch_size} | LR: {args.lr}")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{args.epochs} - Loss: {avg_loss:.4f}")

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Saved LargeCNN weights to: {save_path}")


if __name__ == "__main__":
    main()
