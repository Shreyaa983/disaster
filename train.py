import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import CNN

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

train_data = datasets.ImageFolder(
    "data/Dataset_Images/Train",
    transform=transform
)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN().to(device)
# class counts (your dataset)
class_counts = [1927, 3509, 4063, 3900]

weights = 1 / torch.tensor(class_counts, dtype=torch.float)
weights = weights / weights.sum()
weights = weights.to(device)
 
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 20

for epoch in range(epochs):
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print("Epoch:", epoch+1, "Loss:", total_loss/len(train_loader))

# SAVE MODEL
torch.save(model.state_dict(), "models/model.pth")