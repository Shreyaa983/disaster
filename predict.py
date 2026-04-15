import torch
from PIL import Image
from torchvision import transforms

from model import CNN   # same as train.py

# class names (VERY IMPORTANT: same order as training folders)
classes = ["Earthquake", "Fire", "Flood", "Normal"]

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load model
model = CNN().to(device)
model.load_state_dict(torch.load("models/model.pth", map_location=device))
model.eval()

# image preprocessing (SAME as train)
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

# load image
print("Image path loading...")
img = Image.open(r"C:\Users\Shreya\Downloads\projects\mpr-ml\calamity_ai_pytorchh\data\Dataset_Images\Test\flood1.png").convert("RGB")
print("Image loaded successfully")
img = transform(img).unsqueeze(0).to(device)

# prediction
with torch.no_grad():
    outputs = model(img)
    _, predicted = torch.max(outputs, 1)    
    epochs = 20 or 30

# output
print("🔥 Prediction:", classes[predicted.item()])