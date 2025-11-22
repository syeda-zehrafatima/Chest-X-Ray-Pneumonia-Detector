# ========== STEP 1: Install Libraries ==========
# Run these only once in terminal or notebook
# pip install torch torchvision torchaudio
# pip install kagglehub

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import kagglehub

# ========== STEP 2: Download Dataset ==========
print("📥 Downloading dataset...")
path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
print("✅ Path to dataset files:", path)

# Dataset structure after download:
# path/chest_xray/train
# path/chest_xray/test
# path/chest_xray/val

train_dir = os.path.join(path, "chest_xray", "train")
test_dir = os.path.join(path, "chest_xray", "test")
val_dir = os.path.join(path, "chest_xray", "val")

# ========== STEP 3: Transformations ==========
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])   # normalize grayscale
])

# ========== STEP 4: Load Dataset ==========
train_data = datasets.ImageFolder(train_dir, transform=transform)
val_data = datasets.ImageFolder(val_dir, transform=transform)
test_data = datasets.ImageFolder(test_dir, transform=transform)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(val_data, batch_size=16, shuffle=False)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

print(f"✅ Training samples: {len(train_data)}, Validation samples: {len(val_data)}, Test samples: {len(test_data)}")

# ========== STEP 5: Model ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)   # 2 classes: NORMAL, PNEUMONIA
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ========== STEP 6: Training ==========
epochs = 3  # Start with small epochs
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = 100 * correct / total
    avg_loss = running_loss / len(train_loader)
    print(f"📊 Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Training Accuracy: {train_acc:.2f}%")

# ========== STEP 7: Evaluation ==========
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"✅ Final Test Accuracy: {100 * correct / total:.2f}%")

# ========== STEP 8: Save Model ==========
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/xray_model.pth")
print("💾 Model saved at models/xray_model.pth")
