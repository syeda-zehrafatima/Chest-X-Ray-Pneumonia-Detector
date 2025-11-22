# predict.py
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

MODEL_PATH = "models/xray_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load model
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes: Normal, Pneumonia
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Prediction function
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    img_t = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_t)
        _, predicted = torch.max(outputs, 1)

    classes = ["Normal", "Pneumonia"]
    return classes[predicted.item()]

if __name__ == "__main__":
    image_path = input("🩻 Enter path to X-ray image: ")
    result = predict_image(image_path)
    print(f"✅ Prediction: {result}")
