import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2

# ---------------------------
# Load trained model
# ---------------------------
def load_model(model_path="models/xray_model.pth"):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes: NORMAL, PNEUMONIA
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    print("✅ Model loaded!")
    return model

# ---------------------------
# Image preprocessing
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])   # same as training
])

# ---------------------------
# Grad-CAM function
# ---------------------------
def generate_gradcam(model, img_tensor, target_layer="layer4"):
    gradients = []
    activations = []

    def save_gradients(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    def save_activations(module, input, output):
        activations.append(output)

    # Hook to target layer
    for name, module in model.named_modules():
        if name == target_layer:
            module.register_forward_hook(save_activations)
            module.register_full_backward_hook(save_gradients)

    # Forward pass
    output = model(img_tensor)
    pred_class = output.argmax(dim=1).item()

    # Backward pass
    model.zero_grad()
    class_loss = output[0, pred_class]
    class_loss.backward()

    # Compute Grad-CAM
    grad = gradients[0].mean(dim=[2, 3], keepdim=True)
    cam = (activations[0] * grad).sum(dim=1).squeeze().detach().numpy()
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = cam / (cam.max() + 1e-8)

    return cam, pred_class, output

# ---------------------------
# Wrapper function for Streamlit
# ---------------------------
def get_prediction_and_cam(model, image_path, alpha=0.4):
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)

    cam, pred_class, output = generate_gradcam(model, img_tensor)

    classes = ["NORMAL", "PNEUMONIA"]
    prediction = classes[pred_class]

    # Probabilities
    probs = torch.softmax(output, dim=1).detach().numpy()[0]

    # Heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    # Original image RGB resized
    img_np = np.array(img.resize((224, 224)).convert("RGB"))

    # Resize heatmap if mismatch
    if heatmap.shape[:2] != img_np.shape[:2]:
        heatmap = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))

    overlay = cv2.addWeighted(img_np, 1 - alpha, heatmap, alpha, 0)

    return prediction, overlay, probs
