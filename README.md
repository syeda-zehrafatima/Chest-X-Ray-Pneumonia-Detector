# Chest-X-Ray-Pneumonia-Detector

# 📌 Description & Purpose

This project is a research-focused AI system designed to automatically detect Pneumonia from Chest X-ray images. It uses PyTorch, Streamlit, and a fine-tuned ResNet18 CNN model to classify an image as Normal or Pneumonia.
To enhance transparency, the system also generates Grad-CAM heatmaps, helping users visualize which lung areas influenced the model’s prediction.

# ⚠️ Use Case Notice

This tool is intended for:

Educational demonstrations

Research and academic learning

Showcasing AI capabilities in medical imaging

Visualizing model reasoning using Grad-CAM

Not meant for clinical or diagnostic use.

# 🔧 Technologies Used

PyTorch & Torchvision – Model training and transfer learning

OpenCV & PIL – Image processing and transformations

Streamlit – Interactive web interface

FPDF – Automatic PDF report generation

KaggleHub – Dataset downloading

# ⭐ Key Features

Classifies Chest X-rays into NORMAL or PNEUMONIA

Generates Grad-CAM heatmaps for better interpretability

User-friendly Streamlit web app with patient input fields

Upload X-ray images in PNG/JPG format

Produces a downloadable PDF report with:

Prediction

Probability chart

Heatmap visualization

Model trained on the official Kaggle Chest X-ray Pneumonia dataset

# 📂 Dataset Overview

Using the Kaggle Chest X-ray Pneumonia dataset, structured as:

https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

train/   → training images  
val/     → validation images  
test/    → testing images  

# How to Use
1️⃣ Clone the repository
git clone https://github.com/yourusername/chest-xray-pneumonia-detector.git
cd chest-xray-pneumonia-detector

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ (Optional) Train the model
python train.py

4️⃣ Launch the Streamlit app
streamlit run app.py

5️⃣ Interact with the App

Enter patient details (name, age)

Upload a Chest X-ray image

Click Analyze to view:

Prediction

Probability chart

Grad-CAM heatmap

Downloadable PDF report

# 📁 Project Structure
chest-xray-pneumonia-detector/
│
├── app.py               → Streamlit UI  
├── train.py             → Training script  
├── infer_gradcam.py     → Model + Grad-CAM logic  
├── models/
│     └── xray_model.pth → Saved trained model  
├── requirements.txt      → All dependencies  
└── README.md             → Project overview  





