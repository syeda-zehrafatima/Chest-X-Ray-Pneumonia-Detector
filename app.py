# app.py
import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from infer_gradcam import load_model, get_prediction_and_cam
from fpdf import FPDF
import datetime
import os

# -----------------------
# Load model once
# -----------------------
@st.cache_resource
def load_trained_model():
    return load_model("models/xray_model.pth")

model = load_trained_model()

# -----------------------
# Page Setup
# -----------------------
st.set_page_config(page_title="Chest X-ray Pneumonia Detector", layout="wide")
st.title("🩻 Chest X-ray Pneumonia Detector with Grad-CAM")
st.markdown("### ⚠️ Research Demo — Not a medical diagnostic tool")

# Sidebar
st.sidebar.header("ℹ️ Instructions")
st.sidebar.markdown("""
1. Enter your **Name** and **Age**  
2. Upload a Chest X-ray image (JPG/PNG).  
3. Click **Analyze** to get prediction.  
4. View **Grad-CAM heatmap** and download a PDF report.  
""")
st.sidebar.info("Trained on Chest X-ray dataset (Normal vs Pneumonia).")

# -----------------------
# Patient Info
# -----------------------
name = st.text_input("👤 Enter Patient Name")
age = st.number_input("🎂 Enter Age", min_value=1, max_value=120, value=25)

# -----------------------
# File Upload
# -----------------------
uploaded_file = st.file_uploader("📤 Upload a Chest X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and name and age:
    with open("temp.png", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Two-column layout
    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.image("temp.png", caption="Uploaded X-ray", use_container_width=True)

    with col2:
        if st.button("🔎 Analyze"):
            st.write("⏳ Running prediction...")
            prediction, cam_overlay, probs = get_prediction_and_cam(model, "temp.png")

            # -----------------------
            # Prediction result
            # -----------------------
            if prediction == "PNEUMONIA":
                st.error(f"🩸 Prediction: {prediction}")
                summary = "The X-ray shows signs consistent with Pneumonia."
                treatment = (
                    "- Suggested Treatment (General Guidance):\n"
                    "- Consult a pulmonologist or physician.\n"
                    "- Antibiotics may be prescribed (if bacterial).\n"
                    "- Adequate rest and hydration.\n"
                    "- Hospitalization if severe symptoms."
                )
            else:
                st.success(f"✅ Prediction: {prediction}")
                summary = "The X-ray appears normal. No signs of Pneumonia detected."
                treatment = (
                    "- Suggested Advice:\n"
                    "- Maintain a healthy lifestyle.\n"
                    "- Regular check-ups if symptoms appear.\n"
                    "- No medical intervention needed at this time."
                )

            # -----------------------
            # Confidence chart
            # -----------------------
            fig, ax = plt.subplots()
            bars = ax.bar(["NORMAL", "PNEUMONIA"], probs, color=["#4CAF50", "#F44336"])
            ax.set_ylabel("Probability")
            ax.set_ylim([0, 1])
            ax.bar_label(bars, fmt="%.2f")
            st.pyplot(fig)

            # -----------------------
            # Grad-CAM Visualization
            # -----------------------
            alpha = st.slider("Heatmap Transparency", 0.0, 1.0, 0.5)

            orig_img = cv2.cvtColor(cv2.imread("temp.png"), cv2.COLOR_BGR2RGB)
            cam_resized = cv2.resize(cam_overlay, (orig_img.shape[1], orig_img.shape[0]))

            if len(cam_resized.shape) == 2:
                cam_resized = cv2.cvtColor(cam_resized, cv2.COLOR_GRAY2RGB)

            overlay = cv2.addWeighted(orig_img, 1 - alpha, cam_resized, alpha, 0)
            st.image(overlay, caption="Grad-CAM Visualization", use_container_width=True)

            # -----------------------
            # Generate PDF Report
            # -----------------------
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)

            pdf.cell(200, 10, txt="Chest X-ray Pneumonia Report", ln=True, align="C")
            pdf.ln(5)

            pdf.cell(200, 10, txt=f"Patient Name: {name}", ln=True)
            pdf.cell(200, 10, txt=f"Age: {age}", ln=True)
            pdf.cell(200, 10, txt=f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
            pdf.ln(5)

            pdf.cell(200, 10, txt=f"Prediction: {prediction}", ln=True)
            pdf.cell(200, 10, txt=f"Probabilities:", ln=True)
            pdf.cell(200, 10, txt=f"   - Normal = {probs[0]*100:.2f}%", ln=True)
            pdf.cell(200, 10, txt=f"   - Pneumonia = {probs[1]*100:.2f}%", ln=True)
            pdf.ln(10)

            pdf.multi_cell(0, 10, txt=f"Summary: {summary}")
            pdf.ln(5)
            pdf.multi_cell(0, 10, txt=treatment)

            pdf.ln(10)
            pdf.multi_cell(0, 10, txt="Disclaimer: This is a research demo. Not for clinical use.")

            # Save PDF temporarily
            report_path = "report.pdf"
            pdf.output(report_path)

            # Download button (always visible after analysis)
            with open(report_path, "rb") as file:
                st.download_button("⬇️ Download PDF Report", file, "report.pdf", "application/pdf")

else:
    st.warning("⚠️ Please enter your name, age, and upload an X-ray image to continue.")
