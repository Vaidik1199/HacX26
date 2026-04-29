import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms

# -----------------------------

# PAGE CONFIG

# -----------------------------

st.set_page_config(
page_title="Eye Disease Detection",
page_icon="👁",
layout="wide"
)

# -----------------------------

# LOAD MODEL

# -----------------------------

model = models.efficientnet_b0(pretrained=False)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 4)

model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

# -----------------------------

# IMAGE TRANSFORM

# -----------------------------

transform = transforms.Compose([
transforms.Resize((224, 224)),
transforms.ToTensor(),
])

# -----------------------------

# CLASS NAMES

# -----------------------------

classes = ["cataract", "diabetic_retinopathy", "glaucoma", "normal"]

# -----------------------------

# UI

# -----------------------------

st.title("👁 Eye Disease Detection")
st.caption("Upload a retinal image and get prediction")

uploaded_file = st.file_uploader(
"Upload retinal image",
type=["jpg", "jpeg", "png"]
)

# -----------------------------

# MAIN LOGIC

# -----------------------------

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", width=300)

    with col2:
        if st.button("🚀 Predict"):
            image = image.convert("RGB")
            img_tensor = transform(image).unsqueeze(0)

            with torch.no_grad():
                outputs = model(img_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probs, 1)

            prediction = classes[predicted.item()]
            confidence = int(confidence.item() * 100)

            st.success(f"Prediction: {prediction.upper()}")
            st.info(f"Confidence: {confidence}%")
            st.progress(confidence / 100)