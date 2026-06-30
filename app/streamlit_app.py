import os
import sys

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.explainability import GradCAM, load_image, overlay_heatmap, predict_with_uncertainty
from src.train import get_classifier, UNet


@st.cache_resource
def load_classification_model(model_path: str, num_classes: int = 2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_classifier(num_classes=num_classes, pretrained=False)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model


@st.cache_resource
def load_segmentation_model(model_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=3, out_channels=1)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model


def predict_image(model, image: Image.Image):
    image_resized = image.resize((224, 224))
    image_array = np.array(image_resized).astype(np.float32) / 255.0
    normalized = (image_array - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0).float()
    device = next(model.parameters()).device
    tensor = tensor.to(device)
    with torch.no_grad():
        output = model(tensor)
        scores = torch.softmax(output, dim=1).cpu().numpy()[0]
        label = int(np.argmax(scores))
    return label, scores, tensor
    return label, scores, tensor


def generate_segmentation(model, image: Image.Image):
    image = image.resize((256, 256))
    image_array = np.array(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(image_array.transpose(2, 0, 1)).unsqueeze(0).float()
    device = next(model.parameters()).device
    tensor = tensor.to(device)
    with torch.no_grad():
        output = model(tensor)
        mask = torch.sigmoid(output).cpu().numpy()[0, 0]
    return mask


def corruption_preview(image: np.ndarray, corruption_type: str):
    image = image.astype(np.float32) / 255.0
    if corruption_type == "Noise":
        noise = np.random.normal(0, 0.05, image.shape)
        corrupted = np.clip(image + noise, 0.0, 1.0)
    elif corruption_type == "Blur":
        corrupted = cv2.GaussianBlur((image * 255).astype(np.uint8), (9, 9), 0).astype(np.float32) / 255.0
    else:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 25]
        _, buffer = cv2.imencode('.jpg', (image * 255).astype(np.uint8), encode_param)
        corrupted = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        corrupted = cv2.cvtColor(corrupted, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return (corrupted * 255).astype(np.uint8)


def main():
    st.set_page_config(page_title="Brain Tumor Detection", layout="wide")
    st.title("Explainable Brain Tumor Detection from MRI")
    st.markdown("Upload an MRI image to get a tumor prediction, explanation heatmap, and optional segmentation preview.")

    uploaded_file = st.file_uploader("Upload an MRI scan", type=["png", "jpg", "jpeg", "bmp"])
    classifier_path = os.path.join(ROOT_DIR, "models", "classifier.pth")
    segmentation_path = os.path.join(ROOT_DIR, "models", "unet.pth")

    model = None
    seg_model = None
    if os.path.exists(classifier_path):
        model = load_classification_model(classifier_path)
    if os.path.exists(segmentation_path):
        seg_model = load_segmentation_model(segmentation_path)

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded MRI scan", use_column_width=True)

        if model is None:
            st.warning("No classification model found in models/classifier.pth")
            return

        labels = ["No Tumor", "Tumor"]
        predicted_label, scores, tensor = predict_image(model, image)
        st.subheader("Classification")
        st.write(f"Prediction: **{labels[predicted_label]}**")
        st.write(f"Confidence: {scores[predicted_label]:.3f}")

        with st.expander("Show uncertainty estimation"):
            uncertainty = predict_with_uncertainty(model, tensor, n_samples=12)
            st.write(uncertainty)

        with st.expander("Show explanation heatmap"):
            target_layer = model.layer4[-1]
            gradcam = GradCAM(model, target_layer)
            heatmap = gradcam.generate(tensor, class_idx=predicted_label)
            overlay = overlay_heatmap(np.array(image.resize((224, 224))), heatmap)
            st.image(overlay, caption="Grad-CAM heatmap", use_column_width=True)

        if seg_model is not None:
            st.subheader("Segmentation preview")
            mask = generate_segmentation(seg_model, image)
            mask_overlay = (mask > 0.5).astype(np.uint8) * 255
            st.image(mask_overlay, caption="Predicted tumor mask", use_column_width=True)

        corruption_type = st.selectbox("Apply simulated corruption", ["None", "Noise", "Blur", "JPEG compression"])
        if corruption_type != "None":
            st.subheader("Corrupted input preview")
            corrupted = corruption_preview(np.array(image), corruption_type)
            st.image(corrupted, caption=f"{corruption_type} corrupted image", use_column_width=True)
    else:
        st.info("Upload a brain MRI image to begin analysis.")


if __name__ == "__main__":
    main()
