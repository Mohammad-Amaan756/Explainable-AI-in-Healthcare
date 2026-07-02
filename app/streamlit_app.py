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

from src.explainability import (
    GradCAM,
    overlay_heatmap,
    predict_with_uncertainty,
)
from src.train import get_classifier, UNet


CLASS_NAMES = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary",
    
]


@st.cache_resource
def load_classification_model(model_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_classifier(
        num_classes=len(CLASS_NAMES),
        pretrained=False,
    )

    checkpoint = torch.load(model_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)
    model.eval()

    return model


@st.cache_resource
def load_segmentation_model(model_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(in_channels=3, out_channels=1)

    checkpoint = torch.load(model_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)
    model.eval()

    return model


def predict_image(model, image: Image.Image):
    image = image.resize((224, 224))

    image_array = np.array(image).astype(np.float32) / 255.0

    image_array = (
        image_array
        - np.array([0.485, 0.456, 0.406])
    ) / np.array([0.229, 0.224, 0.225])

    tensor = (
        torch.from_numpy(image_array.transpose(2, 0, 1))
        .unsqueeze(0)
        .float()
    )

    tensor = tensor.to(next(model.parameters()).device)

    with torch.no_grad():
        outputs = model(tensor)
        scores = torch.softmax(outputs, dim=1)[0].cpu().numpy()
        prediction = int(np.argmax(scores))

    return prediction, scores, tensor


def generate_segmentation(model, image):
    image = image.resize((256, 256))

    image = np.array(image).astype(np.float32) / 255.0

    tensor = (
        torch.from_numpy(image.transpose(2, 0, 1))
        .unsqueeze(0)
        .float()
    )

    tensor = tensor.to(next(model.parameters()).device)

    with torch.no_grad():
        output = model(tensor)
        mask = torch.sigmoid(output)[0, 0].cpu().numpy()

    return mask


def corruption_preview(image, corruption_type):
    image = image.astype(np.float32) / 255.0

    if corruption_type == "Noise":
        noise = np.random.normal(0, 0.05, image.shape)
        corrupted = np.clip(image + noise, 0, 1)

    elif corruption_type == "Blur":
        corrupted = cv2.GaussianBlur(
            (image * 255).astype(np.uint8),
            (9, 9),
            0,
        ).astype(np.float32) / 255.0

    else:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), 25]

        _, buffer = cv2.imencode(
            ".jpg",
            (image * 255).astype(np.uint8),
            params,
        )

        corrupted = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        corrupted = cv2.cvtColor(
            corrupted,
            cv2.COLOR_BGR2RGB,
        ).astype(np.float32) / 255.0

    return (corrupted * 255).astype(np.uint8)


def main():
    st.set_page_config(
        page_title="Explainable Brain Tumor Detection",
        layout="wide",
    )

    st.title("🧠 Explainable Brain Tumor Detection")

    uploaded_file = st.file_uploader(
        "Upload MRI Image",
        type=["png", "jpg", "jpeg", "bmp"],
    )

    classifier_path = os.path.join(
        ROOT_DIR,
        "models",
        "classifier.pth",
    )

    segmentation_path = os.path.join(
        ROOT_DIR,
        "models",
        "unet.pth",
    )

    classifier = None
    segmentation = None

    if os.path.exists(classifier_path):
        classifier = load_classification_model(classifier_path)

    if os.path.exists(segmentation_path):
        segmentation = load_segmentation_model(segmentation_path)

    if uploaded_file is None:
        st.info("Upload an MRI image to begin.")
        return

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded MRI")

    if classifier is None:
        st.error("classifier.pth not found.")
        return

    prediction, scores, tensor = predict_image(classifier, image)

    st.subheader("Prediction")

    if prediction < len(CLASS_NAMES):
        st.success(CLASS_NAMES[prediction])
    else:
        st.error("Unknown class index")

    st.metric(
        "Confidence",
        f"{scores[prediction]*100:.2f}%"
    )

    st.subheader("Class Probabilities")

    for cls, prob in zip(CLASS_NAMES, scores):
        st.write(f"{cls}: {prob*100:.2f}%")

    with st.expander("Uncertainty Estimation"):
        result = predict_with_uncertainty(
            classifier,
            tensor,
            n_samples=12,
        )
        st.write(result)

    with st.expander("Grad-CAM Explanation"):
        gradcam = GradCAM(
            classifier,
            classifier.layer4[-1],
        )

        heatmap = gradcam.generate(
            tensor,
            class_idx=prediction,
        )

        overlay = overlay_heatmap(
            np.array(image.resize((224, 224))),
            heatmap,
        )

        st.image(overlay)

    if segmentation is not None:
        st.subheader("Tumor Segmentation")

        mask = generate_segmentation(
            segmentation,
            image,
        )

        st.image(
            (mask > 0.5).astype(np.uint8) * 255,
            caption="Predicted Mask",
        )

    corruption = st.selectbox(
        "Image Corruption Test",
        [
            "None",
            "Noise",
            "Blur",
            "JPEG compression",
        ],
    )

    if corruption != "None":
        corrupted = corruption_preview(
            np.array(image),
            corruption,
        )

        st.image(
            corrupted,
            caption=corruption,
        )


if __name__ == "__main__":
    main()