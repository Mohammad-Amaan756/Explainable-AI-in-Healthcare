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

from src.explainability import GradCAM, overlay_heatmap, predict_with_uncertainty
from src.train import get_classifier, UNet

CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
IMAGENET_MEAN, IMAGENET_STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])

CSS = """
<style>
.stApp{background:linear-gradient(180deg,#0B1120 0%,#0F1B2E 100%);color:#E6EDF3}
.block-container{padding-top:2rem;padding-bottom:2rem}
h1{color:#4CC9F0;font-weight:800;letter-spacing:.5px}
h2,h3{color:#9FD8FF}
section[data-testid="stSidebar"]{background:#0C1424;border-right:1px solid #1E2A3F}
div[data-testid="stMetric"]{background:#131E33;border:1px solid #1E2A3F;border-radius:14px;padding:18px;box-shadow:0 4px 14px rgba(0,0,0,.35)}
div[data-testid="stMetricValue"]{color:#4CC9F0}
.diagnosis-card{background:#131E33;border:1px solid #1E2A3F;border-radius:16px;padding:22px 24px;box-shadow:0 4px 18px rgba(0,0,0,.4);margin-bottom:16px}
.prob-row{margin-bottom:10px}
.prob-label{display:flex;justify-content:space-between;font-size:.9rem;color:#C7D6E8;margin-bottom:4px}
.prob-track{width:100%;height:10px;background:#0C1424;border-radius:6px;overflow:hidden}
.prob-fill{height:100%;border-radius:6px;background:linear-gradient(90deg,#4CC9F0,#4361EE);transition:width .6s ease-in-out}
.stButton>button{background:linear-gradient(90deg,#4361EE,#4CC9F0);color:#fff;border:none;border-radius:8px;font-weight:600}
hr{border-color:#1E2A3F}
div[data-testid="stHorizontalBlock"]{flex-wrap:wrap;gap:1.25rem}
div[data-testid="column"]{min-width:300px;flex:1 1 320px}
img{max-width:100%;height:auto}
@media (max-width:900px){.block-container{padding-left:1.25rem;padding-right:1.25rem}}
@media (max-width:768px){
  .block-container{padding-top:1rem;padding-left:1rem;padding-right:1rem}
  h1{font-size:1.6rem} h2,h3{font-size:1.1rem} p,span,.prob-label{font-size:.85rem}
  div[data-testid="stMetric"]{padding:12px} div[data-testid="stMetricValue"]{font-size:1.4rem}
  .diagnosis-card{padding:16px 18px}
}
@media (max-width:480px){
  h1{font-size:1.3rem} .stCaption{font-size:.75rem}
  div[data-testid="column"]{min-width:100%;flex:1 1 100%}
  div[data-testid="stMetric"]{padding:10px}
}
</style>
"""


@st.cache_resource
def load_classification_model(model_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_classifier(num_classes=len(CLASS_NAMES), pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device)["model_state_dict"])
    return model.to(device).eval()


@st.cache_resource
def load_segmentation_model(model_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=3, out_channels=1)
    model.load_state_dict(torch.load(model_path, map_location=device)["model_state_dict"])
    return model.to(device).eval()


def _to_tensor(image: Image.Image, size: int, normalize: bool) -> torch.Tensor:
    arr = np.array(image.resize((size, size))).astype(np.float32) / 255.0
    if normalize:
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).float()


def predict_image(model, image: Image.Image):
    tensor = _to_tensor(image, 224, normalize=True).to(next(model.parameters()).device)
    with torch.no_grad():
        scores = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
    return int(np.argmax(scores)), scores, tensor


def generate_segmentation(model, image: Image.Image):
    tensor = _to_tensor(image, 256, normalize=False).to(next(model.parameters()).device)
    with torch.no_grad():
        return torch.sigmoid(model(tensor))[0, 0].cpu().numpy()


def corruption_preview(image: np.ndarray, corruption_type: str) -> np.ndarray:
    image = image.astype(np.float32) / 255.0
    if corruption_type == "Noise":
        corrupted = np.clip(image + np.random.normal(0, 0.05, image.shape), 0, 1)
    elif corruption_type == "Blur":
        corrupted = cv2.GaussianBlur((image * 255).astype(np.uint8), (9, 9), 0).astype(np.float32) / 255.0
    else:
        _, buf = cv2.imencode(".jpg", (image * 255).astype(np.uint8), [int(cv2.IMWRITE_JPEG_QUALITY), 25])
        corrupted = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return (corrupted * 255).astype(np.uint8)


def render_probability_bars(scores):
    for idx in np.argsort(scores)[::-1]:
        cls, pct = CLASS_NAMES[idx], scores[idx] * 100
        st.markdown(
            f'<div class="prob-row"><div class="prob-label"><span>{cls}</span><span>{pct:.2f}%</span></div>'
            f'<div class="prob-track"><div class="prob-fill" style="width:{pct:.2f}%;"></div></div></div>',
            unsafe_allow_html=True,
        )


def render_sidebar():
    with st.sidebar:
        st.header("🏥 Hospital Dashboard")
        st.success("System Status: Online")
        st.write("### Model")
        for line in ["✔ ResNet18 Classifier", "✔ Grad-CAM Explainability", "✔ U-Net Tumor Segmentation", "✔ MRI Classification"]:
            st.write(line)
        st.write("---")
        st.info("Upload an MRI scan to receive an AI-assisted diagnosis.")
        return st.file_uploader("Upload MRI Image", type=["png", "jpg", "jpeg", "bmp"])


def reliability_label(confidence: float):
    if confidence > 0.95:
        st.success("Reliability: Very High")
    elif confidence > 0.80:
        st.info("Reliability: High")
    elif confidence > 0.60:
        st.warning("Reliability: Medium")
    else:
        st.error("Reliability: Low")


def render_diagnosis(classifier, segmentation, image, prediction, scores, tensor):
    st.subheader("Diagnosis")
    st.markdown('<div class="diagnosis-card">', unsafe_allow_html=True)
    st.success(f"Predicted Class: **{CLASS_NAMES[prediction]}**")
    st.metric("Confidence", f"{scores[prediction] * 100:.2f}%")
    reliability_label(scores[prediction])
    st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("Class Probabilities")
    render_probability_bars(scores)

    with st.expander("Uncertainty Estimation"):
        st.write(predict_with_uncertainty(classifier, tensor, n_samples=12))

    with st.expander("Grad-CAM Explanation"):
        gradcam = GradCAM(classifier, classifier.layer4[-1])
        heatmap = gradcam.generate(tensor, class_idx=prediction)
        st.image(overlay_heatmap(np.array(image.resize((224, 224))), heatmap))

    if segmentation is not None:
        st.subheader("Tumor Segmentation")
        mask = generate_segmentation(segmentation, image)
        st.image((mask > 0.5).astype(np.uint8) * 255, caption="Predicted Mask")

    st.subheader("Robustness Check")
    corruption = st.selectbox("Image Corruption Test", ["None", "Noise", "Blur", "JPEG compression"])
    if corruption != "None":
        st.image(corruption_preview(np.array(image), corruption), caption=corruption)


def main():
    st.set_page_config(page_title="AI Brain Tumor Diagnosis", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)

    st.title("🏥 AI-Assisted Brain Tumor Diagnosis")
    st.caption("Explainable Artificial Intelligence for Brain Tumor Classification using MRI Images")
    st.markdown("---")

    uploaded_file = render_sidebar()

    classifier_path = os.path.join(ROOT_DIR, "models", "classifier.pth")
    segmentation_path = os.path.join(ROOT_DIR, "models", "unet.pth")
    classifier = load_classification_model(classifier_path) if os.path.exists(classifier_path) else None
    segmentation = load_segmentation_model(segmentation_path) if os.path.exists(segmentation_path) else None

    if uploaded_file is None:
        st.info("Upload an MRI image to begin.")
        st.stop()

    image = Image.open(uploaded_file).convert("RGB")

    if classifier is None:
        st.error("classifier.pth not found.")
        st.stop()

    prediction, scores, tensor = predict_image(classifier, image)

    left_col, right_col = st.columns([1, 1], gap="large")
    with left_col:
        st.subheader("🧠 Uploaded MRI")
        st.image(image, use_container_width=True)
    with right_col:
        render_diagnosis(classifier, segmentation, image, prediction, scores, tensor)


if __name__ == "__main__":
    main()