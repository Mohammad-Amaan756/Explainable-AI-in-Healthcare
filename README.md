# Explainable and Robust Brain Tumor Detection using MRI Scans

This repository provides a complete pipeline for brain tumor detection and segmentation from MRI images with a focus on explainability and robustness.

## Project Overview

- **Classification**: Tumor / No Tumor detection using a CNN-based ResNet backbone.
- **Segmentation**: Tumor region segmentation using a U-Net architecture.
- **Explainability**: Grad-CAM heatmaps visualize model attention on MRI scans.
- **Robustness**: Noise, blur, and compression artifacts are applied to simulate real-world distortions.
- **Deployment (optional)**: A Streamlit app for image upload, prediction, and explanation.

## Repository Structure

- `data/` - placeholder for datasets
- `models/` - saved model weights and checkpoints
- `notebooks/` - demo notebook and prototyping
- `src/` - core modules
  - `preprocessing.py` - data loading, datasets, transforms
  - `train.py` - training logic for classification and segmentation
  - `evaluate.py` - evaluation metrics and reporting
  - `explainability.py` - Grad-CAM and uncertainty estimation
  - `robustness.py` - corruption functions and robustness evaluation
- `app/` - Streamlit user interface
- `results/` - output reports, plots, and heatmaps
- `README.md` - project explanation and usage

## Setup

1. Create a Python environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Prepare your data directories.

### Classification dataset layout

```
data/classification/tumor/*.jpg
 data/classification/no_tumor/*.jpg
```

### Segmentation dataset layout

```
data/segmentation/images/*.png
 data/segmentation/masks/*.png
```

## Usage

### Train classification model

```bash
python src/train.py --task classification --data-dir data/classification --epochs 10 --batch-size 16 --output models/classifier.pth
```

### Train segmentation model

```bash
python src/train.py --task segmentation --image-dir data/segmentation/images --mask-dir data/segmentation/masks --epochs 20 --batch-size 8 --output models/unet.pth
```

### Evaluate models

```bash
python src/evaluate.py --task classification --model-path models/classifier.pth --data-dir data/classification
```

```bash
python src/evaluate.py --task segmentation --model-path models/unet.pth --image-dir data/segmentation/images --mask-dir data/segmentation/masks
```

### Run robustness evaluation

```bash
python src/robustness.py --model-path models/classifier.pth --data-dir data/classification
```

### Start Streamlit UI

```bash
streamlit run app/streamlit_app.py
```

## Future Enhancements

- Add 3D volumetric MRI processing
- Add multi-class tumor segmentation
- Add advanced explainability such as SHAP and integrated gradients
- Add clinical decision support integration
