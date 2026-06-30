# Quick Start Guide: Explainable Brain Tumor Detection

This guide will help you get started with training and using the Explainable AI system for brain tumor detection from MRI scans.

## Prerequisites

1. **Python 3.8+** with pip installed
2. **Dataset**: Download/place your data in the appropriate directories:
   - Classification: `data/classification/Training/` and `data/classification/Testing/`
   - Segmentation: `data/segmentation/images/` and `data/segmentation/masks/`

## Installation

```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

## Training Models

### Classification Model (Tumor Detection)

```bash
# Train ResNet18-based classifier
python src/train.py \
    --task classification \
    --data-dir data/classification/Training \
    --output models/classifier.pth \
    --epochs 20 \
    --batch-size 32 \
    --lr 1e-4 \
    --image-size 224 \
    --pretrained
```

**Expected time**: 5-15 minutes (depending on dataset size and GPU)

### Segmentation Model (Tumor Region)

```bash
# Train U-Net for tumor segmentation
python src/train.py \
    --task segmentation \
    --image-dir data/segmentation/images \
    --mask-dir data/segmentation/masks \
    --output models/unet.pth \
    --epochs 20 \
    --batch-size 16 \
    --lr 1e-4 \
    --image-size 256
```

**Expected time**: 10-30 minutes (depending on dataset size and GPU)

## Model Evaluation

### Evaluate Classification

```bash
python src/evaluate.py \
    --task classification \
    --model-path models/classifier.pth \
    --data-dir data/classification/Testing \
    --batch-size 32 \
    --image-size 224
```

### Evaluate Segmentation

```bash
python src/evaluate.py \
    --task segmentation \
    --model-path models/unet.pth \
    --image-dir data/segmentation/images \
    --mask-dir data/segmentation/masks \
    --batch-size 16 \
    --image-size 256
```

## Robustness Testing

Test model robustness against image corruptions (noise, blur, JPEG compression):

```bash
python src/robustness.py \
    --model-path models/classifier.pth \
    --data-dir data/classification/Testing \
    --batch-size 32 \
    --image-size 224
```

This will apply corruptions at increasing severity levels and report accuracy.

## Interactive Web Application

Launch the Streamlit app for interactive inference with explanations:

```bash
streamlit run app/streamlit_app.py
```

The app provides:
- **Classification**: Tumor/No Tumor detection with confidence scores
- **Explanation**: Grad-CAM heatmaps showing which regions influence predictions
- **Uncertainty**: Monte Carlo Dropout estimates of model uncertainty
- **Segmentation**: Optional tumor region visualization
- **Robustness Testing**: Preview of model behavior under image corruption

Navigate to `http://localhost:8501` in your browser.

## Advanced: Using Python API

```python
import torch
from src.train import get_classifier, UNet
from src.explainability import GradCAM, load_image, predict_with_uncertainty

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_classifier(num_classes=2, pretrained=False)
checkpoint = torch.load("models/classifier.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)

# Predict with uncertainty
image_tensor = load_image("path/to/image.jpg")
result = predict_with_uncertainty(model, image_tensor.to(device), n_samples=20)
print(f"Predicted class: {result['predicted_class']}")
print(f"Uncertainty: {result['uncertainty']:.4f}")

# Generate explanation
gradcam = GradCAM(model, model.layer4[-1])
heatmap = gradcam.generate(image_tensor.to(device), class_idx=result['predicted_class'])
```

## Project Structure

```
.
├── README.md                          # Full project documentation
├── QUICKSTART.md                      # This file
├── requirements.txt                   # Python dependencies
├── data/                              # Datasets
│   ├── classification/
│   │   ├── Training/
│   │   └── Testing/
│   └── segmentation/
│       ├── images/
│       ├── masks/
│       └── kaggle_3m/
├── models/                            # Trained model checkpoints
│   ├── classifier.pth                 # Classification model
│   └── unet.pth                       # Segmentation model
├── results/                           # Output visualizations
├── src/                               # Core library
│   ├── __init__.py
│   ├── train.py                       # Training pipelines
│   ├── evaluate.py                    # Evaluation metrics
│   ├── preprocessing.py               # Data loading & transforms
│   ├── explainability.py              # Grad-CAM & uncertainty
│   └── robustness.py                  # Corruption testing
├── app/                               # Streamlit application
│   └── streamlit_app.py              # Interactive UI
└── notebooks/
    ├── demo.ipynb                     # Jupyter notebook demo
    └── data/
```

## Troubleshooting

### CUDA Memory Issues
```bash
# Reduce batch size
python src/train.py --task classification --batch-size 8 ...

# Or use CPU
# In Python: device = torch.device("cpu")
```

### Data Not Found
- Verify paths match your directory structure
- Check that image files have correct extensions (.jpg, .png, .tif, etc.)
- Ensure masks match image filenames with `_mask` suffix

### Model Not Loading
```python
# Check if file exists
import os
print(os.path.exists("models/classifier.pth"))  # Should be True

# Verify checkpoint contents
checkpoint = torch.load("models/classifier.pth")
print(checkpoint.keys())  # Should contain 'model_state_dict'
```

## Next Steps

1. **Prepare your data** in the correct directory structure
2. **Train the classification model** first to establish baselines
3. **Evaluate and tune hyperparameters** for your specific dataset
4. **Test robustness** to understand failure modes
5. **Launch the Streamlit app** for interactive exploration
6. **Iterate and improve** based on results

## Citation & References

If you use this codebase in research, please cite:
```
@article{explainable-brain-tumors,
  title={Explainable AI for Brain Tumor Detection from MRI Scans},
  year={2024}
}
```

## Support

For issues or questions, check:
- `README.md` for detailed documentation
- Python docstrings in source files
- Example Jupyter notebooks in `notebooks/`

