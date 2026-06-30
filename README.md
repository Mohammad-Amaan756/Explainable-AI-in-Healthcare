# Explainable Brain Tumor Detection using MRI Scans

A comprehensive machine learning system for detecting and segmenting brain tumors from MRI images with **explainability** and **robustness** analysis.

## 🎯 Project Overview

This project combines **deep learning** and **explainable AI (XAI)** to create a transparent, robust system for brain tumor detection:

- **Classification**: Detects presence/absence of tumors using ResNet18
- **Segmentation**: Localizes tumor regions using U-Net architecture
- **Explainability**: Visualizes model decisions via Grad-CAM heatmaps
- **Uncertainty Estimation**: Quantifies prediction confidence using MC Dropout
- **Robustness**: Tests resilience to real-world image corruptions
- **Web UI**: Interactive Streamlit application for deployment

## 📊 Supported Tumor Types

- **Glioma** - Most common malignant brain tumor
- **Meningioma** - Tumor of the membrane surrounding the brain
- **Pituitary** - Tumor of the pituitary gland
- **No Tumor** - Healthy brain scans

## 🏗️ Architecture

### Classification Model (ResNet18)
- Pretrained on ImageNet
- Fine-tuned for tumor classification
- Dropout layers for uncertainty estimation

### Segmentation Model (U-Net)
- Encoder-decoder architecture with skip connections
- Produces binary segmentation masks
- Optimized for medical imaging

## 📦 Requirements

### Python Dependencies
```
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.8.0
scikit-learn>=1.3.0
streamlit>=1.25.0
Pillow>=10.0.0
```

## 🚀 Installation & Quick Start

See **[QUICKSTART.md](QUICKSTART.md)** for detailed setup instructions and usage examples.

### Quick Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Train classification model
python src/train.py --task classification \
  --data-dir data/classification/Training \
  --output models/classifier.pth --epochs 20 --pretrained

# Launch interactive app
streamlit run app/streamlit_app.py
```

## 📚 Core Modules

### `src/train.py` - Training Pipelines
Train classification or segmentation models with flexible hyperparameters.

**Usage:**
```bash
python src/train.py --task classification --help
```

**Classes:**
- `UNet` - Segmentation architecture
- `get_classifier()` - Classification model factory

### `src/evaluate.py` - Model Evaluation
Compute accuracy, precision, recall, F1-score, and Dice coefficient.

**Usage:**
```bash
python src/evaluate.py --task classification --model-path models/classifier.pth
```

### `src/preprocessing.py` - Data Handling
Load, preprocess, and augment medical images.

**Classes:**
- `TumorClassificationDataset` - Classification data loader
- `TumorSegmentationDataset` - Segmentation data loader with augmentation

### `src/explainability.py` - Interpretability
Generate visual explanations and uncertainty estimates.

**Key Functions:**
- `GradCAM` - Class activation mapping
- `predict_with_uncertainty()` - MC Dropout inference
- `overlay_heatmap()` - Visualization helper

### `src/robustness.py` - Robustness Testing
Evaluate model resilience to image corruptions.

**Corruption Types:**
- Gaussian noise
- Blur
- JPEG compression

**Usage:**
```bash
python src/robustness.py --model-path models/classifier.pth
```

### `app/streamlit_app.py` - Interactive Web UI
Upload images and get real-time predictions with explanations.

**Features:**
- Classification with confidence scores
- Grad-CAM explanation heatmaps
- Uncertainty quantification
- Segmentation visualization
- Corruption effect preview

**Launch:**
```bash
streamlit run app/streamlit_app.py
```

## 📂 Project Structure

```
Explainable AI in Healthcare/
├── README.md                 # Full documentation
├── QUICKSTART.md             # Quick start guide
├── requirements.txt          # Dependencies
├── src/                      # Core library
│   ├── train.py             # Model training
│   ├── evaluate.py          # Evaluation metrics
│   ├── preprocessing.py     # Data loading
│   ├── explainability.py    # Grad-CAM & uncertainty
│   └── robustness.py        # Robustness tests
├── app/
│   └── streamlit_app.py     # Web UI
├── data/
│   ├── classification/      # Tumor/No Tumor images
│   └── segmentation/        # Images & masks
├── models/                  # Trained checkpoints
├── results/                 # Outputs & visualizations
└── notebooks/
    ├── demo.ipynb
    └── comprehensive_demo.ipynb
```

## 🧠 Explainability Methods

### Grad-CAM (Gradient-weighted Class Activation Mapping)
Generates visual heatmaps showing which regions influenced the prediction:

```python
from src.explainability import GradCAM

gradcam = GradCAM(model, model.layer4[-1])
heatmap = gradcam.generate(image_tensor, class_idx=1)
```

### MC Dropout Uncertainty
Quantifies prediction confidence via multiple stochastic forward passes:

```python
from src.explainability import predict_with_uncertainty

result = predict_with_uncertainty(model, image_tensor, n_samples=20)
print(f"Uncertainty: {result['uncertainty']:.4f}")
```

## 📊 Training & Evaluation

### Train Classification
```bash
python src/train.py \
  --task classification \
  --data-dir data/classification/Training \
  --output models/classifier.pth \
  --epochs 20 --batch-size 32 --lr 1e-4 --pretrained
```

### Train Segmentation
```bash
python src/train.py \
  --task segmentation \
  --image-dir data/segmentation/images \
  --mask-dir data/segmentation/masks \
  --output models/unet.pth \
  --epochs 20 --batch-size 16
```

### Evaluate Models
```bash
python src/evaluate.py \
  --task classification \
  --model-path models/classifier.pth \
  --data-dir data/classification/Testing
```

### Test Robustness
```bash
python src/robustness.py \
  --model-path models/classifier.pth \
  --data-dir data/classification/Testing
```

## 📖 Jupyter Notebooks

### `notebooks/comprehensive_demo.ipynb`
Complete walkthrough including:
- Data loading and visualization
- Classification predictions
- Grad-CAM explanations
- Uncertainty estimation
- Robustness testing
- Segmentation results

Start with: `jupyter notebook notebooks/comprehensive_demo.ipynb`

## 🔧 Advanced Usage

### Custom Training Loop
```python
import torch
from src.train import get_classifier, train_epoch, validate_classification
from src.preprocessing import build_classification_datasets

# Load data
train_dataset, val_dataset, classes = build_classification_datasets('data/classification/Training')
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)

# Setup model
model = get_classifier(num_classes=len(classes), pretrained=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Train
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(20):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = validate_classification(model, val_loader, criterion, device)
    print(f'Epoch {epoch} - Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}')
```

### Batch Inference
```python
from src.preprocessing import TumorClassificationDataset
from torch.utils.data import DataLoader

dataset = TumorClassificationDataset(filepaths, labels)
loader = DataLoader(dataset, batch_size=32)

predictions = []
with torch.no_grad():
    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        predictions.extend(preds)
```

## 🐛 Troubleshooting

### CUDA Memory Error
```bash
# Reduce batch size
python src/train.py --batch-size 8 ...

# Or use CPU
device = torch.device('cpu')
```

### Data Not Found
- Verify directory structure matches expected layout
- Ensure image extensions are supported (.jpg, .png, .tif)
- For segmentation: check `_mask` suffix on mask files

### Model Not Loading
```python
import os
import torch

# Verify file exists
assert os.path.exists('models/classifier.pth')

# Check checkpoint contents
checkpoint = torch.load('models/classifier.pth')
print(checkpoint.keys())  # Should contain 'model_state_dict'
```

## 📈 Expected Results

| Task | Metric | Target |
|------|--------|--------|
| Classification | Accuracy | 85-95% |
| Classification | Precision | 85-95% |
| Segmentation | Dice Score | 0.80-0.95 |
| Robustness (Noise) | Accuracy | 70-85% |
| Robustness (Blur) | Accuracy | 75-90% |

## 🔗 References

- **Grad-CAM**: [Selvaraju et al. 2016](https://arxiv.org/abs/1610.02055)
- **U-Net**: [Ronneberger et al. 2015](https://arxiv.org/abs/1505.04597)
- **ResNet**: [He et al. 2015](https://arxiv.org/abs/1512.03385)
- **MC Dropout**: [Gal & Ghahramani 2016](https://arxiv.org/abs/1506.02142)
- **TCGA**: [NCI GDC Portal](https://portal.gdc.cancer.gov/)

## ⚠️ Medical Disclaimer

This project is for **educational and research purposes only**. Do not use for clinical diagnosis without proper medical validation and regulatory approval. Always consult qualified medical professionals.

## 📝 Citation

```bibtex
@software{xai_brain_tumors_2024,
  title={Explainable Brain Tumor Detection using MRI Scans},
  year={2024}
}
```

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- TCGA Program for medical imaging dataset
- PyTorch team for deep learning framework
- Streamlit for web UI framework
- Community for feedback and contributions

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make improvements
4. Submit a pull request

## 📞 Support

For issues or questions:
- Check [QUICKSTART.md](QUICKSTART.md) for setup help
- Review example notebooks in `notebooks/`
- Open an issue on GitHub

---

**Version**: 1.0.0 | **Updated**: 2024 | **Status**: Production Ready
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
