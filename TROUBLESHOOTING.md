# ❓ FAQ & Troubleshooting Guide

## Common Questions & Issues

### 🚀 Installation & Setup

#### Q1: How do I install the project dependencies?

**A:** Use pip to install from requirements.txt:

```bash
# Activate your virtual environment first
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Note**: GPU acceleration requires CUDA 11.0+. CPU-only mode works but is slower.

---

#### Q2: What Python version should I use?

**A:** Python 3.8 or newer is recommended. Tested on Python 3.8, 3.9, 3.10, 3.11.

```bash
python --version  # Should show 3.8+
```

---

#### Q3: Do I need a GPU?

**A:** No, CPU is supported but much slower.

**Performance estimates:**
- **GPU (NVIDIA RTX 3090)**: ~500 images/sec for inference
- **GPU (NVIDIA RTX 2080)**: ~200 images/sec
- **CPU (Intel i7)**: ~5-10 images/sec

---

### 📊 Data & Preprocessing

#### Q4: How should I organize my data?

**A:** Follow this structure:

```
data/
├── classification/
│   ├── Training/
│   │   ├── glioma/
│   │   │   ├── image1.jpg
│   │   │   ├── image2.jpg
│   │   │   └── ...
│   │   ├── meningioma/
│   │   ├── notumor/
│   │   ├── pituitary/
│   │   └── tumor/
│   └── Testing/
│       ├── glioma/
│       ├── meningioma/
│       └── ...
└── segmentation/
    ├── images/
    │   ├── image1.tif
    │   ├── image2.tif
    │   └── ...
    └── masks/
        ├── image1_mask.tif
        ├── image2_mask.tif
        └── ...
```

---

#### Q5: What image formats are supported?

**A:** Supported extensions:
- `.jpg`, `.jpeg` - Standard JPEG
- `.png` - PNG with alpha channel
- `.tif`, `.tiff` - TIFF (medical imaging)
- `.bmp` - Windows bitmap

**Recommendation**: Use PNG or TIFF for medical images (lossless).

---

#### Q6: How many images do I need?

**A:** Minimum recommendations:
- **Classification**: 100+ images per class (500+ total recommended)
- **Segmentation**: 50+ image-mask pairs

**Better results with**:
- 1000+ training images for classification
- 500+ pairs for segmentation
- Balanced classes

---

#### Q7: Should I resize images?

**A:** The code handles resizing automatically:
- **Classification**: 224×224 (ResNet standard)
- **Segmentation**: 256×256 (U-Net standard)

Original image size doesn't matter (within reason: 64×64 to 2048×2048 works).

---

#### Q8: How do I create segmentation masks?

**A:** Ground truth masks must be:
1. **Binary** (0 = background, 255 = tumor)
2. **Same size** as original image or will be resized
3. **Named** with `_mask` suffix: `image1.tif` → `image1_mask.tif`
4. **Grayscale** (single channel)

Tools for creating masks:
- **CVAT**: Free open-source annotation tool
- **Labelbox**: Cloud-based annotation
- **LabelImg**: Simple bounding boxes
- **3D Slicer**: Professional medical imaging

---

### 🏋️ Training

#### Q9: My GPU runs out of memory. What do I do?

**A:** Reduce batch size:

```bash
# Default (batch_size=32)
python src/train.py --task classification --batch-size 16

# If still OOM:
python src/train.py --task classification --batch-size 8

# Or use CPU (very slow):
python src/train.py --task classification --batch-size 16  # CPU will use smaller default
```

**Memory usage estimate**:
- Batch size 8: ~3 GB VRAM
- Batch size 16: ~6 GB VRAM
- Batch size 32: ~12 GB VRAM

---

#### Q10: Training is very slow. Why?

**A:** Check several things:

```bash
# 1. Verify GPU usage
nvidia-smi

# 2. Increase batch size if memory allows
python src/train.py --batch-size 64

# 3. Increase learning rate (faster convergence)
python src/train.py --lr 1e-3

# 4. Use pretrained weights (faster convergence)
python src/train.py --pretrained

# 5. Reduce image size (faster processing)
python src/train.py --image-size 128
```

---

#### Q11: Model is overfitting. How do I fix it?

**A:** Try these regularization techniques:

```bash
# 1. Use data augmentation (automatically applied)
# Already in the code

# 2. Increase validation split
python src/train.py --val-split 0.3  # 30% validation

# 3. Early stopping (manual monitoring)
# Stop training when val_loss stops improving

# 4. Reduce learning rate
python src/train.py --lr 5e-5

# 5. Use more training data
# Add more images to data/classification/Training/
```

---

#### Q12: Model accuracy is low (< 80%). What should I check?

**A:** Debugging checklist:

```bash
# 1. Verify data quality
# Look at sample images manually
# Are labels correct?
# Is data representative?

# 2. Check preprocessing
# Verify normalization (0-1 or ImageNet standard)
# Check image resizing
# Look for orientation issues

# 3. Increase training time
python src/train.py --epochs 50

# 4. Use pretrained weights
python src/train.py --pretrained

# 5. Adjust learning rate
python src/train.py --lr 1e-3

# 6. Increase batch size for more stable gradients
python src/train.py --batch-size 64
```

---

### 📈 Evaluation

#### Q13: How do I know if my model is good?

**A:** Use these metrics:

**Classification**:
- **Accuracy**: (TP + TN) / (TP + TN + FP + FN) - Overall correctness
- **Precision**: TP / (TP + FP) - False positive rate
- **Recall**: TP / (TP + FN) - False negative rate (critical in medical)
- **F1-score**: Harmonic mean - Balanced metric

**Targets for medical imaging**:
- Accuracy: 85%+
- Recall: 90%+ (catch all true cases)
- Precision: 80%+ (minimize false alarms)

**Segmentation**:
- **Dice Score**: 0.80+ (95%+ is excellent)
- **IoU (Intersection over Union)**: 0.70+
- **Hausdorff Distance**: < 10 pixels (spatial accuracy)

---

#### Q14: How do I evaluate on my test set?

**A:**

```bash
# Classification
python src/evaluate.py \
  --task classification \
  --model-path models/classifier.pth \
  --data-dir data/classification/Testing

# Segmentation
python src/evaluate.py \
  --task segmentation \
  --model-path models/unet.pth \
  --image-dir data/segmentation/images \
  --mask-dir data/segmentation/masks
```

---

#### Q15: The test metrics are much worse than training. Why?

**A:** This usually means **overfitting**. The model memorized training data.

Solutions:
1. **More training data** - Most effective
2. **Data augmentation** - Already applied
3. **Simpler model** - Reduce epochs
4. **Regularization** - Lower learning rate
5. **Dropout** - Already in model
6. **Cross-validation** - Validate on different subsets

---

### 🎯 Explainability

#### Q16: How do I interpret the Grad-CAM heatmaps?

**A:** Grad-CAM shows which image regions influenced the prediction:

- **Red/Hot areas**: Most important for the prediction
- **Blue/Cool areas**: Less important
- **Sharp focus**: Model confident in specific region
- **Diffuse**: Model uncertain or using global features

**Good signs**:
✅ Heatmap concentrates on tumor region  
✅ Activation in anatomically correct area  
✅ Sharp boundaries

**Bad signs**:
❌ Heatmap on irrelevant areas  
❌ Entire image highlighted  
❌ Completely diffuse activation

---

#### Q17: What does uncertainty score mean?

**A:** Uncertainty (from MC Dropout) shows model confidence:

- **< 0.05**: Very confident (trust prediction)
- **0.05 - 0.15**: Moderately confident (usually reliable)
- **> 0.15**: Low confidence (verify prediction)

High uncertainty indicates:
- Ambiguous image
- Out-of-distribution input
- Uncertain medical case (should get second opinion)

---

#### Q18: Can I get confidence scores?

**A:** Yes! The Streamlit app shows confidence directly.

In code:
```python
from src.explainability import predict_with_uncertainty

result = predict_with_uncertainty(model, image_tensor, n_samples=20)
print(f"Confidence: {result['mean_probability'][predicted_class]:.2%}")
print(f"Uncertainty: {result['uncertainty']:.4f}")
```

---

### 🔧 Robustness & Deployment

#### Q19: How do I test if my model is robust?

**A:**

```bash
python src/robustness.py \
  --model-path models/classifier.pth \
  --data-dir data/classification/Testing
```

This tests:
- **Gaussian noise** (slight artifacts)
- **Blur** (out-of-focus images)
- **JPEG compression** (low-quality scans)

**Expected results**:
- Clean images: 90%+ accuracy
- Light corruption: 85%+ accuracy
- Heavy corruption: 70%+ accuracy

Worse performance indicates need for:
1. More training data
2. Data augmentation
3. Different architecture
4. Better preprocessing

---

#### Q20: How do I deploy to production?

**A:** Use Streamlit for web deployment:

```bash
# Local deployment
streamlit run app/streamlit_app.py

# Cloud deployment (Streamlit Cloud)
# 1. Push code to GitHub
# 2. Connect GitHub repo to Streamlit Cloud
# 3. Deploy (automatic updates on push)

# Docker deployment
docker build -t brain-tumor-detection .
docker run -p 8501:8501 brain-tumor-detection
```

---

### 🐛 Common Errors

#### Error: "FileNotFoundError: data/classification/Training"

**Solution**: Create the directory structure:

```bash
mkdir -p data/classification/Training/tumor
mkdir -p data/classification/Training/notumor
mkdir -p data/classification/Testing/tumor
mkdir -p data/classification/Testing/notumor
# Add images to these directories
```

---

#### Error: "RuntimeError: CUDA out of memory"

**Solution**: Reduce batch size or use CPU:

```bash
# Reduce batch size
python src/train.py --batch-size 8

# Or use CPU explicitly
# Modify code: device = torch.device("cpu")
```

---

#### Error: "AttributeError: 'module' object has no attribute 'TumorClassificationDataset'"

**Solution**: This is fixed in the current version. If you have an old version:

```bash
# Update to latest
git pull
# Or reinstall from scratch
```

---

#### Error: "ValueError: No image-mask pairs found"

**Solution**: Verify segmentation mask naming:

```bash
# Correct format:
images/         masks/
image1.tif  →   image1_mask.tif
image2.tif  →   image2_mask.tif

# Incorrect:
image1.tif  →   mask_image1.tif  ❌
image1.tif  →   image1.tif       ❌ (same name)
```

---

#### Error: "AssertionError: Expected 3 dimensions, got 4"

**Solution**: Ensure images are loaded in RGB format:

```python
# Correct
img = Image.open(path).convert("RGB")  # ✅ RGB = 3 channels

# Also check for:
# - Grayscale images (convert to RGB)
# - Batch images (remove batch dimension)
# - Wrong format (check file type)
```

---

### 💡 Tips & Best Practices

#### Tip 1: Always use a validation set

```bash
# Default 80/20 split is good
# For small datasets: 70/30
# For large datasets: 90/10
python src/train.py --val-split 0.3
```

---

#### Tip 2: Monitor training progress

```bash
# Check logs for:
# ✅ Loss decreasing
# ✅ Accuracy increasing
# ❌ Loss increasing (stop and reduce LR)
# ❌ Validation loss increasing (overfitting)
```

---

#### Tip 3: Save multiple checkpoints

```bash
# Modify src/train.py to save every N epochs:
# if epoch % 5 == 0:
#     torch.save({...}, f"models/checkpoint_epoch{epoch}.pth")
```

---

#### Tip 4: Use test set only once

Save test set for final evaluation only:
1. Train on training data
2. Tune hyperparameters on validation data
3. Evaluate once on test data (don't tune further)

---

#### Tip 5: Document your experiments

Create a log file:
```bash
Date, Task, Epochs, Batch Size, LR, Accuracy, Notes
2024-01-15, classification, 20, 32, 1e-4, 0.92, Pretrained ResNet
2024-01-16, classification, 50, 16, 1e-3, 0.94, Longer training
```

---

### 📞 Additional Help

#### Getting Help
1. **Check logs** - Python error messages are very informative
2. **Review QUICKSTART.md** - Covers most setup issues
3. **Run demo notebook** - `notebooks/comprehensive_demo.ipynb`
4. **Test with minimal example** - Single image test
5. **Read docstrings** - `python -c "from src.train import get_classifier; help(get_classifier)"`

#### Still stuck?
1. Check project structure again
2. Verify all dependencies installed
3. Test with simpler dataset first
4. Enable debug mode (add `print()` statements)
5. Check if issue is reproducible

---

## 📚 Quick Reference

### Command Cheatsheet

```bash
# Setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Training
python src/train.py --task classification --pretrained
python src/train.py --task segmentation

# Evaluation
python src/evaluate.py --task classification --model-path models/classifier.pth
python src/evaluate.py --task segmentation --model-path models/unet.pth

# Robustness
python src/robustness.py --model-path models/classifier.pth

# Deployment
streamlit run app/streamlit_app.py

# Jupyter
jupyter notebook notebooks/comprehensive_demo.ipynb
```

### Useful Flags

```bash
--help              Show all options
--epochs 50         Train for 50 epochs (default: 20)
--batch-size 16     Use batch size 16 (default: 32)
--lr 1e-4          Set learning rate (default: 1e-4)
--pretrained        Use ImageNet pretrained weights
--image-size 256    Resize to 256×256
--val-split 0.2     Use 20% validation data
--seed 42           Set random seed for reproducibility
```

---

**Version**: 1.0.0  
**Last Updated**: 2024  
**Questions?** Check README.md and QUICKSTART.md first!
