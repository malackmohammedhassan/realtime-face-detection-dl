# Training Scratch CNN: Quick Start Guide

## Overview

This guide walks you through training a from-scratch face vs non-face CNN classifier.

---

## 1. Prepare Your Dataset

### Directory Structure

```
data/
├── train/
│   ├── face/      # ~500-1000 face images
│   └── non_face/  # ~500-1000 non-face images
└── val/
    ├── face/      # ~100-200 face images
    └── non_face/  # ~100-200 non-face images
```

### 2.1 Getting Face Images

**Option A: Use Your Photos**

```powershell
# Create face directory
mkdir data\train\face

# Copy your face photos (64x64 recommended, but any size works)
copy your_photo.jpg data\train\face\
```

**Option B: Use Public Datasets**

- [Labeled Faces in the Wild (LFW)](http://vis-www.cs.umass.edu/lfw/)
- [WIDER-FACE](http://shuoyang1213.me/WIDERFACE/)
- [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

**Option C: Generate from MTCNN Detections**

```python
# Use existing MTCNN detector to automatically crop and save faces
python scripts/extract_faces_from_images.py \
    --input_dir raw_photos/ \
    --output_dir data/train/face/
```

### 2.2 Getting Non-Face Images

Popular sources for non-face (background) images:

- [COCO Dataset](https://cocodataset.org/) (objects, scenes)
- Landscape photos
- Building/architecture photos
- Nature images (trees, plants, sky)
- Man-made objects (cars, trees, buildings)

**Key point**: Non-faces should NOT contain visible human faces!

```powershell
# Create non-face directory
mkdir data\train\non_face

# Copy background images
copy background_*.jpg data\train\non_face\
```

### 2.3 Split Dataset

```powershell
# Example: 80% train, 20% val

# For faces
# Move ~20% of data\train\face\ images to data\val\face\

# For non-faces
# Move ~20% of data\train\non_face\ images to data\val\non_face\
```

**Final dataset size guidelines:**

- **Minimal**: 100 faces + 100 non-faces
- **Good**: 500 faces + 500 non-faces
- **Large**: 2000+ faces + 2000+ non-faces
- **Balanced**: Equal number of faces and non-faces

---

## 2. Train the Model

### Basic Training

```powershell
# Activate virtual environment (if not already)
.\venv\Scripts\activate

# Start training with default settings
python scripts/train_scratch_cnn.py
```

**What happens:**

```
Loading datasets...
Loaded 500 images from train set (250 faces, 250 non-faces)
Loaded 100 images from val set (50 faces, 50 non-faces)
Creating model...
Model parameters: 227,536
Starting training...
Epochs: 50
Batch size: 32
Learning rate: 0.001

Epoch [  1/ 50] | Train Loss: 0.6902, Acc: 0.4844 | Val Loss: 0.6845, Acc: 0.5200
Epoch [  2/ 50] | Train Loss: 0.5432, Acc: 0.7188 | Val Loss: 0.4321, Acc: 0.8100
...
✓ Best model saved (Acc: 0.9340)
```

### Advanced Training (Custom Parameters)

```powershell
# Higher learning rate, more epochs
python scripts/train_scratch_cnn.py \
    --epochs 100 \
    --lr 0.0005 \
    --batch_size 64 \
    --weight_decay 1e-4

# Slower training but potentially better accuracy
```

### Training Parameters Explained

| Parameter        | Default | Description                 |
| ---------------- | ------- | --------------------------- |
| `--epochs`       | 50      | Maximum training epochs     |
| `--batch_size`   | 32      | Images per gradient update  |
| `--lr`           | 0.001   | Learning rate (step size)   |
| `--weight_decay` | 1e-4    | L2 regularization strength  |
| `--img_size`     | 64      | Input image size            |
| `--data_dir`     | data    | Path to dataset             |
| `--output_dir`   | models  | Where to save trained model |

### Parameter Tuning Tips

**If validation accuracy is low:**

```powershell
# Option 1: Train longer with early stopping
python scripts/train_scratch_cnn.py --epochs 200

# Option 2: Lower the learning rate for stability
python scripts/train_scratch_cnn.py --lr 0.0001

# Option 3: Increase batch size for better gradient estimates
python scripts/train_scratch_cnn.py --batch_size 64
```

**If training is too slow:**

```powershell
# Reduce batch size (less computation per step)
python scripts/train_scratch_cnn.py --batch_size 16

# Increase stride in detection later
```

**If overfitting (large train-val gap):**

```powershell
# Increase data augmentation (automatic in training data)
# Or add more non-face images
# Or increase epochs (early stopping helps)
python scripts/train_scratch_cnn.py --epochs 100
```

---

## 3. Monitor Training

### During Training

The script outputs training progress:

```
Epoch [  1/ 50] | Train Loss: 0.6902, Acc: 0.4844 | Val Loss: 0.6845, Acc: 0.5200
Epoch [  5/ 50] | Train Loss: 0.2103, Acc: 0.9100 | Val Loss: 0.2540, Acc: 0.8950
Epoch [ 10/ 50] | Train Loss: 0.1023, Acc: 0.9600 | Val Loss: 0.1854, Acc: 0.9230
...
✓ Best model saved (Acc: 0.9340)
Early stopping triggered after 35 epochs
```

### What the Metrics Mean

- **Train Loss**: How well model fits training data (lower = better)
- **Train Acc**: Percentage correct on training set (0-100%)
- **Val Loss**: How well model generalizes (lower = better)
- **Val Acc**: Percentage correct on validation set (0-100%)

**Healthy training pattern:**

```
Loss:  0.69 → 0.50 → 0.30 → 0.15 → 0.10  (decreasing ✓)
Acc:   50% → 75% → 85% → 92% → 95%  (increasing ✓)
```

### Training Results

After training, check the output:

```
models/
├── scratch_cnn.pth              ← Your trained model!
└── training_history.json        ← Loss/accuracy per epoch
```

**View training history:**

```powershell
# Windows
notepad models\training_history.json

# Or use Python
python -c "import json; d=json.load(open('models/training_history.json')); print(f\"Best val acc: {max(d['val_acc']):.4f}\")"
```

---

## 4. Test the Trained Model

### On Webcam

```powershell
# Real-time detection with your trained model
python src/main.py --model scratch_cnn
```

**What to expect:**

- Green bounding boxes around detected faces
- FPS counter (30+ on CPU is good)
- Latency metric (should be <100ms)

### On Static Image

```powershell
# Test on a specific image
python src/main.py --model scratch_cnn --mode image --image "path/to/image.jpg"

# Or use the default test image
python src/main.py --model scratch_cnn --mode image
```

**Output:**

- Image with detected faces saved to `test_outputs/detection_output_N.jpg`
- Console shows: number of faces found, latency, confidence scores

---

## 5. Troubleshooting

### Error: "No module named 'torch'"

```powershell
# Install PyTorch
pip install torch torchvision

# Or if requirements.txt wasn't installed
pip install -r requirements.txt
```

### Error: "No training images found!"

```powershell
# Check your data structure
dir data\train\face\
dir data\train\non_face\
dir data\val\face\
dir data\val\non_face\

# Each folder should have image files (.jpg or .png)
```

### Error: "Model not found at models/scratch_cnn.pth"

```powershell
# You haven't trained yet
python scripts/train_scratch_cnn.py

# Then try inference
python src/main.py --model scratch_cnn
```

### Issue: Very Low Accuracy (50-60%)

**Possible causes:**

1. Too little training data (need 500+ total images)
2. Unbalanced dataset (more faces than non-faces)
3. Poor image quality
4. Learning rate too high (model not converging)

**Fixes:**

```powershell
# Add more data
# Increase dataset size

# Balance dataset
# Ensure equal faces and non-faces

# Lower learning rate
python scripts/train_scratch_cnn.py --lr 0.0001 --epochs 100
```

### Issue: Training Takes Forever

**Causes:** Slow computer, large batch size, high resolution

**Fixes:**

```powershell
# Reduce batch size (faster epochs but noisier)
python scripts/train_scratch_cnn.py --batch_size 16

# Reduce image size processing
python scripts/train_scratch_cnn.py --img_size 48
```

---

## 6. Performance Expectations

### Typical Results

| Dataset Size | Train Accuracy | Val Accuracy | Time per Epoch |
| ------------ | -------------- | ------------ | -------------- |
| 200 images   | 85%            | 75%          | 2 seconds      |
| 1000 images  | 95%            | 88%          | 8 seconds      |
| 5000 images  | 97%            | 92%          | 45 seconds     |

### Speed at Inference

| Device                | Time per Frame | FPS   |
| --------------------- | -------------- | ----- |
| CPU                   | 40-60ms        | 15-25 |
| CPU (GPU-accelerated) | 20-30ms        | 30-50 |

---

## 7. Example Workflows

### Quick Test (5 minutes)

```powershell
# Create minimal dataset
mkdir data\train\face data\train\non_face data\val\face data\val\non_face

# Add 10-20 images to each
copy face*.jpg data\train\face\
copy bg*.jpg data\train\non_face\

# Train
python scripts/train_scratch_cnn.py --epochs 10

# Test
python src/main.py --model scratch_cnn
```

### Production Training (1 hour)

```powershell
# Prepare large dataset: 1000+ images each split
# Train for longer
python scripts/train_scratch_cnn.py --epochs 100 --batch_size 64 --lr 0.0005

# Evaluate thoroughly
python src/main.py --model scratch_cnn --mode image --image test1.jpg
python src/main.py --model scratch_cnn --mode image --image test2.jpg
python src/main.py --model scratch_cnn  # Webcam test
```

### Comparative Study (MTCNN vs Scratch)

```powershell
# Test MTCNN (default)
python src/main.py --model mtcnn

# Note FPS and accuracy

# Test Scratch CNN
python src/main.py --model scratch_cnn

# Note FPS and accuracy

# Compare in your report
```

---

## 8. For Academic Reports

### Key Statistics to Report

```
Model: From-Scratch CNN for Face Detection
- Architecture: 3 conv layers, 2 FC layers
- Parameters: 227,536
- Input: 64×64 RGB images
- Output: Binary classification (face/non-face)

Training Configuration:
- Dataset size: [YOUR SIZE] images
- Train/val split: 80/20
- Batch size: 32
- Learning rate: 0.001
- Optimizer: Adam
- Loss: CrossEntropyLoss
- Regularization: Batch norm + Dropout(0.5)

Results:
- Training accuracy: [YOUR RESULT]%
- Validation accuracy: [YOUR RESULT]%
- Inference time: [YOUR TIME]ms per frame
- FPS achieved: [YOUR FPS] on CPU
```

### Code References to Cite

- Architecture: See `src/scratch_cnn.py` (TinyCNN class)
- Training: See `scripts/train_scratch_cnn.py`
- Inference: See `src/scratch_cnn.py` (ScratchCNNDetector class)

---

## Next Steps

1. ✅ Prepare dataset (500+ images recommended)
2. ✅ Train the model: `python scripts/train_scratch_cnn.py`
3. ✅ Test on webcam: `python src/main.py --model scratch_cnn`
4. ✅ Compare with MTCNN: `python src/main.py --model mtcnn`
5. ✅ Document results for your academic report

---

**Got questions?** Check [SCRATCH_CNN_ARCHITECTURE.md](SCRATCH_CNN_ARCHITECTURE.md) for deeper technical details.
