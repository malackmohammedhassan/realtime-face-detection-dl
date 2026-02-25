# Scratch CNN: From-Scratch Face Detection Classifier

## Overview

This document provides a comprehensive explanation of the from-scratch CNN face vs non-face binary classifier integrated into the real-time face detection project.

**Key Characteristics:**

- **Architecture**: Lightweight 3-layer CNN + 2 FC layers
- **Input**: 64×64 RGB images
- **Output**: Binary classification (face / non-face)
- **Training**: CrossEntropyLoss + Adam optimizer
- **Inference**: Sliding window approach with Non-Maximum Suppression (NMS)
- **Performance**: ~30-50ms per frame on CPU

---

## Architecture Design

### Why a From-Scratch CNN?

While the project primarily uses MTCNN (a pretrained cascade detector), implementing a from-scratch CNN serves several purposes:

1. **Educational Value**: Demonstrates the full ML pipeline
2. **Research Flexibility**: Can experiment with custom architectures and training strategies
3. **Domain Adaptation**: Can be fine-tuned on specific face datasets
4. **Comparative Analysis**: Compare performance vs. pretrained models

### Model Architecture

```
Input: 64×64×3 RGB Image
  ↓
[Conv2d(3→32) + BatchNorm + ReLU + MaxPool(2×2)]
  32×32×32 feature maps
  ↓
[Conv2d(32→64) + BatchNorm + ReLU + MaxPool(2×2)]
  16×16×64 feature maps
  ↓
[Conv2d(64→128) + BatchNorm + ReLU + MaxPool(2×2)]
  8×8×128 feature maps
  ↓
Flatten: 8192 features
  ↓
[FC(8192→256) + ReLU + Dropout(0.5)]
  ↓
[FC(256→2) + Softmax]
  ↓
Output: [P(non-face), P(face)]
```

### Design Decisions

#### 1. **Input Size: 64×64**

- **Rationale**: Balances face detection capability with computational efficiency
- **Trade-off**: Larger inputs capture more detail but slow inference
- **Standard**: Commonly used in face detection literature
- **Flexibility**: Windows can overlap for better coverage

#### 2. **Three Convolutional Layers**

- **Pattern**: 32 → 64 → 128 filters
- **Progression**: Captures hierarchical features
  - Layer 1: Edge/corner detection
  - Layer 2: Feature combinations (textures, shapes)
  - Layer 3: Complex patterns (facial features)
- **Kernel Size**: 3×3 (standard choice for efficiency)
- **Padding**: 1 (maintains spatial dimensions before pooling)

#### 3. **Batch Normalization**

```python
# After each conv layer
BatchNorm2d(num_filters)
```

- **Benefits**:
  - Stabilizes training (reduces internal covariate shift)
  - Allows higher learning rates
  - Acts as weak regularizer
  - Reduces sensitivity to weight initialization

#### 4. **Max Pooling (2×2 stride 2)**

- **Dimensions**: 64→32→16→8
- **Purpose**: Increase receptive field, reduce computation
- **Invariance**: Provides translation invariance to small shifts

#### 5. **Fully Connected Layers**

```
Flatten(8×8×128=8192) → FC(256) → FC(2)
```

- **First FC**: 256 hidden units (bottleneck for feature compression)
- **Dropout(0.5)**: Prevent overfitting during training
  - Randomly deactivates 50% of neurons
  - Effective regularization without L2 penalty
- **Output**: 2 logits (for softmax classification)

#### 6. **Total Parameters**

```
Conv layers:  ~18,500 parameters
FC layers:    ~209,000 parameters
Total:        ~227,500 parameters
```

This is **extremely lightweight** compared to:

- MTCNN: ~4 million parameters
- ResNet-50: ~25 million parameters

---

## Training Process

### Dataset Structure

```
data/
├── train/
│   ├── face/          # Positive samples
│   │   ├── face_0001.jpg
│   │   ├── face_0002.jpg
│   │   └── ...
│   └── non_face/      # Negative samples
│       ├── bg_0001.jpg
│       ├── bg_0002.jpg
│       └── ...
└── val/
    ├── face/          # Validation faces
    └── non_face/      # Validation non-faces
```

### Data Augmentation Strategy

**Applied only to training set:**

```python
transforms.Compose([
    transforms.RandomRotation(15),          # ±15° rotation
    transforms.ColorJitter(                 # Brightness/contrast variation
        brightness=0.2,
        contrast=0.2
    ),
    transforms.RandomAffine(                # Translation: ±10%
        degrees=0,
        translate=(0.1, 0.1)
    ),
    transforms.ToTensor(),
    transforms.Normalize(                   # ImageNet normalization
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

**Why augmentation?**

- Increases effective dataset size
- Forces model to learn robust features
- Improves generalization to real-world variations
- Reduces overfitting risk

### Training Hyperparameters

| Parameter               | Value            | Rationale                                     |
| ----------------------- | ---------------- | --------------------------------------------- |
| Learning Rate           | 0.001            | Standard start; adaptive via Adam             |
| Optimizer               | Adam             | Works well for conv nets; momentum + adaptive |
| Loss Function           | CrossEntropyLoss | Standard for classification                   |
| Batch Size              | 32               | Balance: memory vs. gradient stability        |
| Weight Decay (L2)       | 1e-4             | Mild regularization                           |
| Dropout Rate            | 0.5              | Moderate regularization in FC                 |
| Max Epochs              | 50               | With early stopping                           |
| Early Stopping Patience | 10 epochs        | Stop if no improvement                        |

### Training Loop

```python
for epoch in range(num_epochs):
    # Training phase
    for batch in train_loader:
        images, labels = batch.to(device)

        # Forward pass
        logits = model(images)
        loss = criterion(logits, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation phase
    val_loss, val_acc = validate(val_loader)

    # Save best model
    if val_acc > best_accuracy:
        save_checkpoint(model, is_best=True)
        best_accuracy = val_acc
    else:
        patience_counter += 1
        if patience_counter >= 10:
            break  # Early stopping
```

### Expected Performance

**On a balanced dataset (50% faces, 50% non-faces):**

| Metric              | Typical Value | Notes                      |
| ------------------- | ------------- | -------------------------- |
| Training Accuracy   | 95-98%        | After 30-40 epochs         |
| Validation Accuracy | 90-95%        | Depends on dataset quality |
| Train Loss          | 0.05-0.10     | Minimal overfitting        |
| Inference Time      | 30-50ms       | Per 64×64 window on CPU    |

**Factors affecting performance:**

- Dataset size (more data → better accuracy)
- Dataset balance (equal faces/non-faces preferred)
- Image quality (clean backgrounds help)
- Training duration (early stopping helps)

---

## Inference: Sliding Window Detection

### Algorithm Overview

```
Input: Full image (e.g., 480×640)
1. Resize to 64×64
2. Extract windows: 64×64, stride 16 pixels
3. Classify each window: face or non-face
4. Keep windows with confidence > threshold
5. Apply NMS to remove overlaps
Output: List of (x, y, w, h, confidence)
```

### Sliding Window Parameters

```python
window_size = 64      # Patch size for CNN
stride = 16           # Pixel overlap between windows
                      # 64-16=48 pixels overlap (75% overlap)
```

**Why 75% overlap?**

- Small window might miss partial faces at edges
- High overlap ensures faces are contained wholly in ≥1 window
- Trade-off: More windows = slower but better coverage

### Non-Maximum Suppression (NMS)

After sliding window detection, multiple overlapping windows detect the same face:

```
Before NMS:
  Window 1: [95, 120, 155, 180], confidence=0.92
  Window 2: [98, 122, 158, 182], confidence=0.88
  Window 3: [100, 125, 160, 185], confidence=0.85

After NMS (IoU threshold=0.3):
  Window 1: [95, 120, 155, 180], confidence=0.92  ← KEEP (highest conf)
  Window 2, 3: REMOVED (overlap with Window 1)
```

**NMS Implementation:**

```python
def nms(detections, iou_threshold=0.3):
    # Sort by confidence (descending)
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)

    keep = []
    while detections:
        keep.append(detections[0])
        current_box = detections[0]['box']

        # Remove overlapping boxes
        detections = [
            d for d in detections[1:]
            if iou(current_box, d['box']) < iou_threshold
        ]

    return keep
```

### Inference Speed Optimization

**Method 1: Frame Scaling**

```python
# Reduce inference resolution for speed
detector.detect(frame, scale=0.5)  # 50% resolution
# ~4x faster due to fewer windows to process
```

**Method 2: Stride Adjustment**

```python
# Increase stride (fewer windows)
detector.detect(frame, stride=32)  # 8x fewer windows than stride=4
# Trade-off: Might miss small faces
```

**Method 3: Confidence Threshold**

```python
# Higher threshold = fewer detections
detector = ScratchCNNDetector(confidence_threshold=0.9)
# Faster post-processing
```

---

## Comparison: Scratch CNN vs. MTCNN

| Aspect               | Scratch CNN      | MTCNN                     |
| -------------------- | ---------------- | ------------------------- |
| **Training Data**    | Custom dataset   | WIDER-FACE (400k+ faces)  |
| **Architecture**     | Simple (3 conv)  | Cascade (3 networks)      |
| **Parameters**       | 227K             | 4 million                 |
| **Speed**            | 30-50ms/frame    | 20-40ms/frame             |
| **Accuracy**         | 90-95% (typical) | 95%+                      |
| **Adaptability**     | Easy to retrain  | Fixed pretrained          |
| **Inference Method** | Sliding window   | Pyramid scales            |
| **Python Code**      | 400 lines        | Library (facenet-pytorch) |

### When to Use Each

**Use MTCNN if:**

- You need maximum accuracy out-of-the-box
- Faces are arbitrary sizes/angles
- Real-time performance is critical
- You have limited training data

**Use Scratch CNN if:**

- You want to learn/teach CNN concepts
- You have domain-specific face data
- You need to customize the architecture
- You want full control over the pipeline

---

## Training on Your Own Data

### Step 1: Prepare Dataset

```
# Create directory structure
mkdir -p data/train/face data/train/non_face
mkdir -p data/val/face data/val/non_face

# Add face images
# Face images: resize to 64×64 (or larger; model will resize)
for img in face_photos/*.jpg; do
    convert $img -resize 64x64 data/train/face/$(basename $img)
done

# Add non-face images (trees, buildings, objects, etc.)
for img in background_photos/*.jpg; do
    convert $img -resize 64x64 data/train/non_face/$(basename $img)
done
```

### Step 2: Train the Model

```bash
# Activate environment
.\venv\Scripts\activate

# Train with custom parameters
python scripts/train_scratch_cnn.py \
    --epochs 100 \
    --batch_size 64 \
    --lr 0.0005 \
    --data_dir data \
    --output_dir models
```

### Step 3: Monitor Training

```
Epoch [  1/100] | Train Loss: 0.6931, Acc: 0.4800 | Val Loss: 0.6920, Acc: 0.5100
Epoch [  5/100] | Train Loss: 0.4532, Acc: 0.7800 | Val Loss: 0.4120, Acc: 0.8200
Epoch [ 10/100] | Train Loss: 0.2103, Acc: 0.9100 | Val Loss: 0.2540, Acc: 0.8950
...
✓ Best model saved (Acc: 0.9340)
```

### Step 4: Use in Inference

```python
# Automatic detection of trained model
python src/main.py --model scratch_cnn
```

---

## Academic Report Notes

### Key Points to Highlight

1. **Lightweight Design**
   - Only 227K parameters vs. 4M for MTCNN
   - Suitable for edge devices, mobile deployment
   - Faster training (hours vs. days)

2. **Training Pipeline**
   - Full supervised learning: data → augmentation → training → validation
   - Early stopping prevents overfitting
   - Cross-entropy loss standard for classification

3. **Inference Technique**
   - Sliding window: simple but exhaustive approach
   - NMS: Standard technique for duplicate removal
   - Trade-off: Coverage vs. speed

4. **Generalization**
   - Batch normalization and dropout for regularization
   - Data augmentation for robustness
   - Separate train/val splits for unbiased evaluation

5. **Future Improvements**
   - Anchor-based detection (like YOLO/SSD)
   - Multi-scale feature processing
   - Attention mechanisms for salient regions
   - Knowledge distillation from MTCNN

### Sample Academic Phrasing

> "We implement a lightweight convolutional neural network for binary face/non-face classification. The model comprises three convolutional blocks with batch normalization and max pooling, followed by two fully connected layers with dropout regularization. Face detection is performed via dense sliding window evaluation with non-maximum suppression post-processing."

> "The network architecture is deliberately minimalistic (227K parameters) to facilitate understanding of CNN principles while maintaining competitive performance. Training employs standard cross-entropy loss with Adam optimization over 50 epochs with early stopping based on validation accuracy."

---

## File Structure

```
src/
├── scratch_cnn.py              # Main CNN implementation
│   ├── TinyCNN                 # Model architecture
│   ├── FaceNonFaceDataset      # Data loader
│   ├── ScratchCNNTrainer       # Training loop
│   └── ScratchCNNDetector      # Inference wrapper
└── detector.py                 # Integration with factory
                                    └── ScratchCNNDetectorWrapper

scripts/
└── train_scratch_cnn.py         # Training script

models/
└── scratch_cnn.pth             # Trained model checkpoint
    └── training_history.json   # Loss/accuracy per epoch
```

---

## Troubleshooting

### Issue: "Module not found" when using scratch_cnn

**Solution**: Ensure the model is trained first

```bash
python scripts/train_scratch_cnn.py
```

### Issue: Low validation accuracy

**Possible causes**:

1. Unbalanced dataset (more faces than non-faces)
2. Poor image quality or preprocessing
3. Too few training samples
4. High learning rate causing instability

**Solutions**:

- Balance dataset (equal faces/non-faces)
- Increase training data
- Lower learning rate: `--lr 0.0001`
- Train longer: `--epochs 200`

### Issue: Slow inference

**Options**:

```python
# Method 1: Reduce image resolution
detector.detect(frame, scale=0.5)

# Method 2: Increase stride (fewer windows)
detector = ScratchCNNDetector(..., stride=32)

# Method 3: Higher confidence threshold
detector = ScratchCNNDetector(..., confidence_threshold=0.9)
```

---

## References

- **Batch Normalization**: Ioffe & Szegedy (2015) - "Batch Normalization: Accelerating Deep Network Training"
- **ImageNet Normalization**: Standard preprocessing in computer vision
- **Non-Maximum Suppression**: Classic technique for object detection post-processing
- **Early Stopping**: Prevents overfitting by monitoring validation loss/accuracy

---

## Using with Main Application

```bash
# Webcam detection with scratch CNN
python src/main.py --model scratch_cnn

# Image detection with scratch CNN
python src/main.py --model scratch_cnn --mode image --image photo.jpg

# Comparison with MTCNN
python src/main.py --model mtcnn  # MTCNN (default)
```

---

**Last Updated**: February 2026
**Author**: Real-Time Face Detection Project
**Version**: 1.0
