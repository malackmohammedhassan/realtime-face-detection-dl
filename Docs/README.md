# Real-Time Face Detection with Deep Learning

A production-grade, **review-ready MVP** for real-time webcam face detection using **MTCNN** deep learning model and PyTorch.

Implements **Multi-task Cascaded Convolutional Networks (MTCNN)** - a state-of-the-art cascade-based face detector that balances speed and accuracy. Includes real-time FPS and per-frame latency tracking for performance optimization.

---

## ✨ Key Features

✅ **MTCNN Face Detection** - State-of-the-art cascade CNN architecture  
✅ **GPU Acceleration** - CUDA support with automatic CPU fallback  
✅ **Pretrained Weights** - Auto-downloads on first run (~100MB)  
✅ **Real-Time Performance** - FPS and per-frame latency display  
✅ **Confidence Scores** - Bounding boxes with detection confidence  
✅ **Frame Resizing** - Configurable frame scale for speed optimization  
✅ **Static Image Mode** - Test detection on static images  
✅ **Comprehensive Logging** - Debug and production-level logging  
✅ **Production Ready** - Error handling, graceful degradation, clean code

---

## 🚀 Quick Start

### 1. **Environment Setup (Windows)**

```powershell
# Navigate to project directory
cd realtime-face-detection-dl

# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate
```

### 2. **Install Dependencies**

```powershell
# Upgrade pip
python -m pip install --upgrade pip

# Install requirements (includes MTCNN + PyTorch)
pip install -r requirements.txt
```

**⚠️ First Run Note:**  
MTCNN model weights (~100MB) are auto-downloaded from PyTorch Hub on first inference and cached locally. First detection may take 30-60 seconds.

### 3. **Verify Installation**

```powershell
# Comprehensive test suite (includes static image detection)
python tests/test_webcam.py
```

This will:

- ✓ Check OpenCV, PyTorch, CUDA availability
- ✓ Test MTCNN model loading and initialization
- ✓ Run detection on a test image (synthetic if none exists)
- ✓ Test webcam capture (press Q to exit)

### 4. **Run Real-Time Face Detection**

```powershell
# Webcam mode (default)
python src/main.py

# Or with explicit options
python src/main.py --mode webcam --model mtcnn

# Static image test mode
python src/main.py --mode image --image assets/test.jpg

# Use dummy model (benchmark camera FPS without inference)
python src/main.py --model dummy
```

**Live Display Shows:**

- Real-time webcam feed
- **FPS counter** - Frames per second
- **Inference latency** - Per-frame detection time in milliseconds
- **Face bounding boxes** - Green rectangles around detected faces
- **Confidence scores** - Detection confidence (0.0-1.0)

**Controls:**

- `Q` - Quit application

---

## 📋 Project Structure

```
realtime-face-detection-dl/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies (MTCNN, PyTorch)
├── .gitignore                         # Git ignore rules
│
├── src/
│   ├── main.py                        # Entry point (webcam & image modes)
│   ├── config.py                      # MTCNN & detection configuration
│   ├── detector.py                    # MTCNN face detector implementation
│   └── fps.py                         # FPS & latency measurement
│
├── utils/
│   ├── video.py                       # Webcam capture & display helpers
│   └── logger.py                      # Logging utility
│
├── tests/
│   └── test_webcam.py                 # Comprehensive test suite
│
├── models/                            # Model weights cache directory
│
└── assets/                            # Test images
```

---

## 🧠 How MTCNN Works

**MTCNN** (Multi-task Cascaded Convolutional Networks) is a **three-stage cascade** of CNNs:

### **Stage 1: P-Net (Proposal Network)**

- **Input:** Image at multiple scales (image pyramid)
- **Output:** Face candidate windows with high recall
- **Purpose:** Generate many proposals, filter obvious non-faces
- **Speed:** Very fast (many candidates but rough detections)

```
Fast coarse detection at multiple scales
↓
Generates many candidates (high recall, lower precision)
```

### **Stage 2: R-Net (Refinement Network)**

- **Input:** Candidate windows from P-Net
- **Output:** Refined candidates (removes ~80% of false positives)
- **Purpose:** Filter P-Net proposals, adjust bounding box coordinates
- **Speed:** Moderate (fewer candidates to process)

```
Process P-Net candidates
↓
Remove false positives
↓
Refine bbox coordinates via regression
```

### **Stage 3: O-Net (Output Network)**

- **Input:** Refined candidates from R-Net
- **Output:** Final high-quality detections + landmark points
- **Purpose:** Final detection output with landmarks (5-point face landmarks optional)
- **Speed:** Slower but produces best detections

```
High-precision detection
↓
Final bounding box + confidence score
↓
Optional: 5-point face landmarks (eyes, nose, mouth corners)
```

### **Why Cascade?**

- **Efficiency:** Early stages reject obvious non-faces quickly
- **Accuracy:** Later stages refine detections
- **Speed:** Processes full image in P-Net, only regions in R/O-Net
- **Result:** Fast (GPU: ~5-50ms per frame) + accurate detections

---

## 🔄 Transfer Learning & Pretrained Models

### **Why Use Pretrained MTCNN Instead of Training from Scratch?**

#### **Training from scratch would require:**

- ✗ Huge face datasets (10M+ images)
- ✗ Weeks of GPU training (V100+)
- ✗ Hundreds of dollars in compute costs
- ✗ Tedious hyperparameter tuning
- ✗ Expertise in model design & training

#### **Using pretrained model saves us:**

- ✓ Models trained on WIDER-FACE (420k+ annotated faces)
- ✓ Learned features transfer to any face image worldwide
- ✓ Hours instead of weeks of training
- ✓ $0 training cost (just inference)
- ✓ Production-ready model immediately

### **How Transfer Learning Works**

```
Pretrained MTCNN
    ↓
Learned face features from WIDER-FACE dataset
    ↓
Can generalize to new faces (transfer)
    ↓
Direct inference on your webcam (no retraining)
```

**Key Insight:** MTCNN learned:

- Face patterns, textures, shapes
- Landmark positions (eyes, nose, mouth)
- Multi-scale face variations
- All from training on millions of faces

These features work on ANY face image → we use them directly.

---

## 🔧 Configuration

### `src/config.py` - All Settings in One Place

```python
from src.config import Config

Config.DEVICE              # "cuda" or "cpu" (auto-detected)
Config.DETECTION_MODEL    # "mtcnn" (only option for now)
Config.CONFIDENCE_THRESHOLD  # 0.95 (higher = fewer false positives)
Config.FRAME_SCALE        # 1.0 (resize frame before inference)
Config.USE_HALF_PRECISION # False (FP16 for faster GPU inference)
Config.MEASURE_LATENCY    # True (measure detection time)

Config.print_config()     # Print all settings
```

**Key Parameters:**

| Parameter              | Default | Purpose                                        |
| ---------------------- | ------- | ---------------------------------------------- |
| `CONFIDENCE_THRESHOLD` | 0.95    | Filter low-confidence detections (0.0-1.0)     |
| `FRAME_SCALE`          | 1.0     | Resize frame (0.5 = 2x faster, lower accuracy) |
| `WEBCAM_WIDTH`         | 640     | Frame width in pixels                          |
| `WEBCAM_HEIGHT`        | 480     | Frame height in pixels                         |

---

## 📊 Module Documentation

### `src/detector.py` - MTCNN Implementation

**Classes:**

```python
MTCNNDetector(device, confidence_threshold)
  ├── detect(frame, scale)  # Run inference
  └── device, model_name, model

FaceDetector(model_name, device)
  ├── detect(frame, scale)  # Unified interface
  ├── get_model_name()
  └── get_device()
```

**Usage:**

```python
from src.detector import FaceDetector
import torch

# Initialize detector (auto-downloads MTCNN weights)
detector = FaceDetector(model_name="mtcnn")

# Run detection
detections = detector.detect(frame)

# Returns: [(x, y, w, h, confidence), ...]
for x, y, w, h, conf in detections:
    print(f"Face at ({x}, {y}), size {w}x{h}, confidence {conf:.2f}")
```

**Key Features:**

- ✓ Auto-downloads pretrained MTCNN model
- ✓ GPU acceleration (CUDA)
- ✓ Frame resizing for speed
- ✓ Confidence filtering
- ✓ Error handling with logging

---

### `src/fps.py` - Performance Metrics

**Classes:**

```python
FPSCounter(window_size=30)
  ├── update()           # Call once per frame
  ├── get_fps()
  └── reset()

LatencyCounter(window_size=30)
  ├── start()            # Before inference
  ├── end()              # After inference, returns latency_ms
  ├── get_avg_latency_ms()
  └── reset()
```

**Usage:**

```python
from src.fps import FPSCounter, LatencyCounter

fps_counter = FPSCounter(window_size=30)
latency_counter = LatencyCounter(window_size=30)

# Main loop
while True:
    fps = fps_counter.update()

    latency_counter.start()
    detections = detector.detect(frame)
    latency_ms = latency_counter.end()

    print(f"FPS: {fps:.1f}, Latency: {latency_ms:.2f}ms")
```

---

### `src/main.py` - Application Entry Point

**Modes:**

```bash
# Webcam real-time detection
python src/main.py --mode webcam --model mtcnn

# Static image test
python src/main.py --mode image --image path/to/image.jpg
```

**Features:**

- ✓ Real-time face detection with FPS
- ✓ Per-frame latency tracking
- ✓ Bounding box drawing with confidence scores
- ✓ Graceful shutdown & error handling
- ✓ Static image mode for testing

---

### `utils/video.py` - Webcam & Display

```python
WebcamCapture(camera_index=0, width=640, height=480, fps=30)
  ├── read()             # Returns (success, frame)
  ├── get_frame_size()
  ├── release()
  └── __enter__, __exit__ # Context manager

FrameDisplay(window_name="Face Detection", fps=30)
  ├── imshow(frame)
  ├── wait_key(delay)
  ├── close()
  └── Context manager support
```

**Usage:**

```python
from utils.video import WebcamCapture, FrameDisplay

with WebcamCapture(width=640, height=480) as cam:
    with FrameDisplay() as display:
        while True:
            success, frame = cam.read()
            display.imshow(frame)
            if display.wait_key() == ord('q'):
                break
```

---

### `utils/logger.py` - Logging

```python
from utils.logger import setup_logger

logger = setup_logger(
    name="FaceDetection",
    log_level="INFO",  # DEBUG, INFO, WARNING, ERROR
    log_file=False
)

logger.info("Application started")
logger.error("Error message")
```

---

### `tests/test_webcam.py` - Comprehensive Tests

```bash
python tests/test_webcam.py
```

**Tests:**

1. ✓ OpenCV import & version
2. ✓ PyTorch & CUDA detection
3. ✓ MTCNN model initialization (downloads if needed)
4. ✓ Static image detection test
5. ✓ Webcam capture test

Saves detection output: `test_detection_result.jpg`

---

## 📈 Performance Optimization

### **Frame Resizing (Trade Speed for Accuracy)**

```python
# In config.py:
FRAME_SCALE = 0.5  # Process at 50% resolution
                   # 4x faster, but misses small faces

FRAME_SCALE = 1.0  # Full resolution (default)
                   # Slower but better accuracy
```

### **GPU Acceleration**

```python
# Automatic GPU detection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Typical latencies:
# GPU (CUDA): 5-50ms per frame
# CPU: 100-500ms per frame
```

### **Half Precision (FP16)**

```python
# In config.py:
USE_HALF_PRECISION = True  # GPU only, 2x faster, slight accuracy loss
```

---

## 🐛 Troubleshooting

### MTCNN Model Download Fails

```
RuntimeError: Failed to download MTCNN model
```

**Solution:**

- Check internet connection
- Models cached in `~/.cache/torch/checkpoints/`
- Manual download available from PyTorch Hub

### Low FPS on CPU

**Expected:**

- GPU (CUDA): 20-60 FPS (5-50ms/frame)
- CPU: 2-10 FPS (100-500ms/frame)

**Optimization:**

- Reduce `FRAME_SCALE` in config.py
- Reduce `WEBCAM_WIDTH` and `WEBCAM_HEIGHT`
- Use GPU if available

### No Faces Detected

**Check:**

- `CONFIDENCE_THRESHOLD` is not too high (default 0.95)
- Faces are large enough (P-Net minimum ~20 pixels)
- Good lighting conditions

---

## 📦 Requirements

| Package         | Version  | Purpose                       |
| --------------- | -------- | ----------------------------- |
| opencv-python   | 4.8.1.78 | Frame capture & display       |
| torch           | 2.1.0    | Deep learning framework       |
| torchvision     | 0.16.0   | Computer vision utilities     |
| facenet-pytorch | 2.5.3    | MTCNN model + face embeddings |
| numpy           | 1.24.3   | Numerical operations          |
| Pillow          | 10.0.0   | Image processing              |

---

## 🎯 Next Steps (Future Enhancements)

1. **Face Recognition**
   - Add face embeddings with facenet-pytorch
   - Compare faces, identify persons

2. **Face Landmarks**
   - Extract 5-point landmarks (eyes, nose, mouth)
   - Draw face landmarks on output

3. **Multi-Face Tracking**
   - Track faces across frames
   - Assign IDs to detected faces

4. **Optimization**
   - ONNX export for faster inference
   - TensorFlow Lite for mobile
   - Quantization (INT8) for lower latency

5. **Advanced Features**
   - Age/gender estimation
   - Emotion recognition
   - Face blur/anonymization

---

## 📜 License

MIT License

---

## 📞 Support

**Issues?**

1. Check troubleshooting section above
2. Review logs in `logs/` directory
3. Verify dependencies: `pip list | grep -E "torch|opencv|mtcnn"`
4. Run test suite: `python tests/test_webcam.py`

---

## ✅ Production Checklist

- [x] MTCNN face detection implemented
- [x] GPU acceleration (CUDA) with CPU fallback
- [x] Real-time FPS & latency tracking
- [x] Bounding boxes with confidence scores
- [x] Frame resizing optimization
- [x] Static image test mode
- [x] Comprehensive error handling
- [x] Full logging infrastructure
- [x] Test suite with image detection
- [x] Clean, modular, well-documented code

**Ready for review and deployment!**

---

**Version:** 1.0.0 (Production MVP)  
**Last Updated:** February 23, 2026
