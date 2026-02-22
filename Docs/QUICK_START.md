# Quick Start Guide

## 🚀 Get Running in 5 Minutes

### **Step 1: Setup Virtual Environment** (2 min)

```powershell
cd realtime-face-detection-dl
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### **Step 2: Verify Installation** (2 min)

```powershell
python tests/test_webcam.py
```

✅ If all tests pass, you're good to go!

### **Step 3: Run Face Detection** (1 min)

**Option A: Real-Time Webcam Detection (Recommended)**

```powershell
python src/main.py
```

- Shows live video with detected faces
- Displays FPS and inference latency
- Press `Q` to quit

**Option B: Test on Static Image**

```powershell
python src/main.py --mode image --image assets/test.jpg
```

- Runs detection on a single image
- Saves result to `detection_output.jpg`

**Option C: Benchmark Camera FPS (No Detection)**

```powershell
python src/main.py --model dummy
```

- Shows camera FPS without detection overhead
- Use to compare performance

---

## 📊 What You'll See

### **Live Display**

```
┌─────────────────────────────────────┐
│  Real-Time Face Detection           │
│                                     │
│    FPS: 28.5                       │
│    Inference: 35.42ms (Avg: 38ms)  │
│                                     │
│  [Live video with green bounded    │
│   boxes around detected faces]     │
│                                     │
│  Each box shows confidence score   │
│  Press Q to quit                   │
└─────────────────────────────────────┘
```

---

## ⚙️ Customize Performance

Edit `src/config.py` to tune performance:

```python
# Faster (lower accuracy)
Config.FRAME_SCALE = 0.5                    # Process at 50% resolution
Config.CONFIDENCE_THRESHOLD = 0.85          # More detections
Config.WEBCAM_WIDTH = 320                   # Lower resolution
Config.WEBCAM_HEIGHT = 240

# Slower (higher accuracy)
Config.FRAME_SCALE = 1.0                    # Full resolution
Config.CONFIDENCE_THRESHOLD = 0.95          # Fewer false positives
Config.WEBCAM_WIDTH = 1280                  # Higher resolution
Config.WEBCAM_HEIGHT = 720
```

---

## 🐛 Troubleshooting

### **"MTCNN model download fails"**

- Check internet connection
- First run takes 30-60 seconds to download (~100MB)
- Model is cached in `~/.cache/torch/checkpoints/`

### **"Low FPS on GPU"**

- NVIDIA drivers installed? Check with `nvidia-smi`
- Update CUDA from https://developer.nvidia.com/cuda-downloads
- PyTorch will auto-detect and use GPU if available

### **"No faces detected"**

- Try lower confidence threshold (default 0.95 is very strict)
- Edit `src/config.py`: `Config.CONFIDENCE_THRESHOLD = 0.85`
- Ensure good lighting
- Faces should be at least 20x20 pixels

### **"Camera not found"**

- Check if camera is connected
- Verify no other app is using camera
- Try different camera index in `config.py`: `camera_index = 1`

---

## 📁 Project Files

| File                        | Purpose                  |
| --------------------------- | ------------------------ |
| `src/main.py`               | Application entry point  |
| `src/detector.py`           | MTCNN face detection     |
| `src/config.py`             | Configuration settings   |
| `src/fps.py`                | Performance metrics      |
| `utils/video.py`            | Webcam / display helpers |
| `utils/logger.py`           | Logging                  |
| `tests/test_webcam.py`      | Test suite               |
| `README.md`                 | Full documentation       |
| `IMPLEMENTATION_SUMMARY.md` | Technical details        |

---

## 💡 Key Features

✅ **MTCNN Face Detection** - State-of-the-art 3-stage cascade CNN  
✅ **GPU Acceleration** - CUDA support, automatic CPU fallback  
✅ **Real-Time FPS** - Display frames per second on screen  
✅ **Latency Tracking** - Show inference time per frame (ms)  
✅ **Confidence Scores** - Each detection has confidence 0-1  
✅ **Multiple Modes** - Webcam, static image, or test  
✅ **Production Ready** - Full error handling and logging

---

## 📚 Learn More

- `README.md` - Full documentation, architecture, optimization tips
- `IMPLEMENTATION_SUMMARY.md` - Technical deep dive
- `DELIVERY_SUMMARY.md` - Feature list and delivery checklistA

---

## ✅ Checklist

- [ ] Python 3.10+ installed
- [ ] Virtual environment created
- [ ] Requirements installed (`pip install -r requirements.txt`)
- [ ] Tests pass (`python tests/test_webcam.py`)
- [ ] Webcam works (`python src/main.py`)
- [ ] Detections show on screen

---

## 🎓 Understanding MTCNN

MTCNN = **Multi-task Cascaded Convolutional Networks**

Works in **3 stages**:

```
1. P-Net (Proposal)
   ├─ Scans image at multiple scales
   ├─ Finds face candidates (fast, many proposals)
   └─ Output: ~1000 potential faces

2. R-Net (Refinement)
   ├─ Processes P-Net candidates
   ├─ Removes ~80% false positives
   └─ Output: ~100 refined candidates

3. O-Net (Output)
   ├─ Final high-precision detection
   ├─ Returns bounding box + landmarks
   └─ Output: Final face detections
```

**Why this design?**

- P-Net fast → quickly reject obvious non-faces
- R-Net refines → removes false positives
- O-Net precise → final quality detections
- **Result:** All stages together = accurate + fast!

---

## 🚀 Performance Typical Values

| Device            | FPS       | Latency/Frame |
| ----------------- | --------- | ------------- |
| GPU (NVIDIA CUDA) | 20-60 FPS | 35-50ms       |
| CPU (Intel i7)    | 2-10 FPS  | 150-300ms     |

**First frame:** Takes 30-60 seconds (model download)  
**Subsequent frames:** Real-time performance above

---

**Version: 1.0.0 | Status: Production Ready | Last Updated: Feb 23, 2026**
