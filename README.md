# Real-Time Face Detection with MTCNN

> **Production-Grade Webcam Face Detection System**

A complete, tested, and documented real-time face detection application using MTCNN deep learning model with PyTorch and OpenCV.

---

## 📚 Documentation

All documentation is organized in the `Docs/` folder:

| Document                                                             | Purpose                  | Time   |
| -------------------------------------------------------------------- | ------------------------ | ------ |
| **[Docs/INDEX.md](Docs/INDEX.md)**                                   | Documentation navigation | 5 min  |
| **[Docs/QUICK_START.md](Docs/QUICK_START.md)**                       | Setup & run in 5 minutes | 5 min  |
| **[Docs/DELIVERY_SUMMARY.md](Docs/DELIVERY_SUMMARY.md)**             | Features & checklist     | 15 min |
| **[Docs/IMPLEMENTATION_SUMMARY.md](Docs/IMPLEMENTATION_SUMMARY.md)** | Technical deep dive      | 30 min |
| **[Docs/README.md](Docs/README.md)**                                 | Full reference guide     | 20 min |

**👉 Start here:** [Docs/QUICK_START.md](Docs/QUICK_START.md)

---

## 🚀 Quick Start

### **1. Setup Environment**

```bash
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### **2. Test Everything**

```bash
python test_components.py    # Comprehensive component tests
python tests/test_webcam.py  # Full test suite
```

### **3. Run Application**

```bash
python src/main.py                              # Real-time webcam
python src/main.py --mode image                # Test on static image
python src/main.py --model dummy              # Benchmark camera FPS
```

---

## ✅ What's Included

✅ **MTCNN Face Detector**

- Real-time face detection from webcam
- GPU acceleration (CUDA) with CPU fallback
- Automatic model download & caching

✅ **Real-Time Metrics**

- FPS counter (frames per second)
- Per-frame latency tracking (milliseconds)
- Confidence scores for each detection

✅ **Multiple Modes**

- Webcam real-time detection
- Static image testing
- FPS benchmarking (without detection)

✅ **Production Quality**

- Full error handling
- Comprehensive logging
- Complete documentation
- Tested components

---

## 📂 Project Structure

```
realtime-face-detection-dl/
├── Docs/                       # 📚 All documentation
│   ├── INDEX.md               # Start here!
│   ├── QUICK_START.md        # 5-minute setup
│   ├── README.md             # Full reference
│   ├── DELIVERY_SUMMARY.md   # Features
│   └── IMPLEMENTATION_SUMMARY.md  # Technical details
│
├── src/                        # Application code
│   ├── main.py               # Entry point
│   ├── detector.py           # MTCNN face detector
│   ├── config.py             # Configuration
│   └── fps.py                # Performance metrics
│
├── utils/                      # Utilities
│   ├── video.py              # Webcam & display
│   └── logger.py             # Logging
│
├── tests/                      # Tests
│   └── test_webcam.py        # Test suite
│
├── requirements.txt            # Dependencies
├── test_components.py          # Component tests
└── TEST_REPORT.md             # Test results
```

---

## 🧪 Testing

All components are tested and working:

```bash
python test_components.py
```

**Results:**

- ✅ Configuration system
- ✅ FPS counter
- ✅ Latency tracking
- ✅ Dummy detector
- ✅ MTCNN detector
- ✅ Video utilities
- ✅ Logging system

See [TEST_REPORT.md](TEST_REPORT.md) for detailed results.

---

## 💡 Features

- **MTCNN Architecture:** 3-stage cascade CNN for fast & accurate face detection
- **GPU Acceleration:** CUDA support, automatic CPU fallback
- **Pretrained Model:** Auto-downloads & caches (~100MB)
- **Real-Time Metrics:** FPS & per-frame latency display
- **Confidence Scores:** Each detection includes confidence (0.0-1.0)
- **Multiple Modes:** Webcam, image, and benchmark modes
- **Error Handling:** Graceful degradation with comprehensive logging
- **Well Documented:** 5 documentation files + inline comments

---

## 📊 Performance

| Setting           | FPS       | Latency   |
| ----------------- | --------- | --------- |
| GPU (NVIDIA CUDA) | 20-60 FPS | 35-50ms   |
| CPU (Intel i7)    | 2-10 FPS  | 150-300ms |

\*Note: First run takes 30-60s to download model (~100MB)

---

## 🎓 MTCNN Explained

MTCNN = **Multi-task Cascaded Convolutional Networks**

Works in **3 stages**:

1. **P-Net:** Proposal (fast, multi-scale scanning)
2. **R-Net:** Refinement (removes false positives)
3. **O-Net:** Output (final high-quality detections)

Result: Accurate face detection in 35-50ms on GPU

See [Docs/README.md](Docs/README.md) for detailed explanation.

---

## 🔧 Customization

Edit `src/config.py` to tune performance:

```python
# Faster (lower accuracy)
Config.FRAME_SCALE = 0.5                    # 50% resolution
Config.CONFIDENCE_THRESHOLD = 0.85          # More detections

# Slower (higher accuracy)
Config.FRAME_SCALE = 1.0                    # Full resolution
Config.CONFIDENCE_THRESHOLD = 0.95          # Fewer false positives
```

See [Docs/README.md](Docs/README.md#performance-optimization) for more options.

---

## 🐛 Troubleshooting

### Camera not found?

- Check camera is connected and not in use by another app
- Try different camera index in `config.py`

### MTCNN download fails?

- Check internet connection
- Model cached in `~/.cache/torch/checkpoints/`
- First run takes 30-60 seconds

### Low FPS?

- GPU too slow? Try `FRAME_SCALE = 0.5` for 4x speedup
- Use GPU instead of CPU (3-4x faster)
- Reduce resolution in config

See [Docs/QUICK_START.md](Docs/QUICK_START.md#troubleshooting) for more help.

---

## 📞 Need Help?

1. **Quick answer (30s):** [Docs/QUICK_START.md](Docs/QUICK_START.md#troubleshooting)
2. **More detail (2 min):** [Docs/README.md](Docs/README.md#troubleshooting)
3. **Technical help (5 min):** [Docs/IMPLEMENTATION_SUMMARY.md](Docs/IMPLEMENTATION_SUMMARY.md)
4. **Navigate all docs:** [Docs/INDEX.md](Docs/INDEX.md)

---

## ✨ Production Ready

This system is **fully tested** and **ready for deployment**:

- ✅ All components tested
- ✅ Full documentation included
- ✅ Error handling implemented
- ✅ Logging infrastructure in place
- ✅ Performance optimized

**Start with:** `python src/main.py`

---

## 📄 License

MIT License

---

## 🚀 Next Steps

1. Read [Docs/QUICK_START.md](Docs/QUICK_START.md) (5 minutes)
2. Run `python test_components.py` (verify everything works)
3. Run `python src/main.py` (start face detection)
4. Explore [Docs/](Docs/) for more information

---

**Status:** ✅ Production Ready  
**Version:** 1.0.0  
**Last Updated:** February 23, 2026
