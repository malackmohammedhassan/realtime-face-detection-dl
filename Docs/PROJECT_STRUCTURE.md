# Project Structure

## Directory Tree

```
realtime-face-detection-dl/
├── .gitignore                          # Git ignore file (excludes venv, __pycache__, models, test outputs)
├── README.md                           # Root README - Entry point for users
├── requirements.txt                    # Python package dependencies
│
├── src/                                # Core application source code
│   ├── __init__.py                    # Package initialization
│   ├── main.py                        # Main application entry point
│   │                                  # - Real-time webcam detection
│   │                                  # - Static image test mode
│   │                                  # - Benchmark mode
│   ├── detector.py                    # MTCNN detector implementation
│   │                                  # - MTCNNDetector: Actual face detection
│   │                                  # - DummyDetector: Placeholder for benchmarks
│   │                                  # - FaceDetector: Unified interface
│   ├── config.py                      # Centralized configuration
│   │                                  # - Device selection (GPU/CPU)
│   │                                  # - Model parameters
│   │                                  # - Detection thresholds
│   └── fps.py                         # Performance metrics
│                                      # - FPSCounter: Real-time FPS calculation
│                                      # - LatencyCounter: Inference time tracking
│
├── utils/                             # Utility functions
│   ├── __init__.py                    # Package initialization
│   ├── video.py                       # Video capture and display helpers
│   │                                  # - WebcamCapture: Safe camera access
│   │                                  # - FrameDisplay: Window management
│   │                                  # - put_text_on_frame(): Text overlay
│   │                                  # - draw_bbox(): Bounding box drawing
│   └── logger.py                      # Logging utilities
│                                      # - Configure logging with timestamps
│
├── tests/                             # Test suite
│   ├── __init__.py                    # Package initialization
│   ├── test_webcam.py                 # System-level tests (243 lines)
│   │                                  # - OpenCV/PyTorch imports
│   │                                  # - CUDA detection
│   │                                  # - MTCNN initialization
│   │                                  # - Static image detection
│   │                                  # - Webcam capture verification
│   └── test_components.py             # Component-level tests (120 lines)
│                                      # - Configuration loading
│                                      # - FPS/Latency counters
│                                      # - Detector initialization
│                                      # - Video utilities
│                                      # - Logger setup
│                                      # - MTCNN model loading
│
├── Docs/                              # Complete documentation (2500+ lines)
│   ├── README.md                      # Full technical reference
│   │                                  # - How MTCNN works
│   │                                  # - Transfer learning concepts
│   │                                  # - Architecture deep dive
│   │                                  # - Performance metrics
│   ├── QUICK_START.md                 # 5-minute setup guide
│   │                                  # - Installation steps
│   │                                  # - First run instructions
│   │                                  # - Troubleshooting tips
│   ├── DELIVERY_SUMMARY.md            # Feature overview
│   │                                  # - What's included
│   │                                  # - Feature checklist
│   │                                  # - Capability matrix
│   ├── IMPLEMENTATION_SUMMARY.md      # Technical deep dive
│   │                                  # - Component architecture
│   │                                  # - Device detection logic
│   │                                  # - Model integration
│   │                                  # - Performance optimization
│   ├── INDEX.md                       # Documentation navigation
│   │                                  # - Quick links to all docs
│   │                                  # - FAQs and troubleshooting
│   └── TEST_REPORT.md                 # Test results and bug fixes
│                                      # - All tests passing
│                                      # - Bug fixes applied
│                                      # - Verification results
│
├── models/                            # Machine learning models directory
│   └── .gitkeep                       # Placeholder (actual models auto-download)
│
├── assets/                            # Static assets
│   └── .gitkeep                       # Placeholder for demo images
│
├── scripts/                           # Utility scripts
│   └── create_test_image.py          # Generate synthetic test images for testing
│
├── test_outputs/                      # Test result artifacts
│   ├── detection_output.jpg          # Sample detection result (ignored by git)
│   └── test_detection_result.jpg     # Sample detection result (ignored by git)
│
├── .venv/                             # Virtual environment (ignored by git)
│
└── venv/                              # Alternate venv location (ignored by git)
```

---

## File Organization Rules

### ✅ In Scope (Version Controlled)

- **src/**: Core application code
- **utils/**: Shared utility functions
- **tests/**: Test files
- **Docs/**: Documentation
- **scripts/**: Utility scripts
- **assets/**: Static assets (images, icons)
- **Configuration**: requirements.txt, .gitignore, README.md

### ❌ Out of Scope (Ignored by Git)

- **venv/, .venv/**: Virtual environments
- **models/**: Downloaded pretrained models (too large)
- **test_outputs/**: Generated test results
- ****pycache**/**: Python bytecode
- **\*.log**: Log files
- **.vscode/, .idea/**: IDE configurations

---

## Component Overview

| Component               | Location                   | Purpose                                | Status      |
| ----------------------- | -------------------------- | -------------------------------------- | ----------- |
| **Main Application**    | `src/main.py`              | Entry point for webcam/image detection | ✅ Complete |
| **MTCNN Detector**      | `src/detector.py`          | Face detection using MTCNN             | ✅ Complete |
| **Configuration**       | `src/config.py`            | Centralized settings                   | ✅ Complete |
| **Performance Metrics** | `src/fps.py`               | FPS and latency tracking               | ✅ Complete |
| **Video Utilities**     | `utils/video.py`           | Camera capture and display             | ✅ Complete |
| **Logging**             | `utils/logger.py`          | Logging infrastructure                 | ✅ Complete |
| **System Tests**        | `tests/test_webcam.py`     | Full system verification               | ✅ Complete |
| **Component Tests**     | `tests/test_components.py` | Unit component testing                 | ✅ Complete |
| **Documentation**       | `Docs/`                    | User guides and technical refs         | ✅ Complete |

---

## Dependencies

```
opencv-python==4.8.1.78     # Video capture and display
torch==2.1.0                # Deep learning framework
torchvision==0.16.0         # Vision utilities
facenet-pytorch==2.5.3      # MTCNN implementation
numpy==1.24.3               # Numerical operations
Pillow==10.0.0              # Image processing
```

---

## Key Features

- ✅ Real-time face detection from webcam
- ✅ Static image processing mode
- ✅ GPU/CPU auto-detection
- ✅ Per-frame latency measurement
- ✅ Confidence scores for detections
- ✅ FPS counter with 30-frame rolling average
- ✅ Benchmark mode (dummy detector)
- ✅ Comprehensive logging
- ✅ Production-grade error handling
- ✅ Well-documented codebase

---

## Test Results

```
✓ Configuration loaded successfully
✓ FPS Counter: 312500.0 (rolling window)
✓ Latency Counter: 0.00ms
✓ Dummy Detector: returns 0 faces (expected)
✓ Video Utilities: imported successfully
✓ Logger: initialized successfully
✓ MTCNN Model: loaded on cpu
✓ MTCNN Detection: 0 faces found (synthetic test image)
✓ ALL TESTS PASSED!
```

---

## Quick Start

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/Scripts/activate  # Windows
source venv/bin/activate       # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Run

```bash
# Real-time webcam detection
python src/main.py

# Static image test
python src/main.py --mode image --image path/to/image.jpg

# Benchmark mode (no detection)
python src/main.py --model dummy
```

### Test

```bash
# All tests
python tests/test_components.py
python tests/test_webcam.py

# Specific test
pytest tests/test_webcam.py -v
```

---

## Project Statistics

| Metric              | Value                        |
| ------------------- | ---------------------------- |
| Total Source Files  | 8                            |
| Total Lines of Code | ~2000                        |
| Total Documentation | 2500+ lines                  |
| Test Coverage       | 6 components + 2 test suites |
| Dependencies        | 6 packages                   |
| Python Version      | 3.10+                        |
