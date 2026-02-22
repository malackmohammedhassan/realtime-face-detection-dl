# Test Assets Directory

This directory stores test images, test videos, and other static assets for development and testing.

## Usage

Add test images here for testing the detection pipeline:

```
assets/
├── test.jpg              # Sample face image for testing
├── sample_video.mp4      # Test video file
└── reference/            # Reference images for comparison
    └── expected_output.jpg
```

## Testing with Static Images

You can modify `src/main.py` to test with static images before connecting to a live webcam:

```python
import cv2
from src.detector import FaceDetector

image = cv2.imread("assets/test.jpg")
detector = FaceDetector(model_name="mtcnn")
detections = detector.detect(image)
```

## Git Ignore

Large binary files should not be committed to git. Keep only small test samples in version control, or use `.gitignore` to exclude them.
