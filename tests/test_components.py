#!/usr/bin/env python
"""Quick test script to verify all components work."""

import sys
import os

# Add project to path
sys.path.insert(0, os.path.dirname(__file__))

print("=" * 60)
print("COMPONENT TEST SUITE")
print("=" * 60)

# Test 1: Config
print("\n1. Testing Configuration...")
try:
    from src.config import Config
    Config.print_config()
    print("✓ Configuration loaded successfully")
except Exception as e:
    print(f"✗ Config error: {e}")
    sys.exit(1)

# Test 2: FPS Counter
print("\n2. Testing FPS Counter...")
try:
    from src.fps import FPSCounter, LatencyCounter
    fps = FPSCounter()
    lat = LatencyCounter()
    
    fps.update()
    lat.start()
    lat.end()
    
    print(f"✓ FPS: {fps.get_fps():.1f}")
    print(f"✓ Latency: {lat.get_avg_latency_ms():.2f}ms")
except Exception as e:
    print(f"✗ FPS/Latency error: {e}")
    sys.exit(1)

# Test 3: Dummy Detector
print("\n3. Testing Dummy Detector...")
try:
    from src.detector import FaceDetector
    import numpy as np
    
    detector = FaceDetector(model_name='dummy')
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = detector.detect(frame)
    
    print(f"✓ Dummy detector initialized")
    print(f"✓ Detection returned: {len(detections)} faces")
except Exception as e:
    print(f"✗ Dummy detector error: {e}")
    sys.exit(1)

# Test 4: Video utilities
print("\n4. Testing Video Utilities...")
try:
    from utils.video import WebcamCapture, FrameDisplay, put_text_on_frame
    print("✓ Video utilities imported successfully")
except Exception as e:
    print(f"✗ Video utilities error: {e}")
    sys.exit(1)

# Test 5: Logger
print("\n5. Testing Logger...")
try:
    from utils.logger import setup_logger
    logger = setup_logger("TestLogger", "INFO")
    logger.info("✓ Logger initialized successfully")
except Exception as e:
    print(f"✗ Logger error: {e}")
    sys.exit(1)

# Test 6: MTCNN (optional, takes time)
print("\n6. Testing MTCNN Detector (may take 30-60s on first run)...")
try:
    from src.detector import FaceDetector
    print("  Initializing MTCNN...")
    detector = FaceDetector(model_name='mtcnn')
    print(f"✓ MTCNN loaded on device: {detector.get_device()}")
    
    # Test detection with dummy frame
    import numpy as np
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 200
    detections = detector.detect(frame)
    print(f"✓ MTCNN detection test: {len(detections)} faces found")
except Exception as e:
    print(f"✗ MTCNN error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ ALL TESTS PASSED!")
print("=" * 60)
print("\nYou can now run:")
print("  python src/main.py                    # Real-time webcam (MTCNN)")
print("  python src/main.py --mode image       # Static image test")
print("  python src/main.py --model dummy      # Benchmark FPS")
print("  python src/main.py --model scratch_cnn # For testing trained scratch CNN")
print("\nTo train the scratch CNN:")
print("  python scripts/train_scratch_cnn.py --epochs 50")
