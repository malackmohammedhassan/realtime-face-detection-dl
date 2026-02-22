"""
Comprehensive verification test suite for face detection system.

Tests:
- OpenCV installation and version
- PyTorch and CUDA availability
- Webcam functionality
- Face detection with MTCNN
- Static image detection test

Usage:
    python tests/test_webcam.py

Exit:
    Press 'Q' to exit webcam test.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from src.config import Config
from src.detector import FaceDetector
from utils.video import WebcamCapture, FrameDisplay, put_text_on_frame
from utils.logger import setup_logger

logger = setup_logger("WebcamTest", "INFO")


def test_opencv():
    """Test OpenCV installation and version."""
    logger.info(f"OpenCV version: {cv2.__version__}")
    logger.info("✓ OpenCV imported successfully")


def test_torch():
    """Test PyTorch and CUDA availability."""
    try:
        import torch

        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        logger.info("✓ PyTorch imported successfully")
    except ImportError:
        logger.warning("PyTorch not yet installed")


def test_detector_init():
    """Test MTCNN detector initialization."""
    try:
        logger.info("\n--- Testing MTCNN Detector Initialization ---")
        detector = FaceDetector(model_name="mtcnn")
        logger.info(f"✓ Detector initialized: {detector.get_model_name()}")
        logger.info(f"✓ Device: {detector.get_device()}")
        return True
    except Exception as e:
        logger.error(f"✗ Detector initialization failed: {e}")
        return False


def test_static_image(image_path: str = "assets/test.jpg"):
    """
    Test face detection on a static image.

    Args:
        image_path: Path to test image

    Returns:
        True if test passed, False otherwise
    """
    try:
        image_path = Path(image_path)

        # Create a simple test image if it doesn't exist
        if not image_path.exists():
            logger.info(
                f"\n⚠ Test image not found: {image_path}"
            )
            logger.info("Creating a synthetic test image...")

            # Create a simple image with a white face-like region
            image_path.parent.mkdir(exist_ok=True)
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            # Draw a simple rectangle to simulate a face region
            cv2.rectangle(img, (150, 100), (350, 300), (200, 200, 200), -1)
            cv2.imwrite(str(image_path), img)
            logger.info(f"Created synthetic test image: {image_path}")

        logger.info(f"\n--- Testing Detection on Static Image ---")
        logger.info(f"Loading image: {image_path}")

        frame = cv2.imread(str(image_path))
        if frame is None:
            logger.error(f"Failed to load image: {image_path}")
            return False

        logger.info(f"Image size: {frame.shape}")

        # Initialize detector (will download MTCNN model on first run)
        logger.info("Initializing MTCNN detector (downloading model if needed)...")
        detector = FaceDetector(model_name="mtcnn")

        # Run detection
        logger.info("Running face detection...")
        detections = detector.detect(frame)

        logger.info(f"✓ Detection completed")
        logger.info(f"✓ Found {len(detections)} face(s)")

        for i, (x, y, w, h, conf) in enumerate(detections):
            logger.info(
                f"  Face {i + 1}: pos=({x:.0f}, {y:.0f}), "
                f"size=({w:.0f}x{h:.0f}), confidence={conf:.3f}"
            )

        # Draw detections on image
        frame_with_boxes = frame.copy()
        for x, y, w, h, conf in detections:
            x, y, w, h = int(x), int(y), int(w), int(h)
            cv2.rectangle(
                frame_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2
            )
            label = f"Face {conf:.2f}"
            cv2.putText(
                frame_with_boxes,
                label,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        # Save result
        output_path = Path("test_detection_result.jpg")
        cv2.imwrite(str(output_path), frame_with_boxes)
        logger.info(f"✓ Result saved to: {output_path.absolute()}")

        return True

    except Exception as e:
        logger.error(f"✗ Static image test failed: {e}", exc_info=True)
        return False


def test_webcam_capture():
    """Test webcam capture functionality."""
    logger.info("\n--- Testing Webcam Capture ---")
    logger.info("Press 'Q' to exit webcam test")

    try:
        with WebcamCapture(
            camera_index=0,
            width=Config.WEBCAM_WIDTH,
            height=Config.WEBCAM_HEIGHT,
            fps=Config.WEBCAM_FPS,
        ) as webcam:

            actual_width, actual_height = webcam.get_frame_size()
            logger.info(f"✓ Camera opened: {actual_width}x{actual_height}")

            with FrameDisplay(
                window_name="Webcam Test", fps=Config.WEBCAM_FPS
            ) as display:

                frame_count = 0

                while True:
                    success, frame = webcam.read()

                    if not success:
                        logger.error("Failed to read frame")
                        break

                    frame_count += 1

                    # Display frame info
                    info_text = (
                        f"Frame: {frame_count} | "
                        f"Resolution: {actual_width}x{actual_height}"
                    )
                    frame = put_text_on_frame(
                        frame,
                        info_text,
                        position=(10, 30),
                        font_scale=0.7,
                        color=(0, 255, 0),
                        thickness=2,
                    )

                    display.imshow(frame)

                    key = display.wait_key()
                    if key == ord("q") or key == ord("Q"):
                        logger.info("Webcam test completed")
                        break

            logger.info(f"✓ Webcam test passed. Captured {frame_count} frames")
            return True

    except RuntimeError as e:
        logger.error(f"✗ Webcam test failed: {e}")
        return False


def main():
    """Run all verification tests."""
    logger.info("=" * 60)
    logger.info("Real-Time Face Detection - Comprehensive Test Suite")
    logger.info("=" * 60 + "\n")

    # Test 1: Import checks
    test_opencv()
    test_torch()

    # Test 2: Detector initialization
    logger.info("")
    detector_ok = test_detector_init()

    # Test 3: Static image detection
    logger.info("")
    image_ok = test_static_image()

    # Test 4: Webcam
    logger.info("")
    webcam_ok = test_webcam_capture()

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Detector Initialization: {'✓ PASS' if detector_ok else '✗ FAIL'}")
    logger.info(f"Static Image Detection:  {'✓ PASS' if image_ok else '✗ FAIL'}")
    logger.info(f"Webcam Capture:          {'✓ PASS' if webcam_ok else '✗ FAIL'}")

    all_pass = detector_ok and image_ok and webcam_ok
    if all_pass:
        logger.info("\n✓ All tests passed! System ready for deployment.")
    else:
        logger.error("\n✗ Some tests failed. Check the errors above.")

    logger.info("=" * 60)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
