"""
Real-time webcam face detection system.

Main entry point for the face detection pipeline.
Captures video from webcam, detects faces, and displays results with FPS and latency.

Usage:
    python src/main.py --mode webcam  # Real-time webcam detection
    python src/main.py --mode image   # Static image test (assets/test.jpg)

Exit (Webcam Mode):
    Press 'Q' to quit the application.
"""

import sys
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from config import Config
from fps import FPSCounter, LatencyCounter
from detector import FaceDetector
from utils.video import WebcamCapture, FrameDisplay, put_text_on_frame
from utils.logger import setup_logger

logger = setup_logger("FaceDetection", Config.LOG_LEVEL)


class RealtimeFaceDetector:
    """
    Main application class for real-time face detection.

    Orchestrates camera input, face detection, and visualization.
    """

    def __init__(self, model_name: str = "mtcnn"):
        """
        Initialize the detection system.

        Args:
            model_name: Detection model to use ("mtcnn" or "dummy")
        """
        self.fps_counter = FPSCounter(window_size=30)
        self.latency_counter = LatencyCounter(window_size=30)
        self.detector = FaceDetector(model_name=model_name)
        self.running = True

        logger.info("Real-time Face Detection System Initialized")
        logger.info(f"Using device: {self.detector.get_device()}")
        logger.info(f"Detection model: {self.detector.get_model_name()}")

    def run_webcam(self):
        """
        Run real-time face detection on webcam feed.

        Opens webcam, processes frames in real-time, and displays results.
        Handles graceful shutdown on error or user input.
        """
        try:
            # Initialize webcam with context manager for automatic cleanup
            with WebcamCapture(
                camera_index=0,
                width=Config.WEBCAM_WIDTH,
                height=Config.WEBCAM_HEIGHT,
                fps=Config.WEBCAM_FPS,
            ) as webcam:

                # Initialize display with context manager
                with FrameDisplay(
                    window_name="Real-Time Face Detection",
                    fps=Config.WEBCAM_FPS,
                ) as display:

                    actual_width, actual_height = webcam.get_frame_size()
                    logger.info(
                        f"Webcam opened successfully: {actual_width}x{actual_height}"
                    )

                    self._process_webcam_frames(webcam, display)

        except RuntimeError as e:
            logger.error(f"Camera error: {e}")
            sys.exit(1)
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            sys.exit(1)
        finally:
            self._cleanup()

    def run_image(self, image_path: str = "assets/test.jpg"):
        """
        Run face detection on a static image.

        Args:
            image_path: Path to input image

        Returns:
            True if successful, False otherwise
        """
        image_path = Path(image_path)

        if not image_path.exists():
            logger.error(f"Image not found: {image_path}")
            return False

        try:
            logger.info(f"Loading image: {image_path}")
            frame = cv2.imread(str(image_path))

            if frame is None:
                logger.error(f"Failed to read image: {image_path}")
                return False

            logger.info(f"Image loaded: {frame.shape}")

            # Run detection
            logger.info("Running face detection...")
            self.latency_counter.start()
            detections = self.detector.detect(
                frame, scale=Config.FRAME_SCALE
            )
            latency_ms = self.latency_counter.end()

            logger.info(f"Detection completed in {latency_ms:.2f}ms")
            logger.info(f"Found {len(detections)} face(s)")

            # Draw detections
            frame_output = self._draw_detections(frame, detections)

            # Add latency info
            latency_text = f"Latency: {latency_ms:.2f}ms"
            frame_output = put_text_on_frame(
                frame_output,
                latency_text,
                position=(10, 60),
                font_scale=0.7,
                color=(0, 255, 255),
                thickness=2,
            )

            # Save output image
            output_path = Path("detection_output.jpg")
            cv2.imwrite(str(output_path), frame_output)
            logger.info(f"Output saved to: {output_path.absolute()}")

            # Display result
            display = cv2.namedWindow(
                "Face Detection - Static Image", cv2.WINDOW_NORMAL
            )
            cv2.imshow("Face Detection - Static Image", frame_output)
            logger.info("Press any key to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            return True

        except Exception as e:
            logger.error(f"Error processing image: {e}", exc_info=True)
            return False

    def _process_webcam_frames(
        self, webcam: WebcamCapture, display: FrameDisplay
    ):
        """
        Process frames from webcam in a loop.

        Args:
            webcam: WebcamCapture instance
            display: FrameDisplay instance
        """
        frame_count = 0

        while self.running:
            # Read frame from webcam
            success, frame = webcam.read()
            if not success:
                logger.warning("Failed to read frame, retrying...")
                continue

            # Update FPS counter
            fps = self.fps_counter.update()
            frame_count += 1

            # Run face detection
            if Config.DISPLAY_DETECTIONS:
                self.latency_counter.start()
                detections = self.detector.detect(
                    frame, scale=Config.FRAME_SCALE
                )
                latency_ms = self.latency_counter.end()

                frame = self._draw_detections(frame, detections)

                # Draw latency on frame
                if Config.MEASURE_LATENCY:
                    latency_text = (
                        f"Inference: {latency_ms:.2f}ms "
                        f"(Avg: {self.latency_counter.get_avg_latency_ms():.2f}ms)"
                    )
                    frame = put_text_on_frame(
                        frame,
                        latency_text,
                        position=(10, 60),
                        font_scale=0.6,
                        color=(0, 255, 255),
                        thickness=1,
                    )

            # Draw FPS on frame
            if Config.DISPLAY_FPS:
                fps_text = f"FPS: {fps:.1f}"
                frame = put_text_on_frame(
                    frame,
                    fps_text,
                    position=(10, 30),
                    font_scale=1.0,
                    color=(0, 255, 0),
                    thickness=2,
                )

            # Display frame
            display.imshow(frame)

            # Check for exit key ('Q' or 'q')
            key = display.wait_key()
            if key == ord("q") or key == ord("Q"):
                logger.info("Exit key pressed")
                self.running = False

            # Log progress every 100 frames
            if frame_count % 100 == 0:
                logger.debug(
                    f"Processed {frame_count} frames @ {fps:.1f} FPS, "
                    f"Avg latency: {self.latency_counter.get_avg_latency_ms():.2f}ms"
                )

    def _draw_detections(
        self, frame: np.ndarray, detections: list
    ) -> np.ndarray:
        """
        Draw face detections on frame.

        Args:
            frame: Input frame
            detections: List of (x, y, w, h, confidence) tuples

        Returns:
            Frame with detections drawn
        """
        for x, y, w, h, confidence in detections:
            x, y, w, h = int(x), int(y), int(w), int(h)

            # Draw bounding box (green rectangle)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Draw confidence score above box
            label = f"Face {confidence:.2f}"
            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
            )

            # Background for text
            cv2.rectangle(
                frame,
                (x, y - text_h - baseline - 5),
                (x + text_w, y - baseline),
                (0, 255, 0),
                -1,
            )

            # Text
            cv2.putText(
                frame,
                label,
                (x, y - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                1,
            )

        return frame

    def _cleanup(self):
        """Cleanup resources."""
        cv2.destroyAllWindows()
        logger.info("Application closed")


def main():
    """Entry point for the application."""
    parser = argparse.ArgumentParser(
        description="Real-time face detection system"
    )
    parser.add_argument(
        "--mode",
        choices=["webcam", "image"],
        default="webcam",
        help="Detection mode: webcam for real-time, image for static test",
    )
    parser.add_argument(
        "--model",
        choices=["mtcnn", "dummy"],
        default="mtcnn",
        help="Detection model to use",
    )
    parser.add_argument(
        "--image",
        type=str,
        default="assets/test.jpg",
        help="Path to test image (used with --mode image)",
    )

    args = parser.parse_args()

    Config.print_config()

    detector = RealtimeFaceDetector(model_name=args.model)

    if args.mode == "image":
        success = detector.run_image(args.image)
        sys.exit(0 if success else 1)
    else:  # webcam
        detector.run_webcam()


if __name__ == "__main__":
    main()
