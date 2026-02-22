"""
Video capture and frame handling utilities.

Provides safe, error-tolerant webcam access with proper cleanup.
"""

import cv2
from typing import Optional, Tuple
import numpy as np


class WebcamCapture:
    """
    Safe wrapper around OpenCV VideoCapture.

    Handles camera initialization, error checking, and frame preprocessing.
    """

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ):
        """
        Initialize webcam capture.

        Args:
            camera_index: Index of camera (0 for default/built-in).
            width: Frame width in pixels.
            height: Frame height in pixels.
            fps: Target frames per second.

        Raises:
            RuntimeError: If camera cannot be opened.
        """
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps

        self.cap = cv2.VideoCapture(camera_index)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Failed to open camera at index {camera_index}. "
                "Check if camera is connected and not in use."
            )

        # Configure camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from the camera.

        Returns:
            (success, frame) tuple where success is True if frame was read
            and frame is the BGR image or None if read failed.
        """
        success, frame = self.cap.read()
        return success, frame if success else None

    def get_frame_size(self) -> Tuple[int, int]:
        """Get actual frame size (width, height)."""
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return w, h

    def release(self):
        """Release camera resources."""
        if self.cap:
            self.cap.release()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup."""
        self.release()


class FrameDisplay:
    """
    OpenCV-based frame display utility.

    Handles rendering frames and checking for user input.
    """

    def __init__(self, window_name: str = "Face Detection", fps: int = 30):
        """
        Initialize frame display.

        Args:
            window_name: Title of the display window.
            fps: Target display frame rate in FPS.
        """
        self.window_name = window_name
        self.fps = fps
        self.delay_ms = max(int(1000 / fps), 1)

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 600)

    def imshow(self, frame: np.ndarray):
        """
        Display frame in the window.

        Args:
            frame: BGR image to display.
        """
        cv2.imshow(self.window_name, frame)

    def wait_key(self, delay: Optional[int] = None) -> int:
        """
        Wait for keyboard input.

        Args:
            delay: Delay in milliseconds. If None, uses configured FPS delay.

        Returns:
            Key code pressed, or -1 if timeout.
        """
        if delay is None:
            delay = self.delay_ms
        return cv2.waitKey(delay) & 0xFF

    def close(self):
        """Close the display window."""
        cv2.destroyWindow(self.window_name)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup."""
        self.close()


def put_text_on_frame(
    frame: np.ndarray,
    text: str,
    position: Tuple[int, int] = (10, 30),
    font_scale: float = 1.0,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """
    Put text on frame.

    Args:
        frame: Input frame.
        text: Text to display.
        position: (x, y) position of text.
        font_scale: Font scale multiplier.
        color: BGR color tuple.
        thickness: Text thickness.

    Returns:
        Frame with text drawn.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, text, position, font, font_scale, color, thickness)
    return frame


def draw_bbox(
    frame: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw bounding box on frame.

    Args:
        frame: Input frame.
        x, y: Top-left corner coordinates.
        w, h: Width and height of box.
        color: BGR color tuple.
        thickness: Box line thickness.

    Returns:
        Frame with bounding box drawn.
    """
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
    return frame
