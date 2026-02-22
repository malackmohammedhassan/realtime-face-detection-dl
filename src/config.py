"""
Configuration module for real-time face detection system.

This module centralizes all configuration settings including device
selection (CPU/GPU), detection thresholds, and other parameters.
"""

import torch
from pathlib import Path


class Config:
    """Centralized configuration for the face detection system."""

    # ===== Device Configuration =====
    # Automatically selects GPU if available, falls back to CPU
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===== Webcam Configuration =====
    WEBCAM_WIDTH = 640
    WEBCAM_HEIGHT = 480
    WEBCAM_FPS = 30

    # ===== Detection Configuration =====
    # Model type: "mtcnn" (MultiTask Cascaded CNN) - fast and accurate
    # MTCNN works in 3 stages: P-Net (propose), R-Net (refine), O-Net (output)
    DETECTION_MODEL = "mtcnn"

    # Confidence threshold for face detections (0.0 to 1.0)
    # Higher = fewer false positives, might miss small faces
    CONFIDENCE_THRESHOLD = 0.95

    # Non-maximum suppression (NMS) threshold for overlapping detections
    NMS_THRESHOLD = 0.4

    # Frame resize scale for faster inference (1.0 = original size)
    # 0.5 = half resolution = 4x faster, but less accurate for small faces
    FRAME_SCALE = 1.0

    # Use half precision (FP16) if GPU supports it (faster, lower memory)
    USE_HALF_PRECISION = False

    # Measure per-frame latency
    MEASURE_LATENCY = True

    # ===== Display Configuration =====
    DISPLAY_FPS = True
    DISPLAY_DETECTIONS = True

    # ===== Paths =====
    PROJECT_ROOT = Path(__file__).parent.parent
    MODELS_DIR = PROJECT_ROOT / "models"
    ASSETS_DIR = PROJECT_ROOT / "assets"

    # ===== Logging =====
    LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL

    @classmethod
    def print_config(cls):
        """Print current configuration settings."""
        print("=== Face Detection Configuration ===")
        print(f"Device: {cls.DEVICE}")
        print(f"Detection Model: {cls.DETECTION_MODEL}")
        print(f"Webcam Resolution: {cls.WEBCAM_WIDTH}x{cls.WEBCAM_HEIGHT}")
        print(f"Webcam FPS: {cls.WEBCAM_FPS}")
        print(f"Confidence Threshold: {cls.CONFIDENCE_THRESHOLD}")
        print(f"Frame Scale: {cls.FRAME_SCALE}")
        print(f"Use FP16: {cls.USE_HALF_PRECISION}")
        print(f"Measure Latency: {cls.MEASURE_LATENCY}")
        print(f"Log Level: {cls.LOG_LEVEL}")
        print("===================================\n")
