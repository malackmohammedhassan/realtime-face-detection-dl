"""
Face detection using MTCNN (Multi-task Cascaded Convolutional Networks).

This module implements real-time face detection using pretrained MTCNN,
a state-of-the-art cascade-based face detector.

## How MTCNN Works

MTCNN is a cascade of 3 CNNs that work sequentially:

1. **P-Net (Proposal Network)**
   - Input: Image at multiple scales
   - Output: Face candidate windows
   - Fast, generates many proposals (high recall, lower precision)

2. **R-Net (Refinement Network)**
   - Input: Candidate windows from P-Net
   - Output: Filtered candidates (removes false positives)
   - Refines proposals by classifying and adjusting bbox coordinates

3. **O-Net (Output Network)**
   - Input: Refined candidates from R-Net
   - Output: Final face detections + landmark points
   - Produces high-quality detections with face landmarks (optional)

## Why Pretrained Models (Transfer Learning)

- **Training MTCNN from scratch requires:**
  - Massive face datasets (millions of images)
  - Weeks of GPU training time
  - Tedious hyperparameter tuning

- **Using pretrained models saves us:**
  - Hours of manual training
  - Hundreds of dollars in compute costs
  - Knowledge learned from millions of faces worldwide

- **How it works:**
  - Model trained on WIDER-FACE (huge face detection dataset)
  - Learned features generalize well to any face image
  - We use these features directly (no retraining needed)
  - This is the power of transfer learning in deep learning

## GPU Acceleration

- Inference runs on GPU if CUDA available → 2-4x faster
- CPU fallback for compatibility
- FP16 (half precision) reduces memory and speeds up computation
"""

from typing import List, Tuple, Optional
import numpy as np
from abc import ABC, abstractmethod
import time
import logging
from pathlib import Path

try:
    import torch
    from facenet_pytorch import MTCNN as FacenetMTCNN
    HAS_MTCNN = True
except ImportError:
    HAS_MTCNN = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


logger = logging.getLogger(__name__)


class BaseDetector(ABC):
    """Abstract base class for face detectors."""

    @abstractmethod
    def detect(
        self, frame: np.ndarray
    ) -> List[Tuple[float, float, float, float, float]]:
        """
        Detect faces in the frame.

        Args:
            frame: Input frame (BGR format from OpenCV).

        Returns:
            List of detections as (x, y, w, h, confidence) tuples.
        """
        pass


class MTCNNDetector(BaseDetector):
    """
    MTCNN-based face detector.

    Multi-task Cascaded Convolutional Networks for robust, fast face detection.
    Supports GPU acceleration (CUDA) with automatic CPU fallback.
    """

    def __init__(
        self,
        device: torch.device,
        confidence_threshold: float = 0.95,
        keep_all: bool = False,
    ):
        """
        Initialize MTCNN detector.

        Args:
            device: torch.device (cuda or cpu)
            confidence_threshold: Minimum confidence for detection (0.0-1.0)
            keep_all: If True, return all detections; if False, filter by threshold

        Raises:
            RuntimeError: If MTCNN library is not installed
        """
        if not HAS_MTCNN:
            raise RuntimeError(
                "MTCNN not installed. Install with: pip install facenet-pytorch"
            )

        self.device = device
        self.confidence_threshold = confidence_threshold
        self.keep_all = keep_all

        try:
            # Load pretrained MTCNN model
            # Model is automatically downloaded on first use (~100MB)
            # Cached in ~/.cache/torch/checkpoints/ for faster subsequent loads
            self.model = FacenetMTCNN(
                image_size=160,  # Input size for O-Net
                margin=0,  # No margin around detected faces
                min_face_size=20,  # Minimum face size in pixels
                thresholds=[0.6, 0.7, 0.7],  # P-Net, R-Net, O-Net thresholds
                factor=0.709,  # Scale factor between image pyramid levels
                post_process=True,  # Enable coordinate refinement
                device=device,
                keep_all=keep_all,
            )

            logger.info(f"MTCNN model loaded on device: {device}")
            logger.info(
                "Note: Model weights (~100MB) auto-downloaded on first run."
            )

        except Exception as e:
            logger.error(f"Failed to load MTCNN model: {e}")
            raise RuntimeError(f"MTCNN initialization failed: {e}") from e

    def detect(
        self, frame: np.ndarray, scale: float = 1.0
    ) -> List[Tuple[float, float, float, float, float]]:
        """
        Detect faces in frame using MTCNN.

        Args:
            frame: Input frame (BGR from OpenCV, HxWx3)
            scale: Resize frame by this factor before inference (0.0-1.0)
                  - <1.0 increases speed but reduces accuracy for small faces
                  - Use for real-time processing on slower hardware

        Returns:
            List of (x, y, w, h, confidence) tuples where:
            - (x, y) = top-left corner
            - (w, h) = width and height
            - confidence = detection score (0.0-1.0)
        """
        if frame is None or frame.size == 0:
            return []

        try:
            # Resize frame if scale < 1.0 (for performance)
            original_h, original_w = frame.shape[:2]
            if scale < 1.0 and scale > 0.0:
                new_w = int(original_w * scale)
                new_h = int(original_h * scale)
                import cv2

                frame_scaled = cv2.resize(frame, (new_w, new_h))
            else:
                frame_scaled = frame

            # Convert BGR (OpenCV) to RGB (PyTorch/MTCNN convention)
            frame_rgb = frame_scaled[..., ::-1]  # BGR to RGB

            # Run MTCNN detection
            # Returns: boxes (Nx4), probs (N,), landmarks (Nx10, optional)
            with torch.no_grad():
                boxes, probs, landmarks = self.model.detect(
                    frame_rgb, landmarks=True
                )

            detections = []

            if boxes is not None and probs is not None:
                for bbox, prob in zip(boxes, probs):
                    # Scale back to original frame size if we resized
                    if scale < 1.0 and scale > 0.0:
                        bbox = bbox / scale

                    confidence = float(prob)

                    # Filter by confidence threshold
                    if confidence >= self.confidence_threshold:
                        x, y, x2, y2 = bbox
                        w = x2 - x
                        h = y2 - y
                        detections.append(
                            (
                                float(x),
                                float(y),
                                float(w),
                                float(h),
                                confidence,
                            )
                        )

            return detections

        except Exception as e:
            logger.error(f"Detection error: {e}")
            return []


class DummyDetector(BaseDetector):
    """
    Placeholder detector for testing pipeline.

    Returns empty detections. Useful for benchmarking camera FPS
    without running inference.
    """

    def __init__(self):
        """Initialize dummy detector."""
        self.name = "DummyDetector"
        logger.info("Dummy detector initialized (no inference)")

    def detect(self, frame: np.ndarray, scale: float = 1.0) -> List[Tuple[float, float, float, float, float]]:
        """
        Dummy detection - returns empty list.

        Args:
            frame: Input frame (unused)
            scale: Frame scale parameter (unused, for interface compatibility)

        Returns:
            Empty list.
        """
        return []


class ScratchCNNDetectorWrapper(BaseDetector):
    """
    Wrapper for scratch CNN face detector.
    
    Uses a lightweight from-scratch CNN trained on face vs non-face classification.
    Employs sliding window approach for face localization.
    
    Architecture:
        - 3 convolutional layers (32, 64, 128 filters)
        - 2 fully connected layers
        - Input: 64x64 RGB windows
        - Output: Binary classification (face/non-face)
    """
    
    def __init__(
        self,
        device: torch.device,
        model_path: Optional[Path] = None,
        confidence_threshold: float = 0.8,
        window_stride: int = 16
    ):
        """
        Initialize scratch CNN detector.
        
        Args:
            device: torch.device to use
            model_path: Path to trained model checkpoint. 
                       If None, looks for models/scratch_cnn.pth
            confidence_threshold: Confidence threshold for face detection (0-1)
            window_stride: Stride for sliding window (smaller = more detections, slower)
        
        Raises:
            RuntimeError: If model path doesn't exist or model fails to load
        """
        self.device = device
        self.window_stride = window_stride
        
        # Auto-detect model path if not provided
        if model_path is None:
            model_path = Path(__file__).parent.parent / "models" / "scratch_cnn.pth"
        else:
            model_path = Path(model_path)
        
        if not model_path.exists():
            raise RuntimeError(
                f"Scratch CNN model not found at {model_path}. "
                f"Please train first: python scripts/train_scratch_cnn.py"
            )
        
        try:
            # Import here to avoid circular imports
            from scratch_cnn import ScratchCNNDetector as ScratchCNNInferenceModel
            
            self.detector = ScratchCNNInferenceModel(
                model_path=model_path,
                device=device,
                confidence_threshold=confidence_threshold,
                stride=window_stride
            )
            
            logger.info(f"Scratch CNN model loaded from {model_path}")
            logger.info(f"Using device: {device}")
            
        except Exception as e:
            logger.error(f"Failed to load scratch CNN model: {e}")
            raise RuntimeError(f"Scratch CNN initialization failed: {e}") from e
    
    def detect(
        self, frame: np.ndarray, scale: float = 1.0
    ) -> List[Tuple[float, float, float, float, float]]:
        """
        Detect faces using scratch CNN with sliding window.
        
        Args:
            frame: Input frame in BGR format (from OpenCV)
            scale: Frame scale for inference (0.5-1.0 for speed)
        
        Returns:
            List of detections as (x, y, w, h, confidence) tuples
        """
        try:
            # Detect using sliding window
            detections_raw = self.detector.detect(frame, scale=scale)
            
            # Convert from (x1, y1, x2, y2, confidence) to (x, y, w, h, confidence)
            detections = []
            for det in detections_raw:
                x1, y1, x2, y2 = det['box']
                confidence = det['confidence']
                
                w = x2 - x1
                h = y2 - y1
                
                detections.append((float(x1), float(y1), float(w), float(h), confidence))
            
            return detections
        
        except Exception as e:
            logger.error(f"Scratch CNN detection error: {e}")
            return []


class FaceDetector:
    """
    Main face detector wrapper.

    Provides a unified interface for different detection backends.
    Handles model initialization, error checking, and inference.
    """

    def __init__(self, model_name: str = "mtcnn", device: Optional[torch.device] = None):
        """
        Initialize face detector.

        Args:
            model_name: Name of detection model ("mtcnn", "dummy", or "scratch_cnn")
            device: torch.device (cuda or cpu). If None, auto-detects.

        Raises:
            ValueError: If model_name is not supported
            RuntimeError: If model fails to load
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.model_name = model_name

        try:
            if model_name == "mtcnn":
                self.detector = MTCNNDetector(device=device)
            elif model_name == "dummy":
                self.detector = DummyDetector()
            elif model_name == "scratch_cnn":
                self.detector = ScratchCNNDetectorWrapper(device=device)
            else:
                raise ValueError(
                    f"Model '{model_name}' not supported. "
                    f"Available models: 'mtcnn', 'dummy', 'scratch_cnn'"
                )

            logger.info(f"Face detector '{model_name}' initialized on {device}")

        except Exception as e:
            logger.error(f"Failed to initialize detector '{model_name}': {e}")
            raise

    def detect(
        self, frame: np.ndarray, scale: float = 1.0
    ) -> List[Tuple[float, float, float, float, float]]:
        """
        Detect faces in frame.

        Args:
            frame: Input frame in BGR format
            scale: Frame resize scale (0.0-1.0) for performance

        Returns:
            List of bounding boxes (x, y, w, h, confidence)
        """
        return self.detector.detect(frame, scale)

    def get_model_name(self) -> str:
        """Get name of current detection model."""
        return self.model_name

    def get_device(self) -> str:
        """Get device being used for inference."""
        return str(self.device)

