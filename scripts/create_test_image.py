"""
Create a simple test image with face-like regions.

This helper script generates a synthetic test image for the face detection system.
"""

import cv2
import numpy as np
from pathlib import Path

def create_test_image(output_path="assets/test.jpg"):
    """Create a simple test image with colored rectangles."""
    # Create blank image (480x640, BGR)
    img = np.ones((480, 640, 3), dtype=np.uint8) * 200
    
    # Draw some face-like rectangles
    # Face 1: left side
    cv2.rectangle(img, (100, 80), (280, 280), (180, 180, 220), -1)
    cv2.circle(img, (150, 140), 15, (100, 150, 200), -1)  # Left eye
    cv2.circle(img, (230, 140), 15, (100, 150, 200), -1)  # Right eye
    cv2.circle(img, (190, 200), 10, (150, 100, 150), -1)  # Nose
    
    # Face 2: right side
    cv2.rectangle(img, (360, 100), (540, 300), (220, 180, 180), -1)
    cv2.circle(img, (410, 160), 18, (100, 150, 200), -1)  # Left eye
    cv2.circle(img, (490, 160), 18, (100, 150, 200), -1)  # Right eye
    cv2.circle(img, (450, 220), 12, (150, 100, 150), -1)  # Nose
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(output_path), img)
    print(f"Test image created: {output_path}")

if __name__ == "__main__":
    create_test_image()
