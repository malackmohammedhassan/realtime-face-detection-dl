#!/usr/bin/env python3
"""
Example usage of Scratch CNN detector.

This script demonstrates how to use the ScratchCNNDetector
for face detection with various configurations.
"""

import sys
from pathlib import Path
import cv2
import numpy as np

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.scratch_cnn import ScratchCNNDetector


def example_1_basic_detection():
    """Example 1: Basic face detection on image."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Detection on Image")
    print("="*60)
    
    # Paths
    model_path = Path("models/scratch_cnn.pth")
    image_path = Path("assets/test.jpg")
    
    # Check if model exists
    if not model_path.exists():
        print(f"❌ Model not found at {model_path}")
        print("Train the model first: python scripts/train_scratch_cnn.py")
        return
    
    if not image_path.exists():
        print(f"⚠️  Image not found at {image_path}")
        print("Using webcam capture instead...")
        example_2_webcam_detection()
        return
    
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"❌ Failed to load image: {image_path}")
        return
    
    print(f"✓ Loaded image: {image_path}")
    print(f"  Size: {image.shape}")
    
    # Initialize detector
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✓ Using device: {device}")
    
    detector = ScratchCNNDetector(
        model_path=model_path,
        device=device,
        confidence_threshold=0.8
    )
    
    # Detect faces
    print("🔍 Running detection...")
    detections = detector.detect(image)
    
    print(f"✓ Found {len(detections)} face(s)")
    
    # Draw detections
    output = image.copy()
    for det in detections:
        x1, y1, x2, y2 = det['box']
        confidence = det['confidence']
        
        # Draw box
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw confidence
        label = f"Face: {confidence:.2f}"
        cv2.putText(
            output,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )
        
        print(f"  - Detection: box=({x1}, {y1}, {x2}, {y2}), confidence={confidence:.2f}")
    
    # Save output
    output_path = Path("test_outputs/example1_detections.jpg")
    output_path.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(output_path), output)
    print(f"✓ Output saved to: {output_path}")
    
    # Display
    cv2.imshow("Scratch CNN Detection", output)
    print("Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def example_2_webcam_detection():
    """Example 2: Real-time detection on webcam feed."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Real-Time Webcam Detection")
    print("="*60)
    
    model_path = Path("models/scratch_cnn.pth")
    
    if not model_path.exists():
        print(f"❌ Model not found at {model_path}")
        print("Train first: python scripts/train_scratch_cnn.py")
        return
    
    # Initialize detector
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✓ Using device: {device}")
    
    detector = ScratchCNNDetector(
        model_path=model_path,
        device=device,
        confidence_threshold=0.8
    )
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return
    
    print("✓ Webcam opened")
    print("Press 'Q' to quit...")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect
        detections = detector.detect(frame, scale=0.5)  # Scale for speed
        
        # Draw
        for det in detections:
            x1, y1, x2, y2 = det['box']
            confidence = det['confidence']
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{confidence:.2f}"
            cv2.putText(
                frame, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 0), 1
            )
        
        # Display
        cv2.imshow("Scratch CNN - Real-Time", frame)
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"  Processed {frame_count} frames, found {len(detections)} face(s)")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"✓ Processed {frame_count} total frames")


def example_3_batch_detection():
    """Example 3: Batch detection on multiple images."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Batch Detection")
    print("="*60)
    
    model_path = Path("models/scratch_cnn.pth")
    image_dir = Path("assets")
    
    if not model_path.exists():
        print(f"❌ Model not found")
        return
    
    if not image_dir.exists():
        print(f"❌ Image directory not found: {image_dir}")
        return
    
    # Initialize detector
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = ScratchCNNDetector(model_path=model_path, device=device)
    
    # Get all images
    images = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    
    if not images:
        print(f"❌ No images found in {image_dir}")
        return
    
    print(f"✓ Found {len(images)} image(s)")
    
    results = []
    
    for img_path in images:
        print(f"\nProcessing: {img_path.name}")
        
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  ⚠️  Failed to load")
            continue
        
        detections = detector.detect(image)
        results.append({
            'image': img_path.name,
            'faces': len(detections),
            'detections': detections
        })
        
        print(f"  ✓ Found {len(detections)} face(s)")
        
        for det in detections:
            conf = det['confidence']
            print(f"    - Confidence: {conf:.2f}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    total_faces = sum(r['faces'] for r in results)
    print(f"Total images: {len(results)}")
    print(f"Total faces found: {total_faces}")
    print(f"Average faces per image: {total_faces/len(results):.2f}")


def example_4_custom_parameters():
    """Example 4: Using custom detector parameters."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Custom Parameters")
    print("="*60)
    
    model_path = Path("models/scratch_cnn.pth")
    test_image = Path("assets/test.jpg")
    
    if not model_path.exists():
        print(f"❌ Model not found")
        return
    
    if not test_image.exists():
        print(f"❌ Test image not found")
        return
    
    image = cv2.imread(str(test_image))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test different configurations
    configs = [
        {'stride': 8, 'confidence_threshold': 0.8, 'name': 'Fine (slow, accurate)'},
        {'stride': 16, 'confidence_threshold': 0.8, 'name': 'Medium (balanced)'},
        {'stride': 32, 'confidence_threshold': 0.8, 'name': 'Coarse (fast, misses small)'},
        {'stride': 16, 'confidence_threshold': 0.9, 'name': 'High confidence (strict)'},
        {'stride': 16, 'confidence_threshold': 0.7, 'name': 'Low confidence (loose)'},
    ]
    
    for config in configs:
        print(f"\n{config['name']}")
        print(f"  Stride: {config['stride']}, Threshold: {config['confidence_threshold']}")
        
        detector = ScratchCNNDetector(
            model_path=model_path,
            device=device,
            window_stride=config['stride'],
            confidence_threshold=config['confidence_threshold']
        )
        
        detections = detector.detect(image)
        print(f"  → Found {len(detections)} face(s)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Scratch CNN examples")
    parser.add_argument(
        "--example",
        type=int,
        default=1,
        choices=[1, 2, 3, 4],
        help="Which example to run (1-4)"
    )
    
    args = parser.parse_args()
    
    print("Scratch CNN Detector Examples")
    print("=============================\n")
    
    if args.example == 1:
        example_1_basic_detection()
    elif args.example == 2:
        example_2_webcam_detection()
    elif args.example == 3:
        example_3_batch_detection()
    elif args.example == 4:
        example_4_custom_parameters()
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)
