"""
Scratch CNN: From-scratch face vs non-face binary classifier.

Architecture:
    - 3 convolutional layers with batch normalization
    - 2 fully connected layers
    - Input: 64x64 RGB images
    - Output: Binary classification (face/non-face)

Training:
    - CrossEntropyLoss
    - Adam optimizer
    - Data augmentation for robustness
    - Early stopping based on validation accuracy

Usage:
    Training: python scripts/train_scratch_cnn.py
    Inference: See ScratchCNNDetector class
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
import numpy as np
from PIL import Image
import logging
from typing import Tuple, List

logger = logging.getLogger(__name__)


class TinyCNN(nn.Module):
    """
    Lightweight CNN for face detection.
    
    Designed for 64x64 RGB images with minimal parameters.
    Architecture:
        - Conv Layer 1: 3 -> 32 channels, 3x3 kernel
        - Conv Layer 2: 32 -> 64 channels, 3x3 kernel
        - Conv Layer 3: 64 -> 128 channels, 3x3 kernel
        - FC Layer 1: 128*8*8 -> 256
        - FC Layer 2: 256 -> 2 (binary classification)
    """
    
    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.5):
        """
        Initialize the CNN.
        
        Args:
            num_classes: Number of output classes (default: 2 for face/non-face)
            dropout_rate: Dropout probability for regularization
        """
        super(TinyCNN, self).__init__()
        
        # Convolutional layers with batch normalization
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)  # 64x64 -> 32x32
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)  # 32x32 -> 16x16
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)  # 16x16 -> 8x8
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, num_classes)
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 64, 64)
        
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        # Conv block 1
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        # Conv block 2
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        # Conv block 3
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class FaceNonFaceDataset(Dataset):
    """
    Dataset loader for face vs non-face classification.
    
    Directory structure expected:
        data/train/face/*.jpg
        data/train/non_face/*.jpg
        data/val/face/*.jpg
        data/val/non_face/*.jpg
    """
    
    def __init__(
        self,
        data_dir: Path,
        split: str = "train",
        img_size: int = 64,
        augmentation: bool = True
    ):
        """
        Initialize dataset.
        
        Args:
            data_dir: Path to data directory (e.g., Path("data"))
            split: "train" or "val"
            img_size: Target image size (default: 64)
            augmentation: Enable data augmentation for training
        """
        self.data_dir = Path(data_dir) / split
        self.img_size = img_size
        self.augmentation = augmentation and split == "train"
        
        # Base transforms (all splits)
        self.base_transforms = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Augmentation transforms (training only)
        self.aug_transforms = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Load image paths and labels
        self.images = []
        self.labels = []
        
        # Load face images (label=1)
        face_dir = self.data_dir / "face"
        if face_dir.exists():
            for img_path in face_dir.glob("*.jpg"):
                self.images.append(img_path)
                self.labels.append(1)  # Face
            for img_path in face_dir.glob("*.png"):
                self.images.append(img_path)
                self.labels.append(1)  # Face
        
        # Load non-face images (label=0)
        non_face_dir = self.data_dir / "non_face"
        if non_face_dir.exists():
            for img_path in non_face_dir.glob("*.jpg"):
                self.images.append(img_path)
                self.labels.append(0)  # Non-face
            for img_path in non_face_dir.glob("*.png"):
                self.images.append(img_path)
                self.labels.append(0)  # Non-face
        
        logger.info(
            f"Loaded {len(self.images)} images from {split} set "
            f"({sum(1 for l in self.labels if l == 1)} faces, "
            f"{sum(1 for l in self.labels if l == 0)} non-faces)"
        )
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get an image and its label.
        
        Args:
            idx: Index in dataset
        
        Returns:
            Tuple of (image tensor, label)
        """
        try:
            img_path = self.images[idx]
            label = self.labels[idx]
            
            # Load image
            img = Image.open(img_path).convert("RGB")
            
            # Apply transforms
            transforms_to_use = self.aug_transforms if self.augmentation else self.base_transforms
            img = transforms_to_use(img)
            
            return img, label
        
        except Exception as e:
            logger.error(f"Error loading image {self.images[idx]}: {e}")
            # Return a black image as fallback
            black_img = torch.zeros(3, self.img_size, self.img_size)
            return black_img, self.labels[idx]


class ScratchCNNTrainer:
    """
    Trainer class for the scratch CNN model.
    
    Handles training loop, validation, and model checkpointing.
    """
    
    def __init__(
        self,
        model: TinyCNN,
        device: torch.device,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4
    ):
        """
        Initialize trainer.
        
        Args:
            model: TinyCNN model instance
            device: torch.device to train on
            learning_rate: Adam learning rate
            weight_decay: L2 regularization weight
        """
        self.model = model.to(device)
        self.device = device
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.best_accuracy = 0.0
        self.patience = 10
        self.patience_counter = 0
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training set
        
        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate on validation set.
        
        Args:
            val_loader: DataLoader for validation set
        
        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, path: Path, is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
            is_best: If True, marks this as best model
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_accuracy': self.best_accuracy
        }
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: Path):
        """
        Load model checkpoint.
        
        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_accuracy = checkpoint.get('best_accuracy', 0.0)
        logger.info(f"Checkpoint loaded from {path}")


class ScratchCNNDetector:
    """
    Inference wrapper for scratch CNN face detector.
    
    Uses sliding window approach to detect faces in images.
    """
    
    def __init__(
        self,
        model_path: Path,
        device: torch.device = None,
        window_size: int = 64,
        stride: int = 16,
        confidence_threshold: float = 0.8
    ):
        """
        Initialize detector.
        
        Args:
            model_path: Path to trained model checkpoint
            device: torch device (auto-detect if None)
            window_size: Size of sliding window (default: 64)
            stride: Stride of sliding window (default: 16)
            confidence_threshold: Confidence threshold for face detection (0-1)
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.device = device
        self.window_size = window_size
        self.stride = stride
        self.confidence_threshold = confidence_threshold
        
        # Load model
        self.model = TinyCNN(num_classes=2)
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        # Transforms (ImageNet normalization)
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        logger.info(f"ScratchCNN model loaded from {model_path}")
    
    def detect(self, image: np.ndarray, scale: float = 1.0) -> List[dict]:
        """
        Detect faces in image using sliding window.
        
        Args:
            image: Input image (numpy array, BGR from OpenCV)
            scale: Image scale factor for speed/accuracy trade-off
        
        Returns:
            List of detection dicts with keys:
                - 'box': (x1, y1, x2, y2)
                - 'confidence': float (0-1)
        """
        h, w = image.shape[:2]
        
        # Scale image if needed
        if scale != 1.0:
            image = cv2.resize(
                image,
                (int(w * scale), int(h * scale))
            )
            h, w = image.shape[:2]
        
        detections = []
        
        # Sliding window
        for y in range(0, h - self.window_size + 1, self.stride):
            for x in range(0, w - self.window_size + 1, self.stride):
                # Extract window
                window = image[y:y+self.window_size, x:x+self.window_size]
                
                # Convert BGR to RGB and transform
                window_rgb = cv2.cvtColor(window, cv2.COLOR_BGR2RGB)
                window_rgb = Image.fromarray(window_rgb)
                window_tensor = self.transforms(window_rgb).unsqueeze(0)
                
                # Inference
                with torch.no_grad():
                    window_tensor = window_tensor.to(self.device)
                    outputs = self.model(window_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    face_confidence = probabilities[0, 1].item()  # Confidence for face class
                
                # Keep if above threshold
                if face_confidence > self.confidence_threshold:
                    # Scale back if image was scaled
                    if scale != 1.0:
                        x_orig = int(x / scale)
                        y_orig = int(y / scale)
                        size_orig = int(self.window_size / scale)
                    else:
                        x_orig, y_orig = x, y
                        size_orig = self.window_size
                    
                    detections.append({
                        'box': (x_orig, y_orig, x_orig + size_orig, y_orig + size_orig),
                        'confidence': face_confidence
                    })
        
        # Post-process: remove duplicates and overlaps
        detections = self._nms(detections, iou_threshold=0.3)
        
        return detections
    
    @staticmethod
    def _nms(detections: List[dict], iou_threshold: float = 0.3) -> List[dict]:
        """
        Non-Maximum Suppression to remove overlapping detections.
        
        Args:
            detections: List of detection dicts
            iou_threshold: IoU threshold for suppression
        
        Returns:
            Filtered list of detections
        """
        if not detections:
            return detections
        
        # Sort by confidence (descending)
        detections = sorted(detections, key=lambda d: d['confidence'], reverse=True)
        
        keep = []
        while detections:
            keep.append(detections[0])
            if len(detections) == 1:
                break
            
            current_box = detections[0]['box']
            detections = detections[1:]
            
            # Filter overlapping boxes
            filtered = []
            for detection in detections:
                iou = ScratchCNNDetector._iou(current_box, detection['box'])
                if iou < iou_threshold:
                    filtered.append(detection)
            
            detections = filtered
        
        return keep
    
    @staticmethod
    def _iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        """
        Calculate Intersection over Union (IoU) between two boxes.
        
        Args:
            box1, box2: (x1, y1, x2, y2) format
        
        Returns:
            IoU value (0-1)
        """
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])
        
        if x2_inter < x1_inter or y2_inter < y1_inter:
            return 0.0
        
        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0


# Import cv2 for image operations
try:
    import cv2
except ImportError:
    cv2 = None
