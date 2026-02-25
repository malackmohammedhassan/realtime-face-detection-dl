#!/usr/bin/env python3
"""
Training script for scratch CNN face detector.

This script trains a from-scratch CNN model to classify face vs non-face regions.
The model is designed to be lightweight and fast, suitable for real-time inference.

Dataset structure expected:
    data/train/face/*.jpg       (face images)
    data/train/non_face/*.jpg   (non-face images)
    data/val/face/*.jpg         (validation faces)
    data/val/non_face/*.jpg     (validation non-faces)

Usage:
    python scripts/train_scratch_cnn.py

    Optional arguments:
    --epochs NUM            Number of training epochs (default: 50)
    --batch_size SIZE       Batch size (default: 32)
    --lr RATE              Learning rate (default: 0.001)
    --img_size SIZE        Image size for training (default: 64)
"""

import sys
from pathlib import Path
import argparse
import logging
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scratch_cnn import TinyCNN, FaceNonFaceDataset, ScratchCNNTrainer
from utils.logger import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train scratch CNN face detector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/train_scratch_cnn.py --epochs 100 --batch_size 64
  python scripts/train_scratch_cnn.py --lr 0.0005
        """
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="L2 regularization weight (default: 1e-4)"
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=64,
        help="Input image size (default: 64)"
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data"),
        help="Path to data directory (default: data/)"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("models"),
        help="Path to output directory (default: models/)"
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    args.data_dir = Path(args.data_dir).resolve()
    args.output_dir = Path(args.output_dir).resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Verify data directory exists
    if not args.data_dir.exists():
        logger.error(f"Data directory not found: {args.data_dir}")
        logger.error(f"Please create the following structure:")
        logger.error(f"  {args.data_dir}/train/face/")
        logger.error(f"  {args.data_dir}/train/non_face/")
        logger.error(f"  {args.data_dir}/val/face/")
        logger.error(f"  {args.data_dir}/val/non_face/")
        return 1
    
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Create datasets
    logger.info("Loading datasets...")
    train_dataset = FaceNonFaceDataset(
        args.data_dir,
        split="train",
        img_size=args.img_size,
        augmentation=True
    )
    
    val_dataset = FaceNonFaceDataset(
        args.data_dir,
        split="val",
        img_size=args.img_size,
        augmentation=False
    )
    
    # Check if data was loaded
    if len(train_dataset) == 0:
        logger.error("No training images found!")
        return 1
    
    if len(val_dataset) == 0:
        logger.error("No validation images found!")
        return 1
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Create model
    logger.info("Creating model...")
    model = TinyCNN(num_classes=2, dropout_rate=0.5)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = ScratchCNNTrainer(
        model,
        device,
        learning_rate=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Training loop
    logger.info("Starting training...")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.lr}")
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(args.epochs):
        # Train
        train_loss, train_acc = trainer.train_epoch(train_loader)
        
        # Validate
        val_loss, val_acc = trainer.validate(val_loader)
        
        # Store metrics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Logging
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(
                f"Epoch [{epoch+1:3d}/{args.epochs}] | "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}"
            )
        
        # Save best model
        if val_acc > trainer.best_accuracy:
            trainer.best_accuracy = val_acc
            trainer.patience_counter = 0
            
            model_path = args.output_dir / "scratch_cnn.pth"
            trainer.save_checkpoint(model_path, is_best=True)
            logger.info(f"✓ Best model saved (Acc: {val_acc:.4f})")
        else:
            trainer.patience_counter += 1
            
            # Early stopping
            if trainer.patience_counter >= trainer.patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Final summary
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Best validation accuracy: {trainer.best_accuracy:.4f}")
    logger.info(f"Final training accuracy: {history['train_acc'][-1]:.4f}")
    logger.info(f"Best model saved to: {args.output_dir / 'scratch_cnn.pth'}")
    
    # Save training history
    import json
    history_path = args.output_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    logger.info(f"Training history saved to: {history_path}")
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
