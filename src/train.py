"""
Model Training Script for Solar Panel Fault Detection.

This script trains computer vision models (YOLOv8 or ResNet50) for
classifying solar panel faults.

Usage:
    # Train YOLOv8 model
    python src/train.py --model yolo --epochs 100 --batch-size 16
    
    # Train ResNet50 model
    python src/train.py --model resnet50 --epochs 100 --batch-size 16
    
    # Resume training from checkpoint
    python src/train.py --resume models/checkpoints/best_model.pth
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils import (
    load_config, setup_logging, set_seed, get_device,
    save_checkpoint, create_directories
)
from src.data_loader import create_data_loaders, SolarPanelDataset, get_train_transforms, get_val_transforms


class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'max'):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for maximizing metric, 'min' for minimizing
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current metric score
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class ResNet50Classifier(nn.Module):
    """ResNet50-based classifier for solar panel fault detection."""
    
    def __init__(self, num_classes: int = 4, pretrained: bool = True, dropout: float = 0.5):
        """
        Initialize ResNet50 classifier.
        
        Args:
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            dropout: Dropout probability
        """
        super(ResNet50Classifier, self).__init__()
        
        from torchvision.models import resnet50, ResNet50_Weights
        
        # Load pretrained ResNet50
        if pretrained:
            weights = ResNet50_Weights.IMAGENET1K_V2
            self.backbone = resnet50(weights=weights)
        else:
            self.backbone = resnet50(weights=None)
        
        # Get feature dimension
        in_features = self.backbone.fc.in_features
        
        # Replace final FC layer with custom classifier
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.backbone(x)


class Trainer:
    """Trainer class for solar panel fault detection models."""
    
    def __init__(self, config: Dict, model_type: str = 'yolo'):
        """
        Initialize trainer.
        
        Args:
            config: Configuration dictionary
            model_type: Type of model ('yolo' or 'resnet50')
        """
        self.config = config
        self.model_type = model_type
        self.device = get_device()
        self.logger = setup_logging()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rate': []
        }
        
        self.best_metric = 0.0
        self.start_epoch = 0
        
        # Create directories
        create_directories(config)
        
        # Initialize model
        self._setup_model()
        
        # Initialize data loaders
        self._setup_data()
        
        # Initialize optimizer and scheduler
        self._setup_optimizer()
        
        # Initialize early stopping
        self.early_stopping = EarlyStopping(
            patience=config['training']['early_stopping_patience'],
            mode='max'
        )
    
    def _setup_model(self) -> None:
        """Setup the model."""
        num_classes = len(self.config['model']['classes'])
        
        if self.model_type == 'resnet50':
            self.model = ResNet50Classifier(
                num_classes=num_classes,
                pretrained=self.config['model']['pretrained']
            )
            self.model.to(self.device)
            self.criterion = nn.CrossEntropyLoss()
            
        elif self.model_type == 'yolo':
            from ultralytics import YOLO
            
            # Load YOLO model
            model_name = self.config['model']['name']
            self.model = YOLO(f'{model_name}.pt')
            
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        self.logger.info(f"Model initialized: {self.model_type}")
    
    def _setup_data(self) -> None:
        """Setup data loaders."""
        data_dir = self.config['dataset']['processed_dir']
        
        train_dir = os.path.join(data_dir, 'train')
        val_dir = os.path.join(data_dir, 'val')
        test_dir = os.path.join(data_dir, 'test')
        
        if not os.path.exists(train_dir):
            self.logger.error(f"Training data not found: {train_dir}")
            self.logger.error("Please run download_dataset.py first.")
            raise FileNotFoundError(f"Training data not found: {train_dir}")
        
        if self.model_type == 'resnet50':
            self.train_loader, self.val_loader, self.test_loader, self.class_mapping = create_data_loaders(
                train_dir=train_dir,
                val_dir=val_dir,
                test_dir=test_dir,
                batch_size=self.config['training']['batch_size'],
                num_workers=4,
                use_weighted_sampler=True,
                config=self.config
            )
        else:
            # For YOLO, we just need the data.yaml path
            from src.data_loader import get_yolo_data_yaml
            self.yolo_data_yaml = get_yolo_data_yaml(data_dir, self.config)
    
    def _setup_optimizer(self) -> None:
        """Setup optimizer and learning rate scheduler."""
        if self.model_type == 'resnet50':
            # Optimizer
            optimizer_name = self.config['training']['optimizer']
            lr = self.config['training']['learning_rate']
            weight_decay = self.config['training']['weight_decay']
            
            if optimizer_name == 'Adam':
                self.optimizer = optim.Adam(
                    self.model.parameters(),
                    lr=lr,
                    weight_decay=weight_decay
                )
            elif optimizer_name == 'SGD':
                self.optimizer = optim.SGD(
                    self.model.parameters(),
                    lr=lr,
                    momentum=0.9,
                    weight_decay=weight_decay
                )
            else:
                raise ValueError(f"Unknown optimizer: {optimizer_name}")
            
            # Learning rate scheduler
            scheduler_name = self.config['training']['scheduler']
            if scheduler_name == 'ReduceLROnPlateau':
                self.scheduler = ReduceLROnPlateau(
                    self.optimizer,
                    mode='max',
                    factor=0.5,
                    patience=self.config['training']['patience']
                )
            elif scheduler_name == 'CosineAnnealingLR':
                self.scheduler = CosineAnnealingLR(
                    self.optimizer,
                    T_max=self.config['training']['epochs']
                )
            else:
                self.scheduler = None
    
    def train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self) -> Tuple[float, float]:
        """
        Validate the model.
        
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc='Validation'):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Statistics
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train_resnet50(self) -> Dict:
        """
        Train ResNet50 model.
        
        Returns:
            Training history dictionary
        """
        self.logger.info("="*70)
        self.logger.info("STARTING RESNET50 TRAINING")
        self.logger.info("="*70)
        
        num_epochs = self.config['training']['epochs']
        checkpoint_dir = self.config['training']['checkpoint_dir']
        
        for epoch in range(self.start_epoch, num_epochs):
            self.logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
            self.logger.info("-" * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            
            # Validate
            val_loss, val_acc = self.validate()
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rate'].append(current_lr)
            
            # Log metrics
            self.logger.info(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            self.logger.info(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            self.logger.info(f"Learning Rate: {current_lr:.6f}")
            
            # Update scheduler
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_acc)
            elif self.scheduler is not None:
                self.scheduler.step()
            
            # Save best model
            if val_acc > self.best_metric:
                self.best_metric = val_acc
                checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
                save_checkpoint(
                    self.model, self.optimizer, epoch,
                    self.best_metric, checkpoint_path
                )
                self.logger.info(f"Best model saved! (Val Acc: {val_acc:.2f}%)")
            
            # Save checkpoint periodically
            if (epoch + 1) % 10 == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
                save_checkpoint(
                    self.model, self.optimizer, epoch,
                    self.best_metric, checkpoint_path
                )
            
            # Early stopping
            if self.early_stopping(val_acc):
                self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        self.logger.info("\n" + "="*70)
        self.logger.info("TRAINING COMPLETE!")
        self.logger.info(f"Best Validation Accuracy: {self.best_metric:.2f}%")
        self.logger.info("="*70)
        
        return self.history
    
    def train_yolo(self) -> Dict:
        """
        Train YOLO model.
        
        Returns:
            Training results dictionary
        """
        self.logger.info("="*70)
        self.logger.info("STARTING YOLO TRAINING")
        self.logger.info("="*70)
        
        # Train YOLO model
        results = self.model.train(
            data=self.yolo_data_yaml,
            epochs=self.config['training']['epochs'],
            imgsz=self.config['image']['width'],
            batch=self.config['training']['batch_size'],
            patience=self.config['training']['early_stopping_patience'],
            save=True,
            project='models',
            name='yolo_training',
            exist_ok=True,
            pretrained=True,
            optimizer='Adam',
            lr0=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay'],
            augment=True,
            mosaic=1.0,
            mixup=0.1,
            copy_paste=0.1,
            device=self.device
        )
        
        self.logger.info("\n" + "="*70)
        self.logger.info("YOLO TRAINING COMPLETE!")
        self.logger.info("="*70)
        
        return results
    
    def train(self) -> Dict:
        """
        Train the model.
        
        Returns:
            Training results
        """
        if self.model_type == 'resnet50':
            return self.train_resnet50()
        elif self.model_type == 'yolo':
            return self.train_yolo()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def resume_from_checkpoint(self, checkpoint_path: str) -> None:
        """
        Resume training from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        from src.utils import load_checkpoint
        
        checkpoint = load_checkpoint(
            self.model, checkpoint_path,
            self.optimizer, self.device
        )
        
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_metric = checkpoint.get('best_metric', 0.0)
        
        self.logger.info(f"Resuming from epoch {self.start_epoch}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Solar Panel Fault Detection Model')
    parser.add_argument('--model', type=str, default='yolo',
                       choices=['yolo', 'resnet50'],
                       help='Model type to train')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['learning_rate'] = args.lr
    
    # Set seed
    set_seed(args.seed)
    
    # Create trainer
    trainer = Trainer(config, model_type=args.model)
    
    # Resume from checkpoint if specified
    if args.resume and args.model == 'resnet50':
        trainer.resume_from_checkpoint(args.resume)
    
    # Train
    results = trainer.train()
    
    # Save training history
    history_path = 'results/training_history.json'
    os.makedirs('results', exist_ok=True)
    with open(history_path, 'w') as f:
        json.dump(trainer.history, f, indent=2)
    
    print(f"\nTraining history saved to {history_path}")


if __name__ == "__main__":
    main()
