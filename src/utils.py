"""
Utility functions for Solar Panel Fault Detection project.
"""

import os
import logging
import yaml
import random
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, Tuple, List
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing configuration
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(log_file: str = "logs/app.log", level: str = "INFO") -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_file: Path to log file
        level: Logging level
        
    Returns:
        Logger instance
    """
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Get the best available device (CUDA, MPS, or CPU).
    
    Returns:
        torch.device instance
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    return device


def create_directories(config: Dict[str, Any]) -> None:
    """
    Create necessary directories for the project.
    
    Args:
        config: Configuration dictionary
    """
    dirs = [
        config['dataset']['raw_dir'],
        config['dataset']['processed_dir'],
        config['training']['checkpoint_dir'],
        'logs',
        'results',
        'models'
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")


def plot_confusion_matrix(cm: np.ndarray, 
                         class_names: List[str],
                         save_path: str = None,
                         normalize: bool = True) -> None:
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save the plot
        normalize: Whether to normalize the matrix
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title(title, fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    plt.show()


def plot_training_history(history: Dict[str, List[float]], 
                         save_path: str = None) -> None:
    """
    Plot training and validation metrics.
    
    Args:
        history: Dictionary containing training history
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Loss', fontsize=14)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy
    axes[0, 1].plot(history['train_acc'], label='Train Acc')
    axes[0, 1].plot(history['val_acc'], label='Val Acc')
    axes[0, 1].set_title('Accuracy', fontsize=14)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Precision
    if 'precision' in history:
        axes[1, 0].plot(history['precision'], label='Precision')
        axes[1, 0].set_title('Precision', fontsize=14)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Recall
    if 'recall' in history:
        axes[1, 1].plot(history['recall'], label='Recall')
        axes[1, 1].set_title('Recall', fontsize=14)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    plt.show()


def draw_bounding_boxes(image: np.ndarray,
                       boxes: List[Tuple[int, int, int, int]],
                       labels: List[str],
                       confidences: List[float],
                       color_map: Dict[str, Tuple[int, int, int]] = None) -> np.ndarray:
    """
    Draw bounding boxes on image.
    
    Args:
        image: Input image (BGR format)
        boxes: List of bounding boxes (x1, y1, x2, y2)
        labels: List of labels
        confidences: List of confidence scores
        color_map: Dictionary mapping labels to colors
        
    Returns:
        Image with bounding boxes drawn
    """
    if color_map is None:
        color_map = {
            'normal': (0, 255, 0),
            'cracked': (0, 0, 255),
            'hotspot': (255, 0, 0),
            'dust': (255, 255, 0)
        }
    
    img_copy = image.copy()
    
    for box, label, conf in zip(boxes, labels, confidences):
        x1, y1, x2, y2 = box
        color = color_map.get(label, (255, 255, 255))
        
        # Draw rectangle
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        text = f"{label}: {conf:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        
        cv2.rectangle(img_copy, (x1, y1 - text_height - 10), 
                     (x1 + text_width, y1), color, -1)
        cv2.putText(img_copy, text, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return img_copy


def save_checkpoint(model: torch.nn.Module,
                   optimizer: torch.optim.Optimizer,
                   epoch: int,
                   best_metric: float,
                   checkpoint_path: str) -> None:
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        best_metric: Best metric achieved
        checkpoint_path: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_metric': best_metric,
        'timestamp': datetime.now().isoformat()
    }
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


def load_checkpoint(model: torch.nn.Module,
                   checkpoint_path: str,
                   optimizer: torch.optim.Optimizer = None,
                   device: torch.device = None) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint file
        optimizer: Optional optimizer to load state
        device: Device to load checkpoint on
        
    Returns:
        Dictionary containing checkpoint info
    """
    if device is None:
        device = get_device()
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Checkpoint loaded from {checkpoint_path}")
    print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"Best metric: {checkpoint.get('best_metric', 'N/A')}")
    
    return checkpoint


def calculate_metrics(y_true: np.ndarray, 
                     y_pred: np.ndarray,
                     average: str = 'weighted') -> Dict[str, float]:
    """
    Calculate classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging method for multi-class
        
    Returns:
        Dictionary containing metrics
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average=average, zero_division=0)
    }
    
    return metrics


def get_class_distribution(dataset_path: str) -> Dict[str, int]:
    """
    Get class distribution from dataset directory.
    
    Args:
        dataset_path: Path to dataset directory
        
    Returns:
        Dictionary with class names and counts
    """
    class_dist = {}
    
    for class_name in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_path):
            num_images = len([f for f in os.listdir(class_path) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            class_dist[class_name] = num_images
    
    return class_dist


def print_class_distribution(class_dist: Dict[str, int]) -> None:
    """
    Print class distribution in a formatted way.
    
    Args:
        class_dist: Dictionary with class names and counts
    """
    total = sum(class_dist.values())
    
    print("\n" + "="*50)
    print("CLASS DISTRIBUTION")
    print("="*50)
    
    for class_name, count in sorted(class_dist.items()):
        percentage = (count / total) * 100
        print(f"{class_name:15s}: {count:4d} images ({percentage:5.2f}%)")
    
    print("-"*50)
    print(f"{'TOTAL':15s}: {total:4d} images")
    print("="*50 + "\n")


def resize_image(image: np.ndarray, 
                target_size: Tuple[int, int] = (640, 640),
                keep_aspect_ratio: bool = True) -> np.ndarray:
    """
    Resize image to target size.
    
    Args:
        image: Input image
        target_size: Target (width, height)
        keep_aspect_ratio: Whether to maintain aspect ratio
        
    Returns:
        Resized image
    """
    if keep_aspect_ratio:
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize
        resized = cv2.resize(image, (new_w, new_h))
        
        # Create canvas
        result = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        
        # Center the image
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return result
    else:
        return cv2.resize(image, target_size)


def normalize_image(image: np.ndarray,
                   mean: List[float] = [0.485, 0.456, 0.406],
                   std: List[float] = [0.229, 0.224, 0.225]) -> np.ndarray:
    """
    Normalize image with mean and std.
    
    Args:
        image: Input image (0-255 range)
        mean: Mean values for each channel
        std: Std values for each channel
        
    Returns:
        Normalized image
    """
    image = image.astype(np.float32) / 255.0
    
    mean = np.array(mean).reshape(1, 1, 3)
    std = np.array(std).reshape(1, 1, 3)
    
    normalized = (image - mean) / std
    
    return normalized
