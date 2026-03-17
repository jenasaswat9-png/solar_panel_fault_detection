"""
Data Loader Module for Solar Panel Fault Detection.

This module provides PyTorch Dataset and DataLoader implementations
for loading and batching solar panel images.
"""

import os
import sys
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Callable
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.preprocessing import LabelEncoder

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils import load_config, get_device


class SolarPanelDataset(Dataset):
    """
    PyTorch Dataset for Solar Panel Fault Detection.
    
    Attributes:
        data_dir: Directory containing the dataset
        transform: Optional transform to apply to images
        target_size: Target image size (width, height)
        class_to_idx: Dictionary mapping class names to indices
    """
    
    def __init__(self,
                 data_dir: str,
                 transform: Optional[Callable] = None,
                 target_size: Tuple[int, int] = (640, 640),
                 config: Dict = None):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Directory containing class subdirectories
            transform: Optional transform to apply
            target_size: Target image size
            config: Configuration dictionary
        """
        self.data_dir = data_dir
        self.transform = transform
        self.target_size = target_size
        self.config = config or load_config()
        
        # Get class names and create mapping
        self.classes = sorted([d for d in os.listdir(data_dir) 
                              if os.path.isdir(os.path.join(data_dir, d))])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.idx_to_class = {idx: cls_name for cls_name, idx in self.class_to_idx.items()}
        
        # Load all image paths and labels
        self.image_paths = []
        self.labels = []
        
        for class_name in self.classes:
            class_dir = os.path.join(data_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                    img_path = os.path.join(class_dir, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(class_idx)
        
        print(f"Loaded {len(self.image_paths)} images from {data_dir}")
        print(f"Classes: {self.classes}")
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (image_tensor, label)
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Resize if needed
        if image.size != self.target_size:
            image = image.resize(self.target_size, Image.BILINEAR)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_distribution(self) -> Dict[str, int]:
        """
        Get the distribution of classes in the dataset.
        
        Returns:
            Dictionary mapping class names to counts
        """
        distribution = {}
        for label in self.labels:
            class_name = self.idx_to_class[label]
            distribution[class_name] = distribution.get(class_name, 0) + 1
        return distribution
    
    def get_sample_weights(self) -> np.ndarray:
        """
        Calculate sample weights for balanced sampling.
        
        Returns:
            Array of sample weights
        """
        class_counts = self.get_class_distribution()
        total_samples = len(self.labels)
        num_classes = len(self.classes)
        
        # Calculate weight for each sample
        weights = []
        for label in self.labels:
            class_name = self.idx_to_class[label]
            class_count = class_counts[class_name]
            # Weight inversely proportional to class frequency
            weight = total_samples / (num_classes * class_count)
            weights.append(weight)
        
        return np.array(weights)


def get_train_transforms(config: Dict = None) -> transforms.Compose:
    """
    Get training data augmentation transforms.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Composed transforms
    """
    if config is None:
        config = load_config()
    
    aug_config = config.get('augmentation', {})
    
    transform_list = [
        transforms.Resize((config['image']['height'], config['image']['width'])),
    ]
    
    # Add augmentations
    if aug_config.get('rotation_range', 0) > 0:
        transform_list.append(
            transforms.RandomRotation(aug_config['rotation_range'])
        )
    
    if aug_config.get('flip_horizontal', False):
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
    
    if aug_config.get('flip_vertical', False):
        transform_list.append(transforms.RandomVerticalFlip(p=0.5))
    
    brightness_range = aug_config.get('brightness_range', [1.0, 1.0])
    if brightness_range != [1.0, 1.0]:
        brightness = (brightness_range[0] + brightness_range[1]) / 2
        transform_list.append(
            transforms.ColorJitter(brightness=brightness - 1.0)
        )
    
    # Convert to tensor and normalize
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return transforms.Compose(transform_list)


def get_val_transforms(config: Dict = None) -> transforms.Compose:
    """
    Get validation/test transforms (no augmentation).
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Composed transforms
    """
    if config is None:
        config = load_config()
    
    return transforms.Compose([
        transforms.Resize((config['image']['height'], config['image']['width'])),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def create_data_loaders(train_dir: str,
                       val_dir: str,
                       test_dir: str,
                       batch_size: int = 16,
                       num_workers: int = 4,
                       use_weighted_sampler: bool = False,
                       config: Dict = None) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Create data loaders for train, validation, and test sets.
    
    Args:
        train_dir: Training data directory
        val_dir: Validation data directory
        test_dir: Test data directory
        batch_size: Batch size
        num_workers: Number of worker processes
        use_weighted_sampler: Whether to use weighted sampling for imbalance
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, class_mapping)
    """
    if config is None:
        config = load_config()
    
    # Create datasets
    train_dataset = SolarPanelDataset(
        train_dir,
        transform=get_train_transforms(config),
        config=config
    )
    
    val_dataset = SolarPanelDataset(
        val_dir,
        transform=get_val_transforms(config),
        config=config
    )
    
    test_dataset = SolarPanelDataset(
        test_dir,
        transform=get_val_transforms(config),
        config=config
    )
    
    # Create samplers
    train_sampler = None
    if use_weighted_sampler:
        from torch.utils.data import WeightedRandomSampler
        
        sample_weights = train_dataset.get_sample_weights()
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Get class mapping
    class_mapping = train_dataset.class_to_idx
    
    print(f"\nData loaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader, class_mapping


def get_yolo_data_yaml(data_dir: str, config: Dict = None) -> str:
    """
    Create YAML configuration file for YOLO training.
    
    Args:
        data_dir: Directory containing train/val/test splits
        config: Configuration dictionary
        
    Returns:
        Path to created YAML file
    """
    if config is None:
        config = load_config()
    
    yaml_path = os.path.join(data_dir, 'data.yaml')
    
    yaml_content = f"""
# YOLO Dataset Configuration
path: {data_dir}
train: train
val: val
test: test

# Classes
nc: {len(config['model']['classes'])}
names: {config['model']['classes']}

# Image size
imgsz: {config['image']['width']}
"""
    
    with open(yaml_path, 'w') as f:
        f.write(yaml_content.strip())
    
    print(f"YOLO data YAML created: {yaml_path}")
    return yaml_path


if __name__ == "__main__":
    # Test the data loader
    config = load_config()
    
    data_dir = config['dataset']['processed_dir']
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')
    
    if os.path.exists(train_dir):
        train_loader, val_loader, test_loader, class_mapping = create_data_loaders(
            train_dir=train_dir,
            val_dir=val_dir,
            test_dir=test_dir,
            batch_size=config['training']['batch_size'],
            num_workers=2,
            use_weighted_sampler=True,
            config=config
        )
        
        # Test loading a batch
        for images, labels in train_loader:
            print(f"\nBatch shape: {images.shape}")
            print(f"Labels shape: {labels.shape}")
            print(f"Label values: {labels}")
            break
    else:
        print(f"Train directory not found: {train_dir}")
        print("Please run download_dataset.py first.")
