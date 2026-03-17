"""
Data Preprocessing Module for Solar Panel Fault Detection.

This module handles image preprocessing including resizing, normalization,
and data augmentation for training deep learning models.
"""

import os
import sys
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Union
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils import load_config, resize_image, normalize_image


class ImagePreprocessor:
    """
    Image preprocessor for solar panel images.
    
    Handles resizing, normalization, and augmentation.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize preprocessor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or load_config()
        self.target_size = (
            self.config['image']['width'],
            self.config['image']['height']
        )
        
        # Setup augmentation pipeline
        self.train_transform = self._get_train_transform()
        self.val_transform = self._get_val_transform()
    
    def _get_train_transform(self) -> A.Compose:
        """
        Get training augmentation pipeline using Albumentations.
        
        Returns:
            Composed augmentation pipeline
        """
        aug_config = self.config.get('augmentation', {})
        
        transforms = [
            A.Resize(height=self.target_size[1], width=self.target_size[0]),
        ]
        
        # Add augmentations based on config
        if aug_config.get('rotation_range', 0) > 0:
            transforms.append(
                A.Rotate(limit=aug_config['rotation_range'], p=0.5)
            )
        
        if aug_config.get('flip_horizontal', False):
            transforms.append(A.HorizontalFlip(p=0.5))
        
        if aug_config.get('flip_vertical', False):
            transforms.append(A.VerticalFlip(p=0.5))
        
        brightness_range = aug_config.get('brightness_range', [1.0, 1.0])
        if brightness_range != [1.0, 1.0]:
            brightness_limit = (brightness_range[1] - brightness_range[0]) / 2
            transforms.append(
                A.RandomBrightnessContrast(
                    brightness_limit=brightness_limit,
                    contrast_limit=0.1,
                    p=0.5
                )
            )
        
        # Add more augmentations
        transforms.extend([
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.Blur(blur_limit=3, p=0.2),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
        
        return A.Compose(transforms)
    
    def _get_val_transform(self) -> A.Compose:
        """
        Get validation/test transform pipeline (no augmentation).
        
        Returns:
            Composed transform pipeline
        """
        return A.Compose([
            A.Resize(height=self.target_size[1], width=self.target_size[0]),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    
    def preprocess_for_training(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for training (with augmentation).
        
        Args:
            image: Input image (BGR or RGB)
            
        Returns:
            Preprocessed tensor
        """
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        transformed = self.train_transform(image=image)
        
        return transformed['image']
    
    def preprocess_for_inference(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for inference (no augmentation).
        
        Args:
            image: Input image (BGR or RGB)
            
        Returns:
            Preprocessed tensor
        """
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        transformed = self.val_transform(image=image)
        
        return transformed['image']
    
    def preprocess_image_file(self, 
                             image_path: str,
                             for_training: bool = False) -> torch.Tensor:
        """
        Load and preprocess an image file.
        
        Args:
            image_path: Path to image file
            for_training: Whether to apply training augmentations
            
        Returns:
            Preprocessed tensor
        """
        # Load image
        image = cv2.imread(image_path)
        
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Preprocess
        if for_training:
            return self.preprocess_for_training(image)
        else:
            return self.preprocess_for_inference(image)


def apply_clahe(image: np.ndarray, 
                clip_limit: float = 2.0,
                tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    Useful for enhancing solar panel images.
    
    Args:
        image: Input image (BGR)
        clip_limit: Threshold for contrast limiting
        tile_grid_size: Size of grid for histogram equalization
        
    Returns:
        Enhanced image
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Split channels
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l = clahe.apply(l)
    
    # Merge channels
    enhanced = cv2.merge([l, a, b])
    
    # Convert back to BGR
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    return enhanced


def detect_panel_region(image: np.ndarray,
                       lower_bound: np.ndarray = None,
                       upper_bound: np.ndarray = None) -> np.ndarray:
    """
    Detect solar panel region in image using color segmentation.
    
    Args:
        image: Input image (BGR)
        lower_bound: Lower HSV bound for panel color
        upper_bound: Upper HSV bound for panel color
        
    Returns:
        Mask of detected panel region
    """
    if lower_bound is None:
        lower_bound = np.array([0, 0, 0])
    if upper_bound is None:
        upper_bound = np.array([180, 255, 100])
    
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Create mask
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    # Apply morphological operations
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    return mask


def enhance_contrast(image: np.ndarray, 
                    method: str = 'clahe') -> np.ndarray:
    """
    Enhance image contrast.
    
    Args:
        image: Input image (BGR)
        method: Enhancement method ('clahe', 'histogram', 'adaptive')
        
    Returns:
        Enhanced image
    """
    if method == 'clahe':
        return apply_clahe(image)
    
    elif method == 'histogram':
        # Global histogram equalization
        if len(image.shape) == 3:
            # Convert to YUV and equalize Y channel
            yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
            return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        else:
            return cv2.equalizeHist(image)
    
    elif method == 'adaptive':
        # Adaptive histogram equalization
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply adaptive histogram equalization
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    else:
        raise ValueError(f"Unknown enhancement method: {method}")


def remove_noise(image: np.ndarray,
                method: str = 'gaussian') -> np.ndarray:
    """
    Remove noise from image.
    
    Args:
        image: Input image
        method: Denoising method ('gaussian', 'median', 'bilateral')
        
    Returns:
        Denoised image
    """
    if method == 'gaussian':
        return cv2.GaussianBlur(image, (5, 5), 0)
    
    elif method == 'median':
        return cv2.medianBlur(image, 5)
    
    elif method == 'bilateral':
        return cv2.bilateralFilter(image, 9, 75, 75)
    
    else:
        raise ValueError(f"Unknown denoising method: {method}")


def preprocess_pipeline(image_path: str,
                       config: Dict = None,
                       enhance: bool = True,
                       denoise: bool = True) -> np.ndarray:
    """
    Complete preprocessing pipeline for a single image.
    
    Args:
        image_path: Path to image file
        config: Configuration dictionary
        enhance: Whether to apply contrast enhancement
        denoise: Whether to apply denoising
        
    Returns:
        Preprocessed image array
    """
    # Load image
    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    if config is None:
        config = load_config()
    
    target_size = (config['image']['width'], config['image']['height'])
    
    # Resize
    image = resize_image(image, target_size, keep_aspect_ratio=True)
    
    # Enhance contrast
    if enhance:
        image = enhance_contrast(image, method='clahe')
    
    # Denoise
    if denoise:
        image = remove_noise(image, method='bilateral')
    
    return image


def batch_preprocess(input_dir: str,
                    output_dir: str,
                    config: Dict = None,
                    enhance: bool = True,
                    denoise: bool = True) -> None:
    """
    Preprocess all images in a directory.
    
    Args:
        input_dir: Input directory containing images
        output_dir: Output directory for preprocessed images
        config: Configuration dictionary
        enhance: Whether to apply contrast enhancement
        denoise: Whether to apply denoising
    """
    from tqdm import tqdm
    
    if config is None:
        config = load_config()
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_files.append(os.path.join(root, file))
    
    print(f"Preprocessing {len(image_files)} images...")
    
    for img_path in tqdm(image_files):
        try:
            # Preprocess
            processed = preprocess_pipeline(
                img_path, config, enhance, denoise
            )
            
            # Save
            rel_path = os.path.relpath(img_path, input_dir)
            output_path = os.path.join(output_dir, rel_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            cv2.imwrite(output_path, processed)
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    print(f"Preprocessing complete. Images saved to {output_dir}")


if __name__ == "__main__":

    import shutil
    import random

    config = load_config()

    raw_dir = "data/raw"
    processed_dir = "data/processed"

    train_dir = os.path.join(processed_dir, "train")
    val_dir = os.path.join(processed_dir, "val")
    test_dir = os.path.join(processed_dir, "test")

    classes = ["cracked", "dust", "hotspot", "normal"]

    # create directories
    for split in ["train", "val", "test"]:
        for cls in classes:
            os.makedirs(os.path.join(processed_dir, split, cls), exist_ok=True)

    print("Splitting dataset...")

    for cls in classes:

        cls_path = os.path.join(raw_dir, cls)

        images = [f for f in os.listdir(cls_path)
                  if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        random.shuffle(images)

        total = len(images)

        train_count = int(0.7 * total)
        val_count = int(0.15 * total)

        train_imgs = images[:train_count]
        val_imgs = images[train_count:train_count + val_count]
        test_imgs = images[train_count + val_count:]

        for img in train_imgs:
            shutil.copy(
                os.path.join(cls_path, img),
                os.path.join(train_dir, cls, img)
            )

        for img in val_imgs:
            shutil.copy(
                os.path.join(cls_path, img),
                os.path.join(val_dir, cls, img)
            )

        for img in test_imgs:
            shutil.copy(
                os.path.join(cls_path, img),
                os.path.join(test_dir, cls, img)
            )

        print(f"{cls}: {total} images → train:{len(train_imgs)} val:{len(val_imgs)} test:{len(test_imgs)}")

    print("\nDataset split complete!")
    print("Processed dataset created at: data/processed")