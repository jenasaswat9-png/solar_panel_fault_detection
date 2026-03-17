"""
Dataset Download Script for Solar Panel Fault Detection.

This script downloads the solar panel fault detection dataset from Kaggle,
extracts it, and creates train/validation/test splits.

Usage:
    python src/download_dataset.py

Prerequisites:
    1. Install kaggle: pip install kaggle
    2. Set up Kaggle API credentials:
       - Download kaggle.json from your Kaggle account
       - Place it in ~/.kaggle/kaggle.json
       - Or set environment variables: KAGGLE_USERNAME and KAGGLE_KEY
"""

import os
import sys
import shutil
import zipfile
from pathlib import Path
from typing import Tuple
import random
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.utils import load_config, setup_logging, create_directories, set_seed


def check_kaggle_credentials() -> bool:
    """
    Check if Kaggle API credentials are configured.
    
    Returns:
        True if credentials are available, False otherwise
    """
    # Check for kaggle.json file
    kaggle_json_path = Path.home() / '.kaggle' / 'kaggle.json'
    
    # Check for environment variables
    env_vars_set = os.environ.get('KAGGLE_USERNAME') and os.environ.get('KAGGLE_KEY')
    
    if kaggle_json_path.exists() or env_vars_set:
        return True
    
    print("\n" + "="*70)
    print("KAGGLE API CREDENTIALS NOT FOUND!")
    print("="*70)
    print("\nTo use the Kaggle API, you need to set up credentials:")
    print("\nMethod 1 - Using kaggle.json file:")
    print("  1. Go to https://www.kaggle.com/account")
    print("  2. Click 'Create New API Token'")
    print("  3. Move the downloaded kaggle.json to ~/.kaggle/kaggle.json")
    print("  4. Run: chmod 600 ~/.kaggle/kaggle.json")
    print("\nMethod 2 - Using environment variables:")
    print("  export KAGGLE_USERNAME=your_username")
    print("  export KAGGLE_KEY=your_api_key")
    print("="*70 + "\n")
    
    return False


def download_kaggle_dataset(dataset_name: str, 
                           output_dir: str,
                           logger) -> str:
    """
    Download dataset from Kaggle.
    
    Args:
        dataset_name: Kaggle dataset identifier (e.g., 'username/dataset-name')
        output_dir: Directory to save the downloaded dataset
        logger: Logger instance
        
    Returns:
        Path to downloaded zip file
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        logger.error("kaggle package not installed. Run: pip install kaggle")
        raise
    
    # Authenticate
    api = KaggleApi()
    api.authenticate()
    
    logger.info(f"Downloading dataset: {dataset_name}")
    logger.info(f"Output directory: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Download dataset
    api.dataset_download_files(
        dataset_name,
        path=output_dir,
        unzip=False,
        quiet=False
    )
    
    # Find the downloaded zip file
    zip_files = list(Path(output_dir).glob('*.zip'))
    
    if not zip_files:
        raise FileNotFoundError("No zip file found after download")
    
    zip_path = str(zip_files[0])
    logger.info(f"Dataset downloaded to: {zip_path}")
    
    return zip_path


def extract_dataset(zip_path: str, 
                   extract_dir: str,
                   logger) -> str:
    """
    Extract downloaded dataset zip file.
    
    Args:
        zip_path: Path to zip file
        extract_dir: Directory to extract to
        logger: Logger instance
        
    Returns:
        Path to extracted dataset directory
    """
    logger.info(f"Extracting dataset from: {zip_path}")
    
    os.makedirs(extract_dir, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    
    # Remove zip file after extraction
    os.remove(zip_path)
    logger.info(f"Dataset extracted to: {extract_dir}")
    
    return extract_dir


def organize_images_by_class(raw_dir: str, 
                             processed_dir: str,
                             logger) -> None:
    """
    Organize images into class directories.
    
    Args:
        raw_dir: Directory containing raw images
        processed_dir: Directory to organize images into
        logger: Logger instance
    """
    logger.info("Organizing images by class...")
    
    # Create class directories
    classes = ['normal', 'cracked', 'hotspot', 'dust']
    for class_name in classes:
        os.makedirs(os.path.join(processed_dir, class_name), exist_ok=True)
    
    # Scan raw directory and organize images
    # This assumes the dataset has some structure we need to parse
    # Adjust based on actual dataset structure
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    for root, dirs, files in os.walk(raw_dir):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                src_path = os.path.join(root, file)
                
                # Determine class from directory name or filename
                # This is a heuristic - adjust based on actual dataset
                class_name = None
                lower_path = src_path.lower()
                
                if 'normal' in lower_path or 'clean' in lower_path:
                    class_name = 'normal'
                elif 'crack' in lower_path or 'broken' in lower_path:
                    class_name = 'cracked'
                elif 'hot' in lower_path or 'spot' in lower_path:
                    class_name = 'hotspot'
                elif 'dust' in lower_path or 'dirty' in lower_path or 'soil' in lower_path:
                    class_name = 'dust'
                else:
                    # Default to normal if can't determine
                    class_name = 'normal'
                
                dst_path = os.path.join(processed_dir, class_name, file)
                shutil.copy2(src_path, dst_path)
    
    logger.info("Images organized by class")


def split_dataset(processed_dir: str,
                 output_dir: str,
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.15,
                 seed: int = 42,
                 logger = None) -> Tuple[str, str, str]:
    """
    Split dataset into train/validation/test sets.
    
    Args:
        processed_dir: Directory containing organized images
        output_dir: Directory to save splits
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        test_ratio: Ratio of test data
        seed: Random seed
        logger: Logger instance
        
    Returns:
        Tuple of (train_dir, val_dir, test_dir)
    """
    if logger:
        logger.info("Splitting dataset into train/val/test...")
    
    set_seed(seed)
    
    # Create split directories
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')
    
    for split_dir in [train_dir, val_dir, test_dir]:
        os.makedirs(split_dir, exist_ok=True)
    
    # Process each class
    for class_name in os.listdir(processed_dir):
        class_path = os.path.join(processed_dir, class_name)
        
        if not os.path.isdir(class_path):
            continue
        
        # Get all images in class
        images = [f for f in os.listdir(class_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
        images.sort()
        
        # Shuffle
        random.shuffle(images)
        
        # Calculate split sizes
        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        # Rest goes to test
        
        # Split
        train_images = images[:n_train]
        val_images = images[n_train:n_train + n_val]
        test_images = images[n_train + n_val:]
        
        if logger:
            logger.info(f"Class '{class_name}': {n_total} images "
                       f"(train: {len(train_images)}, val: {len(val_images)}, test: {len(test_images)})")
        
        # Copy images to split directories
        for split_name, split_images, split_dir in [
            ('train', train_images, train_dir),
            ('val', val_images, val_dir),
            ('test', test_images, test_dir)
        ]:
            split_class_dir = os.path.join(split_dir, class_name)
            os.makedirs(split_class_dir, exist_ok=True)
            
            for img_name in split_images:
                src = os.path.join(class_path, img_name)
                dst = os.path.join(split_class_dir, img_name)
                shutil.copy2(src, dst)
    
    if logger:
        logger.info(f"Dataset split complete:")
        logger.info(f"  Train: {train_dir}")
        logger.info(f"  Val: {val_dir}")
        logger.info(f"  Test: {test_dir}")
    
    return train_dir, val_dir, test_dir


def create_synthetic_dataset(output_dir: str, 
                            num_samples_per_class: int = 100,
                            logger = None) -> None:
    """
    Create a synthetic dataset for demonstration purposes.
    Used when actual dataset is not available.
    
    Args:
        output_dir: Directory to save synthetic dataset
        num_samples_per_class: Number of samples per class
        logger: Logger instance
    """
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    
    if logger:
        logger.info("Creating synthetic dataset for demonstration...")
    
    classes = ['normal', 'cracked', 'hotspot', 'dust']
    
    for class_name in classes:
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        for i in range(num_samples_per_class):
            # Create base image (simulating solar panel)
            img = Image.new('RGB', (640, 480), color=(50, 50, 50))
            draw = ImageDraw.Draw(img)
            
            # Draw grid pattern (solar cells)
            for row in range(6):
                for col in range(8):
                    x1 = col * 80
                    y1 = row * 80
                    x2 = x1 + 75
                    y2 = y1 + 75
                    
                    # Base cell color (blue-ish for solar panels)
                    base_color = (30, 60, 100)
                    
                    if class_name == 'normal':
                        # Normal panel - uniform color
                        color = base_color
                    elif class_name == 'cracked':
                        # Cracked panel - add crack lines
                        color = base_color
                        if random.random() > 0.7:
                            color = (20, 40, 80)
                    elif class_name == 'hotspot':
                        # Hotspot - add bright spots
                        if random.random() > 0.8:
                            color = (150, 50, 50)
                        else:
                            color = base_color
                    else:  # dust
                        # Dusty panel - lighter, dusty appearance
                        dust_factor = random.uniform(0.3, 0.7)
                        color = (
                            int(base_color[0] + dust_factor * 100),
                            int(base_color[1] + dust_factor * 80),
                            int(base_color[2] + dust_factor * 60)
                        )
                    
                    draw.rectangle([x1, y1, x2, y2], fill=color)
                    
                    # Add cracks for cracked class
                    if class_name == 'cracked' and random.random() > 0.5:
                        crack_x = random.randint(x1, x2)
                        draw.line([(crack_x, y1), (crack_x, y2)], fill=(10, 10, 10), width=1)
            
            # Save image
            img_path = os.path.join(class_dir, f"{class_name}_{i:04d}.jpg")
            img.save(img_path)
    
    if logger:
        logger.info(f"Synthetic dataset created with {num_samples_per_class} samples per class")


def main():
    """Main function to download and prepare dataset."""
    # Setup logging
    logger = setup_logging()
    
    logger.info("="*70)
    logger.info("SOLAR PANEL FAULT DETECTION - DATASET PREPARATION")
    logger.info("="*70)
    
    # Load configuration
    config = load_config()
    
    # Create directories
    create_directories(config)
    
    raw_dir = config['dataset']['raw_dir']
    processed_dir = config['dataset']['processed_dir']
    
    # Check if we should download from Kaggle or create synthetic dataset
    use_kaggle = check_kaggle_credentials()
    
    if use_kaggle:
        try:
            # Download from Kaggle
            dataset_name = config['dataset']['kaggle_dataset']
            zip_path = download_kaggle_dataset(dataset_name, raw_dir, logger)
            
            # Extract dataset
            extract_dir = extract_dataset(zip_path, raw_dir, logger)
            
            # Organize images
            organize_images_by_class(raw_dir, processed_dir, logger)
            
        except Exception as e:
            logger.error(f"Error downloading from Kaggle: {e}")
            logger.info("Falling back to synthetic dataset...")
            use_kaggle = False
    
    if not use_kaggle:
        # Create synthetic dataset
        create_synthetic_dataset(processed_dir, num_samples_per_class=100, logger=logger)
    
    # Split dataset
    train_dir, val_dir, test_dir = split_dataset(
        processed_dir=processed_dir,
        output_dir=processed_dir,
        train_ratio=config['dataset']['train_split'],
        val_ratio=config['dataset']['val_split'],
        test_ratio=config['dataset']['test_split'],
        seed=config['dataset']['random_seed'],
        logger=logger
    )
    
    # Print summary
    logger.info("\n" + "="*70)
    logger.info("DATASET PREPARATION COMPLETE!")
    logger.info("="*70)
    
    for split_name, split_dir in [('Train', train_dir), ('Val', val_dir), ('Test', test_dir)]:
        total_images = sum(len(files) for _, _, files in os.walk(split_dir))
        logger.info(f"{split_name}: {total_images} images")
    
    logger.info("="*70 + "\n")


if __name__ == "__main__":
    main()
