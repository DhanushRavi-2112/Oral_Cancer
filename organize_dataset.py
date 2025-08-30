#!/usr/bin/env python3
"""
Script to organize the oral cancer dataset into train/val/test splits.
Moves images from dataset/ directory to data/ directory with proper structure.
"""

import os
import shutil
import random
import argparse
from pathlib import Path
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_image_files(directory: str) -> List[str]:
    """Get all image files from a directory."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.jfif', '.webp'}
    image_files = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(root, file))
    
    return image_files


def create_directory_structure(base_dir: str):
    """Create the required directory structure for train/val/test splits."""
    splits = ['train', 'val', 'test']
    classes = ['healthy', 'cancer']
    
    for split in splits:
        for class_name in classes:
            dir_path = os.path.join(base_dir, split, class_name)
            os.makedirs(dir_path, exist_ok=True)
    
    logger.info(f"Created directory structure in {base_dir}")


def split_data(files: List[str], train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15) -> Dict[str, List[str]]:
    """Split files into train, validation, and test sets."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    # Shuffle files randomly
    files = files.copy()
    random.shuffle(files)
    
    total_files = len(files)
    train_end = int(total_files * train_ratio)
    val_end = train_end + int(total_files * val_ratio)
    
    splits = {
        'train': files[:train_end],
        'val': files[train_end:val_end],
        'test': files[val_end:]
    }
    
    logger.info(f"Data split: Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}")
    
    return splits


def copy_files(file_splits: Dict[str, List[str]], class_name: str, destination_base: str):
    """Copy files to their respective directories."""
    for split_name, files in file_splits.items():
        destination_dir = os.path.join(destination_base, split_name, class_name)
        
        for i, src_file in enumerate(files):
            # Create a simple filename to avoid conflicts
            file_extension = os.path.splitext(src_file)[1]
            dst_filename = f"{class_name}_{split_name}_{i+1:04d}{file_extension}"
            dst_path = os.path.join(destination_dir, dst_filename)
            
            try:
                shutil.copy2(src_file, dst_path)
            except Exception as e:
                logger.warning(f"Failed to copy {src_file}: {e}")
                continue
        
        logger.info(f"Copied {len(files)} {class_name} images to {split_name} set")


def verify_organization(data_dir: str):
    """Verify the data organization and print statistics."""
    splits = ['train', 'val', 'test']
    classes = ['healthy', 'cancer']
    
    logger.info("\nData organization verification:")
    logger.info("=" * 50)
    
    total_images = 0
    class_totals = {'healthy': 0, 'cancer': 0}
    
    for split in splits:
        split_total = 0
        logger.info(f"\n{split.upper()} SET:")
        
        for class_name in classes:
            class_dir = os.path.join(data_dir, split, class_name)
            
            if os.path.exists(class_dir):
                count = len([f for f in os.listdir(class_dir) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.jfif', '.webp'))])
                logger.info(f"  {class_name}: {count} images")
                split_total += count
                class_totals[class_name] += count
            else:
                logger.info(f"  {class_name}: 0 images (directory doesn't exist)")
        
        logger.info(f"  Total: {split_total} images")
        total_images += split_total
    
    logger.info(f"\nOVERALL STATISTICS:")
    logger.info(f"  Total images: {total_images}")
    logger.info(f"  Healthy images: {class_totals['healthy']} ({class_totals['healthy']/total_images*100:.1f}%)")
    logger.info(f"  Cancer images: {class_totals['cancer']} ({class_totals['cancer']/total_images*100:.1f}%)")
    
    # Check if data is reasonably balanced
    balance_ratio = class_totals['cancer'] / class_totals['healthy'] if class_totals['healthy'] > 0 else 0
    logger.info(f"  Class balance ratio (cancer/healthy): {balance_ratio:.2f}")
    
    if 0.5 <= balance_ratio <= 2.0:
        logger.info("  ✓ Dataset is reasonably balanced")
    else:
        logger.warning("  ⚠ Dataset is imbalanced - consider data augmentation")


def main():
    parser = argparse.ArgumentParser(description='Organize oral cancer dataset')
    parser.add_argument('--source_dir', type=str, default='dataset',
                       help='Source directory containing the dataset')
    parser.add_argument('--dest_dir', type=str, default='data',
                       help='Destination directory for organized data')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='Ratio of data for training (default: 0.7)')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                       help='Ratio of data for validation (default: 0.15)')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                       help='Ratio of data for testing (default: 0.15)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducible splits')
    parser.add_argument('--clear_dest', action='store_true',
                       help='Clear destination directory before organizing')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Check if source directories exist
    cancer_dir = os.path.join(args.source_dir, 'Oral Cancer photos')
    normal_dir = os.path.join(args.source_dir, 'normal')
    
    if not os.path.exists(cancer_dir):
        logger.error(f"Cancer image directory not found: {cancer_dir}")
        return
    
    if not os.path.exists(normal_dir):
        logger.error(f"Normal image directory not found: {normal_dir}")
        return
    
    # Clear destination if requested
    if args.clear_dest and os.path.exists(args.dest_dir):
        logger.info(f"Clearing destination directory: {args.dest_dir}")
        shutil.rmtree(args.dest_dir)
    
    # Create directory structure
    create_directory_structure(args.dest_dir)
    
    # Get image files
    logger.info("Collecting image files...")
    cancer_files = get_image_files(cancer_dir)
    normal_files = get_image_files(normal_dir)
    
    logger.info(f"Found {len(cancer_files)} cancer images")
    logger.info(f"Found {len(normal_files)} normal/healthy images")
    
    if len(cancer_files) == 0 or len(normal_files) == 0:
        logger.error("No images found in one or both categories!")
        return
    
    # Split data
    logger.info("Splitting data...")
    cancer_splits = split_data(cancer_files, args.train_ratio, args.val_ratio, args.test_ratio)
    normal_splits = split_data(normal_files, args.train_ratio, args.val_ratio, args.test_ratio)
    
    # Copy files
    logger.info("Copying files...")
    copy_files(cancer_splits, 'cancer', args.dest_dir)
    copy_files(normal_splits, 'healthy', args.dest_dir)
    
    # Verify organization
    verify_organization(args.dest_dir)
    
    logger.info(f"\nDataset organization complete!")
    logger.info(f"Data is now ready for training in the '{args.dest_dir}' directory")
    
    # Show how to start training
    logger.info("\nTo start training:")
    logger.info("1. RegNetY320: python src/training/train.py --model regnet --data_dir data")
    logger.info("2. VGG16: python src/training/train.py --model vgg16 --data_dir data")


if __name__ == "__main__":
    main()