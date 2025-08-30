import os
import cv2
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OralCancerDataset(Dataset):
    """Dataset class for oral cancer binary classification."""
    
    def __init__(
        self,
        root_dir: str,
        transform=None,
        mode: str = 'train',
        image_size: Tuple[int, int] = (224, 224)
    ):
        """
        Args:
            root_dir: Root directory containing train/val/test folders
            transform: Albumentations transform pipeline
            mode: One of 'train', 'val', 'test'
            image_size: Target image size (height, width)
        """
        self.root_dir = root_dir
        self.mode = mode
        self.image_size = image_size
        self.transform = transform
        
        # Setup paths
        self.data_dir = os.path.join(root_dir, mode)
        self.classes = ['healthy', 'cancer']
        self.class_to_idx = {'healthy': 0, 'cancer': 1}
        
        # Load image paths and labels
        self.images = []
        self.labels = []
        
        for class_name in self.classes:
            class_dir = os.path.join(self.data_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.images.append(os.path.join(class_dir, img_name))
                        self.labels.append(self.class_to_idx[class_name])
        
        logger.info(f"Loaded {len(self.images)} images for {mode} set")
        logger.info(f"Class distribution: {self._get_class_distribution()}")
    
    def _get_class_distribution(self) -> Dict[str, int]:
        """Get the distribution of classes in the dataset."""
        unique, counts = np.unique(self.labels, return_counts=True)
        return {self.classes[i]: count for i, count in zip(unique, counts)}
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a single sample from the dataset."""
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image, label


def get_transforms(
    mode: str = 'train',
    image_size: Tuple[int, int] = (224, 224),
    model_type: str = 'regnet'
) -> A.Compose:
    """
    Get augmentation pipeline for training/validation/test.
    
    Args:
        mode: One of 'train', 'val', 'test'
        image_size: Target image size
        model_type: 'regnet' or 'vgg16' for model-specific preprocessing
    
    Returns:
        Albumentations Compose object
    """
    # ImageNet normalization values
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    if mode == 'train':
        transform = A.Compose([
            # Resize with padding to maintain aspect ratio
            A.LongestMaxSize(max_size=256),
            A.PadIfNeeded(min_height=256, min_width=256, 
                         border_mode=cv2.BORDER_CONSTANT),
            
            # Augmentations
            A.RandomCrop(height=image_size[0], width=image_size[1]),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.1, 
                contrast_limit=0.1, 
                p=0.5
            ),
            A.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
                hue=0.1,
                p=0.3
            ),
            # A.GaussNoise(var_limit=(0, 10), p=0.3),  # Commented out due to compatibility issues
            
            # Normalization
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])
    else:
        # Validation and test transforms
        transform = A.Compose([
            A.LongestMaxSize(max_size=256),
            A.PadIfNeeded(min_height=256, min_width=256,
                         border_mode=cv2.BORDER_CONSTANT),
            A.CenterCrop(height=image_size[0], width=image_size[1]),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])
    
    return transform


def create_data_loaders(
    data_dir: str,
    batch_size: int = 32,
    image_size: Tuple[int, int] = (224, 224),
    num_workers: int = 0,  # Set to 0 for Windows compatibility
    model_type: str = 'regnet'
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for training, validation, and test sets.
    
    Args:
        data_dir: Root directory containing data
        batch_size: Batch size for training
        image_size: Target image size
        num_workers: Number of worker processes for data loading
        model_type: Type of model for preprocessing
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create transforms
    train_transform = get_transforms('train', image_size, model_type)
    val_transform = get_transforms('val', image_size, model_type)
    test_transform = get_transforms('test', image_size, model_type)
    
    # Create datasets
    train_dataset = OralCancerDataset(
        data_dir, transform=train_transform, mode='train', image_size=image_size
    )
    val_dataset = OralCancerDataset(
        data_dir, transform=val_transform, mode='val', image_size=image_size
    )
    test_dataset = OralCancerDataset(
        data_dir, transform=test_transform, mode='test', image_size=image_size
    )
    
    # Calculate class weights for imbalanced data
    train_labels = np.array(train_dataset.labels)
    class_counts = np.bincount(train_labels)
    class_weights = len(train_labels) / (len(class_counts) * class_counts)
    
    logger.info(f"Class weights: {class_weights}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False  # Disabled for CPU training
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False  # Disabled for CPU training
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False  # Disabled for CPU training
    )
    
    return train_loader, val_loader, test_loader, class_weights


def visualize_augmentations(
    data_dir: str,
    num_samples: int = 4,
    save_path: str = 'outputs/figures/augmentations.png'
):
    """Visualize augmentations on sample images."""
    import matplotlib.pyplot as plt
    
    # Create dataset with augmentations
    transform = get_transforms('train')
    dataset = OralCancerDataset(data_dir, transform=transform, mode='train')
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(12, 3*num_samples))
    
    for i in range(num_samples):
        # Get original image
        img_path = dataset.images[i]
        original = cv2.imread(img_path)
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        original_resized = cv2.resize(original, (224, 224))
        
        # Show original
        axes[i, 0].imshow(original_resized)
        axes[i, 0].set_title(f'Original\nClass: {dataset.classes[dataset.labels[i]]}')
        axes[i, 0].axis('off')
        
        # Show augmented versions
        for j in range(1, 4):
            augmented, _ = dataset[i]
            # Denormalize for visualization
            augmented = augmented.numpy().transpose(1, 2, 0)
            augmented = (augmented * np.array([0.229, 0.224, 0.225]) + 
                        np.array([0.485, 0.456, 0.406]))
            augmented = np.clip(augmented, 0, 1)
            
            axes[i, j].imshow(augmented)
            axes[i, j].set_title(f'Augmented {j}')
            axes[i, j].axis('off')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Augmentation visualization saved to {save_path}")


if __name__ == "__main__":
    # Test the data pipeline
    data_dir = "data"
    
    # Create sample data structure if it doesn't exist
    for split in ['train', 'val', 'test']:
        for class_name in ['healthy', 'cancer']:
            os.makedirs(os.path.join(data_dir, split, class_name), exist_ok=True)
    
    print("Data preprocessing pipeline created successfully!")
    print("Place your images in the following structure:")
    print("data/")
    print("├── train/")
    print("│   ├── healthy/")
    print("│   └── cancer/")
    print("├── val/")
    print("│   ├── healthy/")
    print("│   └── cancer/")
    print("└── test/")
    print("    ├── healthy/")
    print("    └── cancer/")