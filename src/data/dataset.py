import os
import torch
from typing import Tuple, Optional
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnimalDataset(Dataset):
    def __init__(
        self,
        transform: Optional[transforms.Compose] = None,
        split: str = 'train'
    ):
        """
        Initialize the dataset.
        
        Args:
            transform: Optional transform to be applied to images
            split: 'train' or 'test'
        """
        self.transform = transform
        self.split = split
        self.data = []
        self.classes = ["bird", "cat", "dog", "horse"]
        
        # Get data directory
        self.data_dir = os.path.join("temp_dataset", "model_dataset", split)
        if not os.path.exists(self.data_dir):
            raise RuntimeError(f"Data directory not found: {self.data_dir}. Run data processing first.")
        
        self._load_data()

    def _load_data(self):
        """Load image paths and labels."""
        logger.info(f"Loading {self.split} dataset...")
        
        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.exists(class_dir):
                raise RuntimeError(f"Class directory not found: {class_dir}")
            
            # Get all images
            images = [f for f in os.listdir(class_dir) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            # Add image paths and labels
            for img_name in images:
                img_path = os.path.join(class_dir, img_name)
                self.data.append((img_path, class_idx))
        
        # Print statistics
        class_counts = self._get_class_counts()
        
        logger.info(f"{self.split} dataset statistics:")
        for class_name, count in zip(self.classes, class_counts):
            logger.info(f"{class_name}: {count} images")
        logger.info(f"Total: {len(self.data)} images")

    def _get_class_counts(self):
        """Get number of images per class."""
        return [sum(1 for _, label in self.data if label == i) 
                for i in range(len(self.classes))]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a single item from the dataset."""
        img_path, label = self.data[idx]
        
        try:
            # Load and convert image
            image = Image.open(img_path).convert('RGB')
            
            # Apply transforms if any
            if self.transform:
                image = self.transform(image)
                
            return image, label
            
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            # Return black image as fallback
            if self.transform:
                return torch.zeros((3, 224, 224)), label
            return Image.new('RGB', (224, 224), 'black'), label


def get_data_loaders(
    batch_size: int = 32,
    val_split: float = 0.15
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        batch_size: Batch size for data loaders
        val_split: Fraction of training data to use for validation
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # Training augmentations
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Validation/Test transforms
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Load datasets
    train_full = AnimalDataset(transform=train_transform, split='train')
    
    # Split training data
    train_size = int((1.0 - val_split) * len(train_full))
    val_size = len(train_full) - train_size
    
    train_dataset, val_dataset = random_split(train_full, [train_size, val_size])
    
    # Load test dataset
    test_dataset = AnimalDataset(transform=eval_transform, split='test')
    
    logger.info(f"Final split sizes:")
    logger.info(f"Train: {len(train_dataset)}")
    logger.info(f"Validation: {len(val_dataset)}")
    logger.info(f"Test: {len(test_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader