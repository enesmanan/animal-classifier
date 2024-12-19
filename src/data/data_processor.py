import os
import shutil
import zipfile
import requests
import numpy as np
from tqdm import tqdm
from PIL import Image
import logging

from utils.image_utils import ImageProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self):
        self.raw_dir = "temp_dataset/raw_images"
        self.processed_dir = "temp_dataset/processed_images"
        self.final_dir = "temp_dataset/model_dataset"
        self.image_processor = ImageProcessor()
        self.target_size = (224, 224)
        
        # Create directories
        for dir_path in [self.raw_dir, self.processed_dir, self.final_dir]:
            os.makedirs(dir_path, exist_ok=True)

    def download_raw_data(self):
        """Download and extract raw_images.zip from HuggingFace."""
        zip_path = "temp_dataset/raw_images.zip"
        if os.path.exists(zip_path):
            logger.info("Raw images already downloaded.")
            return

        url = "https://huggingface.co/datasets/mertcobanov/train-your-first-classifier/resolve/main/raw_images.zip"
        logger.info("Downloading raw images...")
        
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(zip_path, 'wb') as f, tqdm(
            desc="Downloading",
            total=total_size,
            unit='iB',
            unit_scale=True
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                pbar.update(size)
        
        # Extract
        logger.info("Extracting raw images...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.raw_dir)

    def process_single_image(self, image_path: str, save_path: str):
        """Process a single image with smart crop and resize."""
        try:
            # Load and convert image
            image = Image.open(image_path).convert('RGB')
            
            # Smart crop to square
            cropped = self.image_processor.smart_crop(image, square=True)
            
            # Resize to target size
            processed = cropped.resize(self.target_size, Image.LANCZOS)
            
            # Save processed image
            processed.save(save_path, 'JPEG', quality=95)
            return True
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {str(e)}")
            return False

    def process_images(self):
        """Process all raw images."""
        logger.info("Processing images...")
        
        base_dir = os.path.join(self.raw_dir, "raw_images")
        if not os.path.exists(base_dir):
            raise RuntimeError(f"Raw images directory not found at: {base_dir}")
        
        for class_name in os.listdir(base_dir):
            class_dir = os.path.join(base_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            
            logger.info(f"Processing class: {class_name}")
            
            # Create processed class directory
            processed_class_dir = os.path.join(self.processed_dir, class_name)
            os.makedirs(processed_class_dir, exist_ok=True)
            
            # Get all image files
            image_files = [f for f in os.listdir(class_dir) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            # Process each image
            processed_count = 0
            for img_file in tqdm(image_files, desc=f"Processing {class_name}"):
                src_path = os.path.join(class_dir, img_file)
                dst_path = os.path.join(processed_class_dir, img_file)
                
                if self.process_single_image(src_path, dst_path):
                    processed_count += 1
            
            logger.info(f"Successfully processed {processed_count}/{len(image_files)} images for {class_name}")

    def create_train_test_split(self, test_ratio: float = 0.2):
        logger.info("Creating train/test split...")
        
        # Create train/test directories
        train_dir = os.path.join(self.final_dir, "train")
        test_dir = os.path.join(self.final_dir, "test")
        
        for class_name in os.listdir(self.processed_dir):
            class_path = os.path.join(self.processed_dir, class_name)
            if not os.path.isdir(class_path):
                continue
            
            # Create class directories
            train_class_dir = os.path.join(train_dir, class_name)
            test_class_dir = os.path.join(test_dir, class_name)
            os.makedirs(train_class_dir, exist_ok=True)
            os.makedirs(test_class_dir, exist_ok=True)
            
            # Get all processed images
            images = [f for f in os.listdir(class_path)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if not images:
                logger.warning(f"No images found for class {class_name}")
                continue
            
            # Random split
            np.random.shuffle(images)
            split_idx = int(len(images) * (1 - test_ratio))
            train_images = images[:split_idx]
            test_images = images[split_idx:]
            
            # Copy images to respective directories
            for img in train_images:
                shutil.copy2(
                    os.path.join(class_path, img),
                    os.path.join(train_class_dir, img)
                )
            
            for img in test_images:
                shutil.copy2(
                    os.path.join(class_path, img),
                    os.path.join(test_class_dir, img)
                )
            
            logger.info(f"{class_name}: {len(train_images)} train, {len(test_images)} test")

    def process_pipeline(self):
        self.download_raw_data()
        self.process_images()
        self.create_train_test_split()
        logger.info("Data processing complete!")


def prepare_dataset():
    processor = DataProcessor()
    processor.process_pipeline()