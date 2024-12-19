import logging
import os
from typing import List, Tuple, Union

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


class ImageProcessor:
    
    @staticmethod
    def check_is_dir(path: str) -> bool:
        """Check if the given path is a directory."""
        if not os.path.isdir(path):
            raise ValueError(f"Provided path: {path} is not a directory")
        return True

    @staticmethod
    def filter_images(files: List[str]) -> List[str]:
        """Filter image files based on their extensions."""
        valid_extensions = {".jpg", ".jpeg", ".png", ".webp"}
        return [
            file for file in files
            if any(file.lower().endswith(ext) for ext in valid_extensions)
        ]

    @staticmethod
    def load_image(image_path: str) -> Image.Image:
        """Load an image from path."""
        return Image.open(image_path).convert("RGB")

    @staticmethod
    def save_image(
        image: Union[np.ndarray, Image.Image], 
        save_path: str
    ) -> None:
        """Save image to the specified path."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        if not isinstance(image, Image.Image):
            raise ValueError("Input image must be a numpy array or PIL Image")

        if image.mode != "RGB":
            image = image.convert("RGB")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        image.save(save_path)
        logging.info(f"Saved image to {save_path}")

    def read_images_from_dir(self, dir_path: str) -> List[Image.Image]:
        """Read all images from a directory."""
        self.check_is_dir(dir_path)
        files = os.listdir(dir_path)
        image_files = self.filter_images(files)
        image_paths = [os.path.join(dir_path, file) for file in image_files]
        images = [self.load_image(path) for path in tqdm(image_paths)]
        logging.info(f"Loaded {len(images)} images from {dir_path}")
        return images

    def get_image_paths(self, dir_path: str) -> List[str]:
        """Get paths of all images in a directory."""
        self.check_is_dir(dir_path)
        files = os.listdir(dir_path)
        image_files = self.filter_images(files)
        return [os.path.join(dir_path, file) for file in image_files]

    @staticmethod
    def smart_crop(
        image: Image.Image, 
        square: bool = False
    ) -> Image.Image:
        """
        Crop image using SIFT feature detection.
        
        Args:
            image: Input PIL image
            square: If True, crop to square around features
            
        Returns:
            Cropped PIL image
        """
        # Convert PIL to OpenCV format
        img = np.array(image)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Detect features
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create(edgeThreshold=8)
        kp = sift.detect(gray, None)
        
        # Get feature points
        points = np.array([k.pt for k in kp])
        x_points, y_points = points[:, 0], points[:, 1]
        
        # Calculate boundaries
        x_min, x_max = int(x_points.min()), int(x_points.max())
        y_min, y_max = int(y_points.min()), int(y_points.max())
        
        if square:
            # Calculate square crop
            center_x = (x_max + x_min) // 2
            center_y = (y_max + y_min) // 2
            side = min(x_max - x_min, y_max - y_min)
            half_side = side // 2
            
            x_min = center_x - half_side
            x_max = center_x + half_side
            y_min = center_y - half_side
            y_max = center_y + half_side
        
        # Crop and convert back to PIL
        cropped = img[y_min:y_max, x_min:x_max]
        cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        return Image.fromarray(cropped)

    @staticmethod
    def resize_max_resolution(
        image: Image.Image,
        max_width: int,
        max_height: int
    ) -> Image.Image:
        """Resize image maintaining aspect ratio."""
        width, height = image.size
        if width > max_width or height > max_height:
            ratio = min(max_width / width, max_height / height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            return image.resize((new_width, new_height), Image.LANCZOS)
        return image

    @staticmethod
    def check_min_resolution(
        image: Image.Image,
        min_width: int,
        min_height: int
    ) -> bool:
        """Check if image meets minimum resolution requirements."""
        width, height = image.size
        return width >= min_width and height >= min_height