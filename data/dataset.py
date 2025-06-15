import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path

class VehicleCountingDataset(Dataset):
    def __init__(self, root_dir, transform=None, split='train'):
        """
        Args:
            root_dir (string): Directory with all the images and labels
            transform (callable, optional): Optional transform to be applied on a sample
            split (string): 'train' or 'valid'
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.split = split
        
        # Set up paths
        self.img_dir = self.root_dir / split / 'images'
        self.csv_path = self.root_dir / split / 'labels' / '_annotations.csv'
        
        # Verify paths exist
        if not self.img_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Annotations file not found: {self.csv_path}")
        
        # Load annotations
        self.annotations = pd.read_csv(self.csv_path)
        
        # Get unique image filenames
        self.image_files = sorted(list(set(self.annotations['filename'])))
        
        print(f"Found {len(self.image_files)} images in {split} dataset")
        print(f"Image directory: {self.img_dir}")
        print(f"Annotations file: {self.csv_path}")
        
    def __len__(self):
        return len(self.image_files)
    
    def generate_density_map(self, boxes, img_size):
        """Generate density map from bounding boxes"""
        density_map = np.zeros(img_size, dtype=np.float32)
        
        for box in boxes:
            xmin, ymin, xmax, ymax = box
            # Calculate center point
            center_x = int((xmin + xmax) / 2)
            center_y = int((ymin + ymax) / 2)
            
            # Add Gaussian kernel at center point
            sigma = 4  # Adjust this value to control the spread of the Gaussian
            x = np.arange(0, img_size[1], 1, float)
            y = np.arange(0, img_size[0], 1, float)
            y = y[:, np.newaxis]
            
            # Create 2D Gaussian kernel
            gaussian = np.exp(-((x - center_x) ** 2 + (y - center_y) ** 2) / (2 * sigma ** 2))
            density_map += gaussian
            
        return density_map
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = self.img_dir / img_name
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        # Get all boxes for this image
        img_boxes = self.annotations[self.annotations['filename'] == img_name]
        boxes = img_boxes[['xmin', 'ymin', 'xmax', 'ymax']].values
        
        # Generate density map
        density_map = self.generate_density_map(boxes, image.shape[:2])
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=density_map)
            image = transformed['image']
            density_map = transformed['mask']
        
        # Convert density map to tensor and add channel dimension
        if not isinstance(density_map, torch.Tensor):
            density_map = torch.from_numpy(density_map)
        density_map = density_map.unsqueeze(0).float()
        
        # Get count from number of boxes
        count = torch.tensor(len(boxes), dtype=torch.float32)
        
        return {
            'image': image,
            'density_map': density_map,
            'count': count,
            'global_count': count,
            'points': boxes,
            'image_name': img_name
        } 