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
        self.label_dir = self.root_dir / split / 'labels'
        
        # Verify paths exist
        if not self.img_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")
        if not self.label_dir.exists():
            raise FileNotFoundError(f"Label directory not found: {self.label_dir}")
        
        # Get all image filenames
        self.image_files = sorted([f for f in os.listdir(self.img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f"Found {len(self.image_files)} images in {split} dataset")
        print(f"Image directory: {self.img_dir}")
        print(f"Label directory: {self.label_dir}")

    def parse_yolo_label(self, label_path, img_width, img_height):
        """Parse YOLO .txt label file and return bounding boxes in pixel coordinates."""
        boxes = []
        if not os.path.exists(label_path):
            return boxes
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls, x_center, y_center, w, h = map(float, parts)
                # Convert normalized to pixel coordinates
                x_center *= img_width
                y_center *= img_height
                w *= img_width
                h *= img_height
                xmin = int(x_center - w / 2)
                ymin = int(y_center - h / 2)
                xmax = int(x_center + w / 2)
                ymax = int(y_center + h / 2)
                boxes.append([xmin, ymin, xmax, ymax])
        return boxes

    def __len__(self):
        return len(self.image_files)
    
    def generate_density_map(self, boxes, img_size):
        """Generate density map from bounding boxes"""
        density_map = np.zeros(img_size, dtype=np.float32)
        for box in boxes:
            xmin, ymin, xmax, ymax = box
            center_x = int((xmin + xmax) / 2)
            center_y = int((ymin + ymax) / 2)
            sigma = 3  # Reduced from 4 to 3 for sharper density peaks
            x = np.arange(0, img_size[1], 1, float)
            y = np.arange(0, img_size[0], 1, float)
            y = y[:, np.newaxis]
            gaussian = np.exp(-((x - center_x) ** 2 + (y - center_y) ** 2) / (2 * sigma ** 2))
            gaussian /= gaussian.sum()  # Normalize so sum is 1
            density_map += gaussian
        return density_map
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = self.img_dir / img_name
        label_path = self.label_dir / (img_name.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt'))
        # Load image
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        img_height, img_width = image.shape[:2]
        # Parse YOLO label file
        boxes = self.parse_yolo_label(label_path, img_width, img_height)
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