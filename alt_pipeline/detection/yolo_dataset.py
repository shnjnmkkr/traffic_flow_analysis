import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

class YOLODetectionDataset(Dataset):
    """
    PyTorch Dataset for YOLO-format detection data.
    Loads images and bounding boxes from a directory, parses YOLO .txt files.
    Returns image tensor, boxes (x1, y1, x2, y2), and labels.
    """
    def __init__(self, images_dir, labels_dir, img_size=224, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.img_size = img_size
        self.transform = transform
        self.image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        label_path = os.path.join(self.labels_dir, os.path.splitext(img_name)[0] + '.txt')
        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.img_size, self.img_size))
        # Load labels
        boxes = []
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls, x, y, w, h = map(float, parts)
                    # Convert YOLO (cx, cy, w, h) normalized to (x1, y1, x2, y2) absolute
                    x1 = (x - w/2) * self.img_size
                    y1 = (y - h/2) * self.img_size
                    x2 = (x + w/2) * self.img_size
                    y2 = (y + h/2) * self.img_size
                    boxes.append([x1, y1, x2, y2])
                    labels.append(int(cls))
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)
        # Transform
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        return image, boxes, labels

    @staticmethod
    def collate_fn(batch):
        images, boxes, labels = zip(*batch)
        images = torch.stack(images, 0)
        return images, boxes, labels

# Example usage:
# dataset = YOLODetectionDataset('images/', 'labels/', img_size=224)
# image, boxes, labels = dataset[0] 