import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from models.vehicle_counter import VehicleCounter
import yaml
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from pathlib import Path
import torch.cuda.amp as amp
from concurrent.futures import ThreadPoolExecutor
import pickle
from PIL import Image
import torchvision.transforms as transforms

class ImageProcessor:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.transform = A.Compose([
            A.Resize(config['data']['image_size'], config['data']['image_size']),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        self.cache_dir = Path('cache')
        self.cache_dir.mkdir(exist_ok=True)
        self.scaler = amp.GradScaler()
        
    def preprocess_image(self, image_path):
        cache_path = self.cache_dir / f"{Path(image_path).stem}_processed.pkl"
        
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transformed = self.transform(image=image)
        image_tensor = transformed['image'].unsqueeze(0)
        
        with open(cache_path, 'wb') as f:
            pickle.dump((image, image_tensor), f)
            
        return image, image_tensor

def load_model(checkpoint_path, device):
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create model
    model = VehicleCounter().to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, config

def process_batch(model, image_tensors, device):
    with torch.no_grad(), amp.autocast():
        image_tensors = torch.cat(image_tensors, dim=0).to(device)
        density_maps, counts, _ = model(image_tensors)
        return density_maps.cpu().numpy(), counts.cpu().numpy()

def visualize_results(image, density_map, count, save_path=None):
    # Create figure
    plt.figure(figsize=(15, 5))
    
    # Plot original image
    plt.subplot(131)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    # Plot density map
    plt.subplot(132)
    plt.imshow(density_map, cmap='jet')
    plt.colorbar()
    plt.title(f'Density Map (Count: {count:.1f})')
    plt.axis('off')
    
    # Plot overlay
    plt.subplot(133)
    plt.imshow(image)
    plt.imshow(density_map, cmap='jet', alpha=0.5)
    plt.colorbar()
    plt.title('Overlay')
    plt.axis('off')
    
    # Save or show
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=150)
        plt.close()
    else:
        plt.show()

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model, config = load_model('checkpoints/best_model.pth', device)
    
    # Initialize processor
    processor = ImageProcessor(config, device)
    
    # Create output directory
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    # Get all image paths
    test_dir = Path('traffic_wala_dataset/valid/images')
    image_paths = list(test_dir.glob('*.jpg')) + list(test_dir.glob('*.png')) + list(test_dir.glob('*.jpeg'))
    
    # Process images in batches
    batch_size = 4 if torch.cuda.is_available() else 1
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing batches"):
        batch_paths = image_paths[i:i + batch_size]
        images = []
        image_tensors = []
        
        # Preprocess batch
        for image_path in batch_paths:
            image, image_tensor = processor.preprocess_image(image_path)
            images.append(image)
            image_tensors.append(image_tensor)
        
        # Process batch
        density_maps, counts = process_batch(model, image_tensors, device)
        
        # Save results
        for idx, (image_path, image, density_map, count) in enumerate(zip(batch_paths, images, density_maps, counts)):
            save_path = results_dir / f"{image_path.stem}_result.png"
            visualize_results(image, density_map, count, save_path)
            print(f'Processed {image_path.name}: Count = {count:.1f}')

if __name__ == '__main__':
    main()

# Load the trained model
model = VehicleCounter(...)  # Initialize with the same parameters as during training
model.load_state_dict(torch.load('checkpoints/best_model.pth'))
model.eval()  # Set the model to evaluation mode

# Define the same transforms used during training
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Adjust size as needed
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load and preprocess the new image
image_path = r"C:\Users\User\Work\College\AIMS\Summer Projects\Vehicle Detection\traffic_wala_dataset\valid\images\2_mp4-1_jpg.rf.b52c12a365fbeb71e302f0505038959b.jpg"  # Replace with your image path
image = Image.open(image_path).convert('RGB')
image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

# Generate predictions
with torch.no_grad():
    predictions = model(image_tensor)

# Post-process predictions
# This step depends on your model's output format
# For example, if your model outputs bounding boxes, you might need to convert them to pixel coordinates

# Visualize or use the results
# For example, you can use OpenCV or Matplotlib to draw bounding boxes on the image 