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
import logging
from skimage.transform import resize

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    try:
        logging.info(f"Loading model from: {checkpoint_path}")
        # Load config
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Create model
        model = VehicleCounter(
            backbone_channels=config['model']['backbone_channels'],
            fpn_channels=config['model']['fpn_channels'],
            dropout_rate=config['model'].get('dropout_rate', 0.3)
        ).to(device)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        logging.info(f"Checkpoint keys: {checkpoint.keys()}")
        
        # Try to load the model state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # If the checkpoint is just the state dict itself
            model.load_state_dict(checkpoint)
            
        model.eval()
        logging.info("Model loaded successfully")
        
        return model, config
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        raise

def process_batch(model, image_tensors, device):
    try:
        with torch.no_grad(), amp.autocast():
            image_tensors = torch.cat(image_tensors, dim=0).to(device)
            predictions = model(image_tensors)
            return predictions['density_map'].cpu().numpy(), predictions['count'].cpu().numpy()
    except Exception as e:
        logging.error(f"Error processing batch: {str(e)}")
        raise

def visualize_results(image, density_map, count, save_path):
    try:
        plt.figure(figsize=(15, 5))
        
        # Original image
        plt.subplot(131)
        plt.imshow(image)
        plt.title(f'Original Image\nCount: {count:.1f}')
        plt.axis('off')
        
        # Density map
        plt.subplot(132)
        plt.imshow(density_map[0], cmap='jet')
        plt.title('Density Map')
        plt.axis('off')
        
        # Overlay
        # Resize density map to match image size
        density_resized = resize(
            density_map[0],
            (image.shape[0], image.shape[1]),
            preserve_range=True,
            anti_aliasing=True
        )
        plt.subplot(133)
        plt.imshow(image)
        plt.imshow(density_resized, cmap='jet', alpha=0.5)
        plt.title('Overlay')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        logging.info(f"Saved visualization to: {save_path}")
    except Exception as e:
        logging.error(f"Error visualizing results: {str(e)}")
        raise

def main():
    try:
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {device}")
        
        # Load model
        checkpoint_path = r'C:\Users\User\Work\College\AIMS\Summer Projects\Vehicle Detection\logs\20250616_214903\best_model.pth'
        model, config = load_model(checkpoint_path, device)
        
        # Initialize processor
        processor = ImageProcessor(config, device)
        
        # Create output directory
        results_dir = Path('results')
        results_dir.mkdir(exist_ok=True)
        
        # Get all image paths
        image_paths = [Path(r'C:\Users\User\Work\College\AIMS\Summer Projects\Vehicle Detection\traffic_wala_dataset\valid\images\16_mp4-1_jpg.rf.3493f4b7618e207609847857a20dbaff.jpg')]
        logging.info(f"Testing on image: {image_paths[0]}")
        
        if len(image_paths) == 0:
            logging.error(f"No images found in {image_paths[0]}")
            return
        
        # Process images in batches
        batch_size = 4 if torch.cuda.is_available() else 1
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing batches"):
            batch_paths = image_paths[i:i + batch_size]
            images = []
            image_tensors = []
            
            # Preprocess batch
            for image_path in batch_paths:
                try:
                    image, image_tensor = processor.preprocess_image(image_path)
                    images.append(image)
                    image_tensors.append(image_tensor)
                except Exception as e:
                    logging.error(f"Error preprocessing image {image_path}: {str(e)}")
                    continue
            
            if not images:
                continue
                
            # Process batch
            try:
                density_maps, counts = process_batch(model, image_tensors, device)
                
                # Save results
                for idx, (image_path, image, density_map, count) in enumerate(zip(batch_paths, images, density_maps, counts)):
                    save_path = results_dir / f"{image_path.stem}_result.png"
                    visualize_results(image, density_map, count, save_path)
                    logging.info(f'Processed {image_path.name}: Count = {count:.1f}')
            except Exception as e:
                logging.error(f"Error processing batch: {str(e)}")
                continue

    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        raise

if __name__ == '__main__':
    main()

# Load the trained model
model = VehicleCounter(...)  # Initialize with the same parameters as during training
model.load_state_dict(torch.load(r'C:\Users\User\Work\College\AIMS\Summer Projects\Vehicle Detection\logs\20250616_214903\best_model.pth'))
model.eval()  # Set the model to evaluation mode

# Define the same transforms used during training
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Adjust size as needed
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load and preprocess the new image
image_path = r"C:\Users\User\Work\College\AIMS\Summer Projects\Vehicle Detection\traffic_wala_dataset\valid\images\16_mp4-1_jpg.rf.3493f4b7618e207609847857a20dbaff.jpg"  # Replace with your image path
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