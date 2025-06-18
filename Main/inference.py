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
import time

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

def load_model(checkpoint_path, config, device):
    """Load the model from checkpoint."""
    # Create model with custom backbone for June 16th checkpoint
    if "20250616" in checkpoint_path:
        config['model']['backbone_type'] = 'custom'  # Override backbone type for June 16th model
    
    model = VehicleCounter(
        backbone_channels=config['model']['backbone_channels'],
        fpn_channels=config['model']['fpn_channels'],
        dropout_rate=config['model'].get('dropout_rate', 0.3),
        backbone_type=config['model'].get('backbone_type', 'se')
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

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

class VideoFrameProcessor:
    def __init__(self, config):
        self.transform = A.Compose([
            A.Resize(config['data']['image_size'], config['data']['image_size']),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

    def preprocess(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        transformed = self.transform(image=image)
        image_tensor = transformed['image'].unsqueeze(0)
        return image_tensor

def run_video_inference(video_path, output_path, model, processor, device):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width*3, height))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logging.info(f"Processing {frame_count} frames at {fps} FPS, resolution {width}x{height}")

    with torch.no_grad():
        for _ in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break
            input_tensor = processor.preprocess(frame).to(device)
            outputs = model(input_tensor)
            density_map = outputs['density_map']
            count = outputs['count']
            # global_count = outputs['global_count']  # Not used in visualization

            # Convert predictions to numpy
            density_map = density_map.squeeze().cpu().numpy()
            count = count.item()
            
            # Create visualization
            visualization = create_visualization(frame, density_map, count)
            
            # Write frame
            out.write(visualization)
    
    cap.release()
    out.release()
    logging.info(f"Saved output video to {output_path}")

def get_normalize_params(config):
    # Find the Normalize transform in val_transform
    for t in config['data']['val_transform']:
        if t.get('name', '').lower() == 'normalize':
            return t['mean'], t['std']
    # Fallback to defaults if not found
    return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

def preprocess_image(frame, config):
    mean, std = get_normalize_params(config)
    transform = A.Compose([
        A.Resize(config['data']['image_size'], config['data']['image_size']),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])
    transformed = transform(image=frame)
    return transformed['image'].unsqueeze(0)

def create_visualization(frame, density_map, count):
    """Create a side-by-side visualization of the original frame, density map, and overlay."""
    # Resize density map to match frame size
    density_map = cv2.resize(density_map, (frame.shape[1], frame.shape[0]))
    
    # Normalize density map for visualization
    density_map = (density_map - density_map.min()) / (density_map.max() - density_map.min() + 1e-8)
    density_map = (density_map * 255).astype(np.uint8)
    
    # Convert density map to RGB heatmap
    density_map_colored = cv2.applyColorMap(density_map, cv2.COLORMAP_JET)
    
    # Create overlay by blending original frame with density map
    overlay = cv2.addWeighted(frame, 0.7, density_map_colored, 0.3, 0)
    
    # Add count text with backdrop
    count_text = f"Count: {count:.1f}"
    
    # Add backdrop for text
    text_size = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
    text_x = 20
    text_y = 50
    cv2.rectangle(overlay, 
                 (text_x - 10, text_y - text_size[1] - 10),
                 (text_x + text_size[0] + 10, text_y + 10),
                 (0, 0, 0),  # Black background
                 -1)  # Filled rectangle
    
    # Add count text
    cv2.putText(overlay, count_text, (text_x, text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    
    # Combine images horizontally
    vis = np.hstack([frame, density_map_colored, overlay])
    return vis

def main():
    """Main function for video inference."""
    # Load configuration
    logging.info("Loading configuration from config.yaml")
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Load model
    model_path = "logs/20250616_214903/best_model.pth"  # June 16th model
    logging.info(f"Loading model from {model_path}")
    model = load_model(model_path, config, device)
    logging.info("Model loaded successfully")

    # Open video
    video_path = "traffic_wala_dataset/test/traffic.mp4"
    logging.info(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    logging.info("Video properties:")
    logging.info(f"- Resolution: {width}x{height}")
    logging.info(f"- FPS: {fps}")
    logging.info(f"- Total frames: {total_frames}")
    logging.info(f"- Duration: {duration:.2f} seconds")
    
    # Create output video writer
    output_path = "traffic_wala_dataset/test/traffic_output_june16.mp4"  # Different output filename
    logging.info(f"Creating output video: {output_path}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width*3, height))  # *3 for side-by-side visualization

    # Process frames
    logging.info("Starting frame processing...")
    processed_frames = 0
    start_time = time.time()
    pbar = tqdm(total=total_frames, desc="Processing frames")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess frame
        preprocessed = preprocess_image(frame, config)

        # Run inference
        with torch.no_grad():
            outputs = model(preprocessed.to(device))
            density_map = outputs['density_map'].squeeze().cpu().numpy()  # Remove batch and channel dims
            count = float(outputs['count'].cpu().numpy())  # Convert to scalar

        # Create visualization
        visualization = create_visualization(frame, density_map, count)
        
        # Write frame
        out.write(visualization)

        # Update progress
        processed_frames += 1
        pbar.update(1)

        # Log progress every 100 frames
        if processed_frames % 100 == 0:
            elapsed_time = time.time() - start_time
            fps = processed_frames / elapsed_time
            logging.info(f"Processed {processed_frames}/{total_frames} frames ({processed_frames/total_frames*100:.1f}%) - Processing speed: {fps:.1f} FPS")

    # Clean up
    pbar.close()
    cap.release()
    out.release()

    # Log final statistics
    total_time = time.time() - start_time
    avg_time_per_frame = total_time / processed_frames * 1000  # ms
    avg_fps = processed_frames / total_time

    logging.info("\nProcessing complete!")
    logging.info(f"Total processing time: {total_time:.2f} seconds")
    logging.info(f"Average frame processing time: {avg_time_per_frame:.1f} ms")
    logging.info(f"Average processing speed: {avg_fps:.1f} FPS")
    logging.info(f"Output saved to: {output_path}")

if __name__ == "__main__":
    main()