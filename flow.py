import os
import torch
import cv2
import numpy as np
from datetime import datetime
from tqdm import tqdm
from scipy.ndimage import center_of_mass
import yaml
import albumentations as A
from albumentations.pytorch import ToTensorV2
from models.vehicle_counter import VehicleCounter
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = 'logs/20250617_113933/best_model.pth'  # June 17th model
VIDEO_PATH = 'traffic_wala_dataset/test/traffic.mp4'
OUTPUT_PATH = 'traffic_wala_dataset/test/traffic_flow_analysis.mp4'

def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def get_transform():
    return A.Compose([
        A.Resize(640, 640),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])

def preprocess_frame(frame, transform):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    transformed = transform(image=frame_rgb)
    return transformed["image"].unsqueeze(0)

def get_density_centroid(density_map):
    density_np = density_map.squeeze().cpu().numpy()
    threshold = 0.3 * density_np.max()
    mask = (density_np > threshold).astype(np.float32)
    if mask.sum() == 0:
        return None
    centroid = center_of_mass(mask)
    return centroid[::-1]  # Convert (y,x) to (x,y)

def interpret_angle(angle):
    if -45 <= angle <= 45:
        return "Right"
    elif 45 < angle <= 135:
        return "Down"
    elif angle > 135 or angle < -135:
        return "Left"
    elif -135 <= angle < -45:
        return "Up"
    else:
        return "Unknown"

def categorize_congestion(score):
    if score < 0.33:
        return "Low", (0, 255, 0)  # Green
    elif score < 0.66:
        return "Moderate", (0, 255, 255)  # Yellow
    else:
        return "High", (0, 0, 255)  # Red

def create_visualization(frame, density_map, count, congestion, direction, max_capacity):
    """Create visualization with original frame, density map, and overlay with flow information."""
    # Resize density map and convert to heatmap
    density_map = cv2.resize(density_map, (frame.shape[1], frame.shape[0]))
    density_map = (density_map - density_map.min()) / (density_map.max() - density_map.min() + 1e-8)
    density_map = (density_map * 255).astype(np.uint8)
    density_map = cv2.applyColorMap(density_map, cv2.COLORMAP_JET)
    
    # Create overlay
    overlay = frame.copy()
    congestion_level, color = categorize_congestion(congestion)
    
    # Add information with backdrop
    info_text = [
        f"Count: {count:.1f}",
        f"Congestion: {congestion:.2f} ({congestion_level})",
        f"Direction: {direction}",
        f"Max Capacity: {max_capacity:.1f}"
    ]
    
    for i, text in enumerate(info_text):
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        text_x = 20
        text_y = 40 + i * 40
        
        # Add backdrop
        cv2.rectangle(overlay, 
                     (text_x - 10, text_y - text_size[1] - 10),
                     (text_x + text_size[0] + 10, text_y + 10),
                     (0, 0, 0),
                     -1)
        
        # Add text
        cv2.putText(overlay, text, (text_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    
    # Combine visualizations horizontally
    return np.hstack([frame, density_map, overlay])

def main():
    # Load configuration and model
    config = load_config()
    transform = get_transform()
    
    logging.info(f"Loading model from {MODEL_PATH}")
    model = VehicleCounter(
        backbone_channels=config['model']['backbone_channels'],
        fpn_channels=config['model']['fpn_channels'],
        dropout_rate=config['model'].get('dropout_rate', 0.3),
        backbone_type=config['model'].get('backbone_type', 'se')
    )
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    logging.info("Model loaded successfully")
    
    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        logging.error(f"Error: Could not open video {VIDEO_PATH}")
        return
        
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width*3, height))
    
    # Initialize variables for flow analysis
    prev_centroid = None
    max_capacity = 0
    direction = "Calculating..."
    frame_counts = []
    directions = []
    
    # Process frames
    logging.info("Starting flow analysis...")
    pbar = tqdm(total=total_frames, desc="Processing")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Preprocess frame
        input_tensor = preprocess_frame(frame, transform)
        
        # Run inference
        with torch.no_grad():
            outputs = model(input_tensor.to(DEVICE))
            density_map = outputs['density_map'].cpu().numpy()[0, 0]
            count = float(outputs['count'].cpu().numpy())
            
        # Update max capacity
        frame_counts.append(count)
        max_capacity = max(max_capacity, np.percentile(frame_counts, 90))
        
        # Calculate congestion
        congestion = min(count / max_capacity if max_capacity > 0 else 0, 1.0)
        
        # Calculate direction
        curr_centroid = get_density_centroid(torch.tensor(density_map))
        if prev_centroid is not None and curr_centroid is not None:
            dx = curr_centroid[0] - prev_centroid[0]
            dy = curr_centroid[1] - prev_centroid[1]
            angle = np.arctan2(dy, dx) * 180 / np.pi
            directions.append(interpret_angle(angle))
            
            # Update overall direction
            if len(directions) > 10:  # Use last 10 frames for direction
                recent_dirs = directions[-10:]
                if len(set(recent_dirs)) > 2:
                    direction = "Bidirectional"
                else:
                    direction = max(set(recent_dirs), key=recent_dirs.count)
        
        prev_centroid = curr_centroid
        
        # Create visualization
        vis = create_visualization(frame, density_map, count, congestion, direction, max_capacity)
        
        # Write frame
        out.write(vis)
        pbar.update(1)
        
    # Cleanup
    pbar.close()
    cap.release()
    out.release()
    
    logging.info(f"\nFlow analysis complete!")
    logging.info(f"Output saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
