"""
Alternative Vehicle Detection and Counting Pipeline
Inspired by RefineDet architecture with GMM background subtraction
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
from pathlib import Path
import logging
from typing import List, Tuple, Dict, Optional

# Import our modules
from detection.refinedet import RefineDet
from counting.gmm import GMMBackgroundSubtractor
from counting.count import VehicleCounter
from utils.metrics import DetectionMetrics

class VehicleDetectionPipeline:
    """
    End-to-end vehicle detection and counting pipeline
    """
    
    def __init__(self, 
                 detection_model_path: str,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the pipeline
        
        Args:
            detection_model_path: Path to trained RefineDet model
            device: Device to run inference on
        """
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        # Initialize detection model
        self.detector = RefineDet(num_classes=1)  # Only vehicle class
        self.load_detection_model(detection_model_path)
        self.detector.to(device)
        self.detector.eval()
        
        # Initialize counting components
        self.gmm = GMMBackgroundSubtractor()
        self.counter = VehicleCounter()
        
        # Metrics
        self.metrics = DetectionMetrics()
        
        self.logger.info(f"Pipeline initialized on {device}")
    
    def load_detection_model(self, model_path: str):
        """Load trained detection model"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.detector.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.detector.load_state_dict(checkpoint)
            self.logger.info(f"Detection model loaded from {model_path}")
        except Exception as e:
            self.logger.error(f"Failed to load detection model: {e}")
            raise
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for detection model"""
        # Resize to model input size
        image_resized = cv2.resize(image, (512, 512))
        
        # Normalize to [0, 1]
        image_normalized = image_resized.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0)
        
        return image_tensor.to(self.device)
    
    def postprocess_detections(self, 
                              detections: torch.Tensor, 
                              original_shape: Tuple[int, int]) -> List[Dict]:
        """
        Post-process detection outputs to get bounding boxes
        
        Args:
            detections: Model output [batch, num_detections, 6] (x1, y1, x2, y2, conf, class)
            original_shape: Original image shape (height, width)
            
        Returns:
            List of detection dictionaries with bbox, confidence, class_id
        """
        h, w = original_shape
        
        # Filter by confidence threshold
        conf_threshold = 0.5
        mask = detections[0, :, 4] > conf_threshold
        filtered_dets = detections[0, mask]
        
        results = []
        for det in filtered_dets:
            x1, y1, x2, y2, conf, class_id = det.cpu().numpy()
            
            # Scale back to original image size
            x1 = int(x1 * w)
            y1 = int(y1 * h)
            x2 = int(x2 * w)
            y2 = int(y2 * h)
            
            results.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': float(conf),
                'class_id': int(class_id)
            })
        
        return results
    
    def detect_vehicles(self, image: np.ndarray) -> List[Dict]:
        """
        Detect vehicles in image
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of vehicle detections
        """
        # Preprocess
        input_tensor = self.preprocess_image(image)
        
        # Inference
        with torch.no_grad():
            detections = self.detector(input_tensor)
        
        # Postprocess
        detections = self.postprocess_detections(detections, image.shape[:2])
        
        return detections
    
    def count_vehicles(self, image: np.ndarray, detections: List[Dict]) -> int:
        """
        Count vehicles using GMM background subtraction and tracking
        
        Args:
            image: Input image
            detections: Vehicle detections from detector
            
        Returns:
            Vehicle count
        """
        # Update GMM background model
        self.gmm.update(image)
        
        # Get foreground mask
        fg_mask = self.gmm.get_foreground_mask(image)
        
        # Count vehicles using detection + tracking
        count = self.counter.count_vehicles(image, detections, fg_mask)
        
        return count
    
    def process_image(self, image: np.ndarray) -> Dict:
        """
        Process single image for detection and counting
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Dictionary with detections, count, and processing info
        """
        # Detect vehicles
        detections = self.detect_vehicles(image)
        
        # Count vehicles
        vehicle_count = self.count_vehicles(image, detections)
        
        # Calculate metrics
        metrics = {
            'num_detections': len(detections),
            'avg_confidence': np.mean([d['confidence'] for d in detections]) if detections else 0.0,
            'processing_time': 0.0  # Could add timing if needed
        }
        
        return {
            'detections': detections,
            'vehicle_count': vehicle_count,
            'metrics': metrics
        }
    
    def process_video(self, video_path: str, output_path: Optional[str] = None) -> Dict:
        """
        Process video for vehicle detection and counting
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video (optional)
            
        Returns:
            Dictionary with per-frame results and summary
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Video info
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Output video writer
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_results = []
        frame_count = 0
        
        self.logger.info(f"Processing video: {video_path}")
        self.logger.info(f"Video info: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            result = self.process_image(frame)
            frame_results.append(result)
            
            # Draw results on frame
            annotated_frame = self.draw_results(frame, result)
            
            # Write output frame
            if out:
                out.write(annotated_frame)
            
            frame_count += 1
            if frame_count % 100 == 0:
                self.logger.info(f"Processed {frame_count}/{total_frames} frames")
        
        cap.release()
        if out:
            out.release()
        
        # Calculate video summary
        total_vehicles = sum(r['vehicle_count'] for r in frame_results)
        avg_vehicles_per_frame = total_vehicles / len(frame_results) if frame_results else 0
        
        summary = {
            'total_frames': len(frame_results),
            'total_vehicles_detected': total_vehicles,
            'avg_vehicles_per_frame': avg_vehicles_per_frame,
            'frame_results': frame_results
        }
        
        self.logger.info(f"Video processing complete. Total vehicles: {total_vehicles}")
        return summary
    
    def draw_results(self, image: np.ndarray, result: Dict) -> np.ndarray:
        """
        Draw detection and counting results on image
        
        Args:
            image: Input image
            result: Processing result from process_image
            
        Returns:
            Annotated image
        """
        annotated = image.copy()
        
        # Draw detections
        for det in result['detections']:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw confidence
            label = f"Vehicle: {conf:.2f}"
            cv2.putText(annotated, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw count
        count = result['vehicle_count']
        cv2.putText(annotated, f"Count: {count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        
        return annotated
    
    def evaluate_on_dataset(self, dataset_path: str) -> Dict:
        """
        Evaluate pipeline on dataset
        
        Args:
            dataset_path: Path to dataset directory
            
        Returns:
            Evaluation metrics
        """
        # This would integrate with your YOLO dataset
        # For now, return placeholder
        return {
            'mAP': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0
        }


def main():
    """Example usage of the pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Vehicle Detection and Counting Pipeline')
    parser.add_argument('--model', required=True, help='Path to detection model')
    parser.add_argument('--input', required=True, help='Input image/video path')
    parser.add_argument('--output', help='Output path for video')
    parser.add_argument('--device', default='auto', help='Device to use (cuda/cpu/auto)')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # Initialize pipeline
    pipeline = VehicleDetectionPipeline(args.model, device)
    
    # Process input
    input_path = Path(args.input)
    if input_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
        # Process image
        image = cv2.imread(str(input_path))
        if image is None:
            raise ValueError(f"Could not load image: {input_path}")
        
        result = pipeline.process_image(image)
        annotated = pipeline.draw_results(image, result)
        
        # Save result
        output_path = args.output or f"result_{input_path.name}"
        cv2.imwrite(output_path, annotated)
        
        print(f"Vehicle count: {result['vehicle_count']}")
        print(f"Detections: {len(result['detections'])}")
        
    elif input_path.suffix.lower() in ['.mp4', '.avi', '.mov']:
        # Process video
        result = pipeline.process_video(str(input_path), args.output)
        print(f"Video processing complete:")
        print(f"Total vehicles: {result['total_vehicles_detected']}")
        print(f"Average per frame: {result['avg_vehicles_per_frame']:.2f}")
    
    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}")


if __name__ == "__main__":
    main() 