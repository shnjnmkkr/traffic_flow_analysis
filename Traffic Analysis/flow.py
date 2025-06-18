import math
import os
import cv2
import numpy as np
import math
from ultralytics import YOLO
from statistics import mode

'''model.train(
    data="traffic_wala_dataset/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    name="traffic_yolov8_custom"
)
'''

# --- CONFIG ---
VIDEO_PATH = "traffic_test_data/traffic.mp4"
OUTPUT_VIDEO_PATH = "annotated_output_final.mp4"
MODEL_PATH = "best.pt"
VEHICLE_CLASSES = ['Vehicle']
UPDATE_INTERVAL = 30 
TRACKING_DIST_THRESHOLD = 50

# --- Load YOLO Model ---
model = YOLO(MODEL_PATH)
print(" Model Classes:", model.names)

# --- Utility Functions ---
def compute_density(boxes, frame_area):
    total_area = sum((x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in boxes)
    return (total_area / frame_area) * 100 if frame_area else 0

def get_centroids(boxes):
    return [((x1 + x2) // 2, (y1 + y2) // 2) for x1, y1, x2, y2 in boxes]

def interpret_angle(angle):
    if -45 <= angle <= 45:
        return "Right"
    elif 45 < angle <= 135:
        return "Down"
    elif angle > 135 or angle < -135:
        return "Left"
    elif -135 <= angle < -45:
        return "Up"
    return "Unknown"

def estimate_direction(centroids):
    directions = []
    prev = None
    for curr in centroids:
        if prev and curr:
            dx, dy = curr[0] - prev[0], curr[1] - prev[1]
            angle = math.degrees(math.atan2(dy, dx))
            directions.append(interpret_angle(angle))
        prev = curr
    if directions:
        if len(set(directions)) > 2:
            return "Bidirectional"
        return mode(directions)
    return "Unknown"

# --- Video Setup ---
cap = cv2.VideoCapture(VIDEO_PATH)
fps = int(cap.get(cv2.CAP_PROP_FPS))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (w, h))
frame_area = w * h
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# --- Estimate Max Capacity ---
print("Estimating traffic max capacity using every frame...")
sample_scores = []
sample_cap = cv2.VideoCapture(VIDEO_PATH)
while True:
    ret, frame = sample_cap.read()
    if not ret:
        break
    results = model(frame)[0]
    boxes = []
    if results.boxes:
        for box, cls in zip(results.boxes.xyxy.cpu().numpy(), results.boxes.cls.cpu().numpy().astype(int)):
            if model.names[cls] in VEHICLE_CLASSES:
                boxes.append(tuple(map(int, box)))
    density = compute_density(boxes, frame_area)
    sample_scores.append(density)
sample_cap.release()

max_capacity = np.percentile(sample_scores, 90) if sample_scores else 1

# --- State Tracking ---
vehicle_tracks = defaultdict(list)
vehicle_speeds = defaultdict(lambda: 0.0)
last_update_frame = defaultdict(lambda: -UPDATE_INTERVAL)

frame_idx = 0
congestion_scores = []
centroid_path = []
frames_data = []
total_vehicle_count = 0
vehicle_speed_list = []

# --- Frame-by-Frame Processing ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True, tracker="ultralytics/cfg/trackers/bytetrack.yaml")[0]
    current_boxes = []
    current_centroids = []

    if results and results.boxes and results.boxes.id is not None:
        boxes = results.boxes.xyxy.cpu().numpy()
        ids = results.boxes.id.cpu().numpy().astype(int)
        classes = results.boxes.cls.cpu().numpy().astype(int)

        for box, id, cls in zip(boxes, ids, classes):
            if model.names[cls] not in VEHICLE_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            current_boxes.append((x1, y1, x2, y2))
            current_centroids.append((cx, cy))
            vehicle_tracks[id].append((cx, cy, frame_idx))

            # Speed update
            if frame_idx - last_update_frame[id] >= UPDATE_INTERVAL and len(vehicle_tracks[id]) >= UPDATE_INTERVAL:
                x_prev, y_prev, f_prev = vehicle_tracks[id][-UPDATE_INTERVAL]
                distance_px = math.hypot(cx - x_prev, cy - y_prev)
                time_elapsed = (frame_idx - f_prev) / fps
                speed_px_per_s = distance_px / time_elapsed if time_elapsed > 0 else 0
                speed_kmh = speed_px_per_s * 0.1195259 
                vehicle_speeds[id] = speed_kmh
                vehicle_speed_list.append(speed_kmh)
                last_update_frame[id] = frame_idx

            # Annotate
            speed_display = vehicle_speeds.get(id, 0)
            label = f"{speed_display:.1f} km/h"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 0, 0), 1)

    total_vehicle_count += len(current_boxes)
    density = compute_density(current_boxes, frame_area)
    congestion = min(density / max_capacity, 1.0) * 100 if max_capacity else 0
    congestion_scores.append(congestion)
    frames_data.append((frame.copy(), len(current_boxes)))
    avg_centroid = tuple(np.mean(current_centroids, axis=0)) if current_centroids else None
    centroid_path.append(avg_centroid)

    frame_idx += 1

cap.release()

# --- Summary Stats ---
flow_dir = estimate_direction(centroid_path)
avg_cong = np.mean(congestion_scores)
max_cong = np.max(congestion_scores)
min_cong = np.min(congestion_scores)
max_cong_frame = np.argmax(congestion_scores)
min_cong_frame = np.argmin(congestion_scores)
avg_speed = np.mean(vehicle_speed_list) if vehicle_speed_list else 0

# --- Annotate Output ---
for idx, (frame, num_vehicles) in enumerate(frames_data):
    # Determine congestion level
    if avg_cong > 80:
        congestion_level = "Severe Congestion"
        color = (0, 0, 255)  
    elif avg_cong > 55:
        congestion_level = "High Congestion"
        color = (0, 165, 255)  
    elif avg_cong > 25:
        congestion_level = "Moderate Congestion"
        color = (0, 255, 255)  
    else:
        congestion_level = "Low Congestion"
        color = (0, 255, 0)  

    # Draw annotations
    cv2.putText(frame, f"Vehicles: {num_vehicles}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
    cv2.putText(frame, f"Avg Congestion: {avg_cong:.2f}%", (10, h - 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, f"Congestion Level: {congestion_level}", (10, h - 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, f"Flow Dir: {flow_dir}", (10, h - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, f"Max Congestion: {(max_cong_frame / fps):.2f}s", (10, h - 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, f"Min Congestion: {(min_cong_frame / fps):.2f}s", (10, h - 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    out.write(frame)
out.release()

# --- Final Print ---
print(f"\n Avg Congestion: {avg_cong:.2f}%")
print(f" Max Congestion: {max_cong:.2f}% at {max_cong_frame/30} second mark")
print(f" Min Congestion: {min_cong:.2f}% at frame {min_cong_frame/30} second mark")
print(f" Flow Direction: {flow_dir}")
print(f" Total Vehicles Detected: {total_vehicle_count}")
print(f" Avg Vehicle Speed: {avg_speed:.2f} km/h")
print(f"\n Annotated video saved to: {OUTPUT_VIDEO_PATH}")
