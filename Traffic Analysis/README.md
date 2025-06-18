# Traffic Congestion & Speed Analysis

This project implements a pipeline for traffic video analysis using the YOLOv8 object detection model. It performs vehicle detection, tracking, speed estimation, congestion level computation, direction analysis, and visual annotation—integrated into a single workflow.

---

## Features

* **Vehicle Detection** using a custom-trained YOLOv8 model
* **Object Tracking** via ByteTrack algorithm
* **Real-time Speed Estimation** in km/h
* **Congestion Estimation** using dynamic frame-wise density calculations
* **Traffic Flow Direction Analysis** based on centroid angle shifts
* **Pixel to Real-World Conversion** tool using manual measurement
* **Video Annotation and Export** with congestion level, direction, speed, and vehicle counts

---



## Folder Structure

```
traffic_test_data/
├── traffic.mp4               # Input traffic video
├── frame.jpg                 # Extracted frame for pixel calibration
best.pt                       # Trained YOLOv8 model weights
annotated_output_final.mp4    # Output annotated video
convert.py                    # To Calculate Speed conversion factor
flow.py                       # Main File with Pipeline that outputs annotated video
```

---

## Setup and Use

Before running analysis, run the calibration tool (`convert.py`) to get the scale:

```bash
python convert.py
```

Click two points with a known distance between them. The script will calculate how many meters one pixel represents.

Run `flow.py` after updating input video

```bash
python flow.py
```

---

## How It Works

1. **Pixel Calibration Tool (`convert.py`)**

   * Extracts a frame from the video
   * User clicks on two points of known real-world distance (e.g., lane width)
   * Computes a meters-per-pixel ratio for accurate speed estimation

2. **Detection & Tracking**

   * Vehicles are detected using YOLOv8
   * Tracked using ByteTrack, assigning persistent IDs across frames

3. **Speed Estimation**

   * Based on pixel displacement over time using centroid tracking
   * Converts pixel/s to km/h using calibrated real-world scale

4. **Congestion Estimation**

   * Computes bounding box area as a percentage of frame area
   * Uses a dynamic "maximum capacity"

5. **Flow Direction Analysis**

   * Estimates vector angles between centroids frame-to-frame
   * Aggregates direction using statistical mode to classify the dominant flow

6. **Annotation & Export**

   * Each frame is annotated with:

     * Vehicle count
     * Average congestion percentage
     * Congestion level (Low, Moderate, High, Severe)
     * Flow direction
     * Individual vehicle speed
   * Annotated video is exported as `annotated_output_final.mp4`

---

## Congestion Thresholds

* > > 80% → Severe Congestion
* > > 55% → High Congestion
* > > 25% → Moderate Congestion
* > ≤ 25% → Low Congestion

---

## Notes

* Works best with fixed-angle traffic surveillance videos
* Update `KNOWN_REAL_DISTANCE_METERS` according to your video
* Customize the YOLO model and `VEHICLE_CLASSES` if using a different dataset

---

## Future Enhancements

* Real-time processing via webcam or CCTV stream
* Integration with more advanced trackers like DeepSORT or StrongSORT
* Fully Automated Scaling Factor

---

Let me know if you'd like a LaTeX version, GitHub-compatible badges, or to split this into sections like `docs/`, boss.
