import math
import cv2

model = YOLO("yolov8m.yaml") 

cap = cv2.VideoCapture("traffic_test_data/traffic.mp4")
ret, frame = cap.read()
cv2.imwrite("traffic_test_data/frame.jpg", frame)
cap.release()

# --- CONFIG ---
KNOWN_REAL_DISTANCE_METERS = 3.5  
IMAGE_PATH = "traffic_test_data/frame.jpg"

# --- Load Image ---
img = cv2.imread(IMAGE_PATH)
clone = img.copy()
points = []

# --- Mouse Click Handler ---
def click_event(event, x, y, flags, param):
    global points, img
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f" Point {len(points)}: ({x}, {y})")
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)

        if len(points) == 2:
            x1, y1 = points[0]
            x2, y2 = points[1]
            pixel_distance = math.hypot(x2 - x1, y2 - y1)
            meters_per_pixel = KNOWN_REAL_DISTANCE_METERS / pixel_distance
            print(f"\n Pixel Distance: {pixel_distance:.2f} px")
            print(f" 1 pixel = {meters_per_pixel:.6f} meters")
            print(f" You can click again to remeasure.\n")

            img = clone.copy()
            cv2.line(img, points[0], points[1], (255, 0, 0), 2)
            cv2.circle(img, points[0], 5, (0, 0, 255), -1)
            cv2.circle(img, points[1], 5, (0, 0, 255), -1)
            points = []

        cv2.imshow("Lane Marking Measurement", img)

# --- Run Tool ---

cv2.imshow("Lane Marking Measurement", img)
cv2.setMouseCallback("Lane Marking Measurement", click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()
