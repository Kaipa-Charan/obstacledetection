import cv2
import math
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter

def define_trapezoidal_roi(frame, margin=50):
    height, width = frame.shape[:2]
    bottom_width = width - 2 * margin
    top_width = width // 2
    top_height = height // 3
    
    vertices = np.array([[
        (margin, height),
        (width - margin, height),
        (width - margin - bottom_width // 2, height - top_height),
        (margin + bottom_width // 2, height - top_height)
    ]], dtype=np.int32)

    return vertices

def is_object_in_trapezoidal_roi(box, roi_vertices):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    points = [(x1, y1), (x2, y1), (x1, y2), (x2, y2), (cx, cy)]

    for point in points:
        if cv2.pointPolygonTest(roi_vertices, point, False) >= 0:
            return True
    return False

def initialize_realsense():
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    pipeline = rs.pipeline()
    pipeline.start(config)

    align_to = rs.stream.color
    align = rs.align(align_to)

    return pipeline, align

# Initialize YOLO model
model = YOLO("/home/raillabs/labelImg/ultralytics/runs/detect/train7/weights/best.pt")
model.to('cuda')  # Use 'cuda' for GPU, 'cpu' for CPU

# Class names for YOLO
classNames = ["Human Obstacle", "Animal Obstacle", "Obstacle"]

# Define ROI vertices once outside the loop
roi_vertices = define_trapezoidal_roi(np.zeros((480, 640, 3), dtype=np.uint8))

# Initialize Kalman filter
kalman = KalmanFilter(dim_x=4, dim_z=2)  # State vector: [x, y, dx/dt, dy/dt]
kalman.transition_matrices = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
kalman.measurement_matrices = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
kalman.process_noise_covariance = 0.03 * np.eye(4, dtype=np.float32)
kalman.measurement_noise_covariance = 1e-1 * np.eye(2, dtype=np.float32)

# Initialize flags
flag1 = 1  # No objects detected
flag2 = 2  # Object detected outside trapezoidal ROI
flag3 = 3  # Object detected inside trapezoidal ROI

# Background subtraction
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

# Initialize RealSense
pipeline, align = initialize_realsense()

while True:
    # Capture frames from RealSense
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()

    if not color_frame or not depth_frame:
        continue

    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.08), cv2.COLORMAP_JET)

    # Apply background subtraction
    fgmask = bg_subtractor.apply(color_image)

    # Apply Gaussian blur to reduce noise
    img_blur = cv2.GaussianBlur(color_image, (5, 5), 0)

    # Perform object detection
    results = model.predict(img_blur, stream=True, conf=0.75)

    # Reset flags for each frame
    flag1 = 1
    flag2 = 2
    flag3 = 3

    for i, r in enumerate(results):
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Kalman filter prediction and correction
            kalman.predict()
            kalman.update(np.array([cx, cy], dtype=np.float32))
            pred_cx, pred_cy = kalman.x[:2]

            # Determine if the object is inside the ROI
            if is_object_in_trapezoidal_roi(box, roi_vertices):
                color = (0, 0, 255)  # Red color for inside ROI
                flag2 = 0  # Reset flag2 since an object is inside ROI
                flag3 = 3  # Set flag3 since object is inside ROI
            else:
                color = (255, 0, 0)  # Blue color for outside ROI
                flag3 = 0  # Reset flag3 since an object is outside ROI
                flag2 = 2  # Set flag2 since object is outside ROI

            flag1 = 0  # Set flag1 since an object is detected

            # Draw bounding box on color image
            cv2.rectangle(color_image, (x1, y1), (x2, y2), color, 3)

            # Draw center point
            cv2.circle(color_image, (cx, cy), 5, (0, 255, 0), -1)

            # Calculate distance to the object
            depth_value = depth_image[cy, cx]  # Depth value at the center of the bounding box
            distance = depth_value * 0.001  # Convert to meters (depth values are in millimeters)

            # Display class name, confidence, and distance on color image
            confidence = math.ceil(box.conf[0] * 100) / 100
            cls = int(box.cls[0])
            label = f"{classNames[cls]}: {confidence} - Dist: {distance:.2f}m"
            cv2.putText(color_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Display distance on depth colormap
            cv2.putText(depth_colormap, f"Dist: {distance:.2f}m", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Draw the ROI boundary for visualization
    cv2.polylines(color_image, [roi_vertices], isClosed=True, color=(0, 255, 0), thickness=2)

    # Display the frames
    cv2.imshow('Color Image with ROI', color_image)
    cv2.imshow('Depth Image', depth_colormap)

    # Print corresponding outputs based on flags
    if flag1 == 1:
        print("1")
    elif flag2 == 2:
        print("2")
    elif flag3 == 3:
        print("3")

    # Exit when 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release resources
pipeline.stop()
cv2.destroyAllWindows()

