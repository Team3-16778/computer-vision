# MIT License
# Copyright (c) 2019-2022 JetsonHacks

# A simple code snippet
# Using two  CSI cameras (such as the Raspberry Pi Version 2) connected to a
# NVIDIA Jetson Nano Developer Kit with two CSI ports (Jetson Nano, Jetson Xavier NX) via OpenCV
# Drivers for the camera and OpenCV are included in the base image in JetPack 4.3+

# This script will open a window and place the camera stream from each camera in a window
# arranged horizontally.
# The camera streams are each read in their own thread, as when done sequentially there
# is a noticeable lag

import cv2
import threading
import numpy as np
from ultralytics import YOLO
import torch



class CSI_Camera:

    def __init__(self):
        # Initialize instance variables
        # OpenCV video capture element
        self.video_capture = None
        # The last captured image from the camera
        self.frame = None
        self.grabbed = False
        # The thread where the video capture runs
        self.read_thread = None
        self.read_lock = threading.Lock()
        self.running = False

    def open(self, gstreamer_pipeline_string):
        try:
            self.video_capture = cv2.VideoCapture(
                gstreamer_pipeline_string, cv2.CAP_GSTREAMER
            )
            # Grab the first frame to start the video capturing
            self.grabbed, self.frame = self.video_capture.read()

        except RuntimeError:
            self.video_capture = None
            print("Unable to open camera")
            print("Pipeline: " + gstreamer_pipeline_string)


    def start(self):
        if self.running:
            print('Video capturing is already running')
            return None
        # create a thread to read the camera image
        if self.video_capture != None:
            self.running = True
            self.read_thread = threading.Thread(target=self.updateCamera)
            self.read_thread.start()
        return self

    def stop(self):
        self.running = False
        # Kill the thread
        self.read_thread.join()
        self.read_thread = None

    def updateCamera(self):
        # This is the thread to read images from the camera
        while self.running:
            try:
                grabbed, frame = self.video_capture.read()
                with self.read_lock:
                    self.grabbed = grabbed
                    self.frame = frame
            except RuntimeError:
                print("Could not read image from camera")
        # FIX ME - stop and cleanup thread
        # Something bad happened

    def read(self):
        with self.read_lock:
            frame = self.frame.copy()
            grabbed = self.grabbed
        return grabbed, frame

    def release(self):
        if self.video_capture != None:
            self.video_capture.release()
            self.video_capture = None
        # Now kill the thread
        if self.read_thread != None:
            self.read_thread.join()


""" 
gstreamer_pipeline returns a GStreamer pipeline for capturing from the CSI camera
Flip the image by setting the flip_method (most common values: 0 and 2)
display_width and display_height determine the size of each camera pane in the window on the screen
Default 1920x1080
"""


def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=1920,
    display_height=1080,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

def run_cameras():
    window_title = "Dual CSI Cameras with YOLOv8"
    # model = YOLO('yolov8n.pt').export(format='engine', device='cuda:0')
    
    # model = YOLO('yolov8n.pt').export(format='onnx')
    model = YOLO('yolov8n.onnx')
    
    # device = torch.device("cuda")
    # model = YOLO('yolov8n.pt').export(format='engine', device=device)

    # model = YOLO('yolov8n.pt')

    class_names = model.names

    # Initialize cameras
    left_camera = CSI_Camera()
    left_camera.open(gstreamer_pipeline(sensor_id=0, display_width=960, display_height=540))
    left_camera.start()

    right_camera = CSI_Camera()
    right_camera.open(gstreamer_pipeline(sensor_id=1, display_width=960, display_height=540))
    right_camera.start()

    try:
        while True:
            # Read frames from both cameras
            _, left_image = left_camera.read()
            _, right_image = right_camera.read()

            # Perform YOLO detection
            left_results = model(left_image, verbose=False)[0]
            right_results = model(right_image, verbose=False)[0]

            # Draw detections
            left_detected = draw_detections(left_image, left_results, class_names)
            right_detected = draw_detections(right_image, right_results, class_names)

            # Combine frames
            combined = np.hstack((left_detected, right_detected))
            
            # Display output
            cv2.imshow(window_title, combined)
            
            # Exit on ESC
            if cv2.waitKey(1) & 0xFF == 27:
                break
                
    finally:
        # Cleanup
        left_camera.stop()
        left_camera.release()
        right_camera.stop()
        right_camera.release()
        cv2.destroyAllWindows()



def colormask(image):

    # Convert BGR to RGB for proper display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  

    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define color range for masking (adjust as needed)
    lower_bound = np.array([120, 30, 30])  # Relaxed: Purple lower bound
    upper_bound = np.array([180, 255, 255])  # Relaxed: Purple upper bound

    # Create the mask (black and white)
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Compute average coordinates of the masked area
    coordinates = np.column_stack(np.where(mask > 0))  # Get all nonzero pixel locations
    if len(coordinates) > 0:
        avg_y, avg_x = np.mean(coordinates, axis=0).astype(int)  # Compute average coordinates
        print(f"Average object location: ({avg_x}, {avg_y})")
    else:
        avg_x, avg_y = -1, -1  # Default if nothing is detected

    # Apply the mask to the original image
    masked_image = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)

    # Convert mask to 3-channel for visualization
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

    # Draw a red marker at the average location
    if avg_x >= 0 and avg_y >= 0:
        cv2.drawMarker(mask_bgr, (avg_x, avg_y), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

    return image_rgb, masked_image, mask_bgr


def draw_detections(frame, results, class_names):
    # Extract detection results
    boxes = results.boxes.xyxy.cpu().numpy()
    scores = results.boxes.conf.cpu().numpy()
    class_ids = results.boxes.cls.cpu().numpy().astype(int)
    
    # Draw bounding boxes and labels
    for box, score, class_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = map(int, box)
        label = f"{class_names[class_id]} {score:.2f}"
        
        # Draw rectangle and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame


if __name__ == "__main__":
    run_cameras()
