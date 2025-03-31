"""
==========================================
Fisheye Camera Calibration & Cartesian Mapping
==========================================
Device: Arducam MIPI CSI (Jetson Orion Nano)
Goal: Calibrate fisheye camera using a checkerboard and map the image to Cartesian coordinates.

Pseudocode Outline:
1. **Initialize Camera**:
   - Open Arducam MIPI CSI using OpenCV.

2. **Collect Calibration Images**:
   - Detect a checkerboard pattern in multiple images.
   - Store the detected corner points.

3. **Calibrate the Camera**:
   - Compute intrinsic matrix (K) and distortion coefficients (D) from collected images.

4. **Undistort the Fisheye Image**:
   - Use `cv2.fisheye.undistortImage()` to remove fisheye distortion.

5. **Convert to Cartesian Coordinates**:
   - Map fisheye pixels to Cartesian coordinates using a radial transformation.

6. **Live Processing**:
   - Continuously capture and undistort frames.
   - Display the undistorted image and apply Cartesian mapping.

Press 'q' to exit.
"""


import cv2
import numpy as np
import glob

# Checkerboard parameters
CHECKERBOARD = (6, 9)  # (rows, columns) in your pattern
SQUARE_SIZE = 0.025  # Set the square size in meters

# Capture frames from Arducam
def initialize_camera():
    cap = cv2.VideoCapture(0)  # Use /dev/video0 for MIPI CSI on Jetson
    if not cap.isOpened():
        raise Exception("Failed to open Arducam MIPI CSI camera")
    return cap

# Detect checkerboard and collect calibration images
def collect_calibration_images(cap, num_images=20):
    obj_points = []  # 3D points in real-world space
    img_points = []  # 2D points in image plane

    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[1], 0:CHECKERBOARD[0]].T.reshape(-1, 2) * SQUARE_SIZE

    collected = 0
    while collected < num_images:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

        if ret:
            img_points.append(corners)
            obj_points.append(objp)
            collected += 1
            print(f"Collected {collected}/{num_images} calibration images")

        cv2.imshow("Checkerboard", frame)
        if cv2.waitKey(500) & 0xFF == ord('q'):  # Press 'q' to exit early
            break

    cap.release()
    cv2.destroyAllWindows()
    return obj_points, img_points, frame.shape[:2]

# Compute camera matrix and distortion coefficients
def calibrate_camera(obj_points, img_points, image_shape):
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))

    ret, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
        obj_points, img_points, image_shape[::-1], K, D,
        flags=cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
    )

    if not ret:
        raise Exception("Camera calibration failed")
    
    print("Camera Matrix:\n", K)
    print("Distortion Coefficients:\n", D)
    
    return K, D

# Undistort fisheye image
def undistort_fisheye(frame, K, D):
    h, w = frame.shape[:2]
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (w, h), np.eye(3), balance=0)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2)
    undistorted = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted

# Convert fisheye to Cartesian coordinates
def fisheye_to_cartesian(undistorted_frame):
    h, w = undistorted_frame.shape[:2]
    cartesian_map = np.zeros((h, w, 2), dtype=np.float32)

    for y in range(h):
        for x in range(w):
            r = np.sqrt((x - w / 2) ** 2 + (y - h / 2) ** 2)  # Radius from center
            theta = np.arctan2(y - h / 2, x - w / 2)  # Angle in polar coordinates
            
            x_cart = r * np.cos(theta)
            y_cart = r * np.sin(theta)

            cartesian_map[y, x] = [x_cart, y_cart]

    return cartesian_map

# Main function
def main():
    cap = initialize_camera()

    print("Collecting calibration images...")
    obj_points, img_points, image_shape = collect_calibration_images(cap)
    
    print("Calibrating camera...")
    K, D = calibrate_camera(obj_points, img_points, image_shape)

    cap = initialize_camera()
    while True:
        frame = capture_frame(cap)
        undistorted_frame = undistort_fisheye(frame, K, D)
        cartesian_map = fisheye_to_cartesian(undistorted_frame)
        
        cv2.imshow("Undistorted Frame", undistorted_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run script
if __name__ == "__main__":
    main()
