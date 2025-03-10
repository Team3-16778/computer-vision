"""
Camera calibration implementation using OpenCV
Mechatronic Design, Michael (Chase) Allen

This script is based on the following OpenCV tutorial:
https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
"""



import cv2
import numpy as np
import glob
import warnings

"""
Find checkerboard
"""

# Define checkerboard size
CHECKERBOARD = (9, 6)  # Adjust based on your checkerboard
square_size = 1.0  # Set appropriately if needed

# Prepare object points (0,0,0), (1,0,0), ..., (8,5,0) assuming z=0 (planar)
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size  # Scale if square size is known

# Storage for object points & image points
objpoints = []  # 3D points in real-world space
imgpoints = []  # 2D points in image plane

# Load images
images = glob.glob("checkerboard_images/*.jpg")

for fname in images:
    print(f"Processing {fname}")
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find the checkerboard corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)
        
        # Draw and show corners
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners, ret)
        cv2.imshow('Checkerboard', img)
        cv2.waitKey(1000)
    else:
        warnings.warn(f"Could not detect corners in {fname}", UserWarning)

cv2.destroyAllWindows()


"""    
Perform Camera Calibration
"""

# Get image size (all were taken with same camera)
h, w = gray.shape[:2]
print(f"\nDetected Image Size: {w}x{h}")

# Run camera calibration using objpoints & imgpoints arrays
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w, h), None, None)

# Print results
print("\nCamera Calibration Results:")
print("Camera Matrix:\n", mtx)  # Intrinsic parameters
print("Distortion Coefficients:\n", dist)  # Lens distortion
print("Rotation Vectors:\n", rvecs)  # Extrinsic rotation for each image
print("Translation Vectors:\n", tvecs)  # Extrinsic translation for each image

# Save calibration parameters
np.savez("camera_calibration_data.npz", camera_matrix=mtx, dist_coeffs=dist, rvecs=rvecs, tvecs=tvecs)

# Reprojection error (to check accuracy)
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error += error

print(f"\nMean Reprojection Error: {mean_error / len(objpoints)}")


"""
Undistort Images"
"""
for fname in images:
    img = cv2.imread(fname)
    
    # Get the optimal new camera matrix (better field of view)
    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # Undistort the image
    undistorted = cv2.undistort(img, mtx, dist, None, new_camera_mtx)

    # Crop and display (optional)
    # x, y, w, h = roi
    # undistorted = undistorted[y:y+h, x:x+w]
    
    cv2.imshow("Undistorted", undistorted)
    cv2.waitKey(1000)

cv2.destroyAllWindows()


"""
Reprojection For future Data 
"""

NotImplementedError("Reprojection for future data not implemented yet")