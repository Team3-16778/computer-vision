import cv2
import numpy as np
import glob

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
images = glob.glob("checkerboard_images/*.jpg")  # Adjust path

for fname in images:
    print(f"Processing {fname}")
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find the checkerboard corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)
        
        # Optional: Draw and show corners
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners, ret)
        cv2.imshow('Checkerboard', img)
        cv2.waitKey(1000)
    else:
        print(f"Could not detect corners in {fname}")

cv2.destroyAllWindows()
