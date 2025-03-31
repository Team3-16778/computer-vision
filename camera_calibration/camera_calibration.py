import cv2
import numpy as np
import glob
import warnings
import matplotlib.pyplot as plt

def find_checkerboard_corners(images, checkerboard):
    objp = np.zeros((checkerboard[0] * checkerboard[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)
    objpoints, imgpoints = [], []
    
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, checkerboard, None)
        
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
            cv2.drawChessboardCorners(img, checkerboard, corners, ret)
        else:
            warnings.warn(f"Could not detect corners in {fname}", UserWarning)
    
    return objpoints, imgpoints, gray.shape[:2]

def calibrate_camera(objpoints, imgpoints, image_size):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None, flags=cv2.CALIB_FIX_K3)
    np.savez("camera_calibration_data.npz", camera_matrix=mtx, dist_coeffs=dist, rvecs=rvecs, tvecs=tvecs)
    return mtx, dist

def undistort_images(images, mtx, dist, image_size):
    undistorted_images = []
    new_camera_mtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, image_size, 1, image_size)
    
    for fname in images:
        img = cv2.imread(fname)
        undistorted = cv2.undistort(img, mtx, dist, None, new_camera_mtx)
        undistorted_images.append(undistorted)
    
    return undistorted_images

def show_results(original, corners_img, undistorted):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    axes[1].imshow(cv2.cvtColor(corners_img, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Corners Detected")
    axes[1].axis("off")
    
    axes[2].imshow(cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Undistorted Image")
    axes[2].axis("off")
    
    plt.show()

def main():
    CHECKERBOARD = (9, 6)
    images = glob.glob("*.png")
    
    if not images:
        print("No images found!")
        return
    
    objpoints, imgpoints, image_size = find_checkerboard_corners(images, CHECKERBOARD)
    mtx, dist = calibrate_camera(objpoints, imgpoints, image_size)
    
    undistorted_images = undistort_images(images, mtx, dist, image_size)
    # save the undistorted images
    for img_num in range(len(undistorted_images)): 
        img_name = "test_img_{}_undistorted.jpg".format(img_num+1)
        cv2.imwrite(img_name, undistorted_images[img_num])
    
    original_img = cv2.imread(images[0])
    corners_img = original_img.copy()
    gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    if ret:
        cv2.drawChessboardCorners(corners_img, CHECKERBOARD, corners, ret)
    undistorted_img = undistorted_images[0]
    
    show_results(original_img, corners_img, undistorted_img)

if __name__ == "__main__":
    main()