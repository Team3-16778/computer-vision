import os
import cv2
import numpy as np

file_dir = os.path.dirname(os.path.realpath(__file__))
# Load images
img_mode0 = cv2.imread(file_dir + "\\test_images\\" + "mode0.png")
img_mode4 = cv2.imread(file_dir + "\\test_images\\" + "mode4.png")

# Initialize ArUco detector
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()

# Detect tags in Mode 0
corners_mode0, ids_mode0, _ = cv2.aruco.detectMarkers(img_mode0, aruco_dict, parameters=parameters)
# Detect tags in Mode 4
corners_mode4, ids_mode4, _ = cv2.aruco.detectMarkers(img_mode4, aruco_dict, parameters=parameters)
print("ids in Mode 0: ", ids_mode0)
print("ids in Mode 4: ", ids_mode4)

def get_tag_center(corners):
    """Return the center (x, y) of an ArUco tag."""
    return np.mean(corners[0], axis=0)

# Get centers of Tag 0 in both modes
tag0_center_mode0 = get_tag_center(corners_mode0[ids_mode0 == 0][0])
tag0_center_mode4 = get_tag_center(corners_mode4[ids_mode4 == 0][0])

# Get centers of Tag 1 in both modes
tag1_center_mode0 = get_tag_center(corners_mode0[ids_mode0 == 1][0])
tag1_center_mode4 = get_tag_center(corners_mode4[ids_mode4 == 1][0])