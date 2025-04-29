import os
import cv2
import numpy as np

file_dir = os.path.dirname(os.path.realpath(__file__))
# Load images
img_mode0 = cv2.imread(file_dir + "\\test_images\\" + "mode0.png")
height_mode0, width_mode0, _ = img_mode0.shape
print("Mode 0 shape: ", img_mode0.shape) # should be 3280 x 2464
img_mode4 = cv2.imread(file_dir + "\\test_images\\" + "mode4.png")
height_mode4, width_mode4, _ = img_mode4.shape 
print("Mode 4 shape: ", img_mode4.shape) # should be 1280 x 720

# Initialize ArUco detector
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()

# Detect tags in Mode 0
corners_mode0, ids_mode0, _ = cv2.aruco.detectMarkers(img_mode0, aruco_dict, parameters=parameters)
# Detect tags in Mode 4
corners_mode4, ids_mode4, _ = cv2.aruco.detectMarkers(img_mode4, aruco_dict, parameters=parameters)
print("ids in Mode 0: ", ids_mode0) # 11, 12, 13, 3, 4, 10
print("ids in Mode 4: ", ids_mode4) # 11, 12, 13

def get_tag_center(corners):
    """Return the center of the tag (x, y)"""
    return np.mean(corners[0], axis=0)

# get centers of tags
centers_mode0 = {tag_id: get_tag_center(corners) for tag_id, corners in zip(ids_mode0.flatten(), corners_mode0)}
centers_mode4 = {tag_id: get_tag_center(corners) for tag_id, corners in zip(ids_mode4.flatten(), corners_mode4)}
print("centers in Mode 0: ", centers_mode0)
print("centers in Mode 4: ", centers_mode4)

# Calculate scale factor
# Construct matrix A * scale_x = b
A_x = []
b_x = []
for i in [11, 12, 13]:
    for j in [11, 12, 13]:
        if i != j:
            A_x.append(centers_mode0[i][0] - centers_mode0[j][0])  # x difference of Mode 0
            b_x.append(centers_mode4[i][0] - centers_mode4[j][0])  # x difference of Mode 4
A_x = np.array(A_x).reshape(-1, 1)
b_x = np.array(b_x)
scale_x = np.linalg.lstsq(A_x, b_x, rcond=None)[0][0]  # Least square method

# Construct matrix A * scale_y = b
A_y = []
b_y = []
for i in [11, 12, 13]:
    for j in [11, 12, 13]:
        if i != j:
            A_y.append(centers_mode0[i][1] - centers_mode0[j][1])  # y difference of Mode 0
            b_y.append(centers_mode4[i][1] - centers_mode4[j][1])  # y difference of Mode 4
A_y = np.array(A_y).reshape(-1, 1)
b_y = np.array(b_y)
scale_y = np.linalg.lstsq(A_y, b_y, rcond=None)[0][0]

print("scale_x: ", scale_x) # 0.50049686
print("scale_y: ", scale_y) # 0.5006737

# get expected center from Mode 4 to Mode 0
expected_x_mode0 = centers_mode4[11][0] / scale_x
expected_y_mode0 = centers_mode4[11][1] / scale_y

# get offset
offset_x = centers_mode0[11][0] - expected_x_mode0
offset_y = centers_mode0[11][1] - expected_y_mode0

print(f"Crop offset (X, Y): {offset_x:.1f}, {offset_y:.1f}") # 361.1, 513.5

# Calculate crop size
crop_width = 1280 / scale_x    # Mode 4 width / scale_x
crop_height = 720 / scale_y    # Mode 4 height / scale_y
print(f"Theoretical crop size: {crop_width:.1f}x{crop_height:.1f}") # 2557.5x1438.1

# check
cv2.rectangle(
    img_mode0,
    (int(offset_x), int(offset_y)),
    (int(offset_x + crop_width), int(offset_y + crop_height)),
    (0, 255, 0), 2  # Green Box
)
cv2.imwrite("crop_region_mode0.png", img_mode0)