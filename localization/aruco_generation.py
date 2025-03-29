import os
import cv2

# Find the current working directory
dir_path = os.path.dirname(os.path.realpath(__file__))

# Create the directory if it doesn't exist
acuco_dir = dir_path + '/aruco_images'
if not os.path.exists(acuco_dir):
    os.makedirs(acuco_dir)

# Define the dictionary we want to use
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

# Generate multiple marker, ids 0 to 4 and 10 to 14
marker_size = 100 # Size in pixels, width around 23mm on American Letter
for marker_id in [*range(0, 5), *range(10, 15)]:
    marker_image = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)  # Generate the image
    cv2.imwrite(acuco_dir + '/aruco_' + str(marker_id) + '.png', marker_image)



