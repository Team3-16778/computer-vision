import os
import cv2
import cv2.aruco as aruco
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))

# Open camera
# cap = cv2.VideoCapture(0) # for WebCam
CSI_camera_params = 'nvarguscamerasrc ! video/x-raw(memory:NVMM), width=3280, height=2464, format=(string)NV12, framerate=(fraction)21/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink'
cap = cv2.VideoCapture(CSI_camera_params, cv2.CAP_GSTREAMER)

# Get camera internal and external parameters from exist file
camera_internal_file = dir_path + "/camera2_calibration_data.npz"
internal_data = np.load(camera_internal_file)
camera_matrix = internal_data["camera_matrix"]
dist_coeffs = internal_data["dist_coeffs"]

# Check if camera opened successfully
if not cap.isOpened():
    print("Could not open video device")
    exit()

# Find the current working directory
dir_path = os.path.dirname(os.path.realpath(__file__))
# Create a directory for test images if it doesn't exist
test_dir = dir_path + '/test_images'
if not os.path.exists(test_dir):
    os.makedirs(test_dir)


# Define the ArUco dictionary
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()


img_num = 0
while True:
    # Read the frame from the camera, if frame is read correctly ret is True
    ret, frame = cap.read()
    # if reading the frame was not successful
    if not ret:
        print("Can't receive frame. Exiting ...")
        break
    #frame_re = cv2.resize(frame,(800, 600))
    frame_re = frame

    # Detect the markers in the frame
    corners, ids, rejected = aruco.detectMarkers(frame_re, aruco_dict, parameters=parameters)

    if ids is not None:
        # Estimate pose of detected markers
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 0.023, camera_matrix, dist_coeffs) # 0.1 is the side length of the marker in meters

        # Draw frame axes for each marker
        for i in range(len(ids)):
            cv2.drawFrameAxes(frame_re, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.05) # 0.05 is the length of the axes in meters

    # Draw detected markers
    aruco.drawDetectedMarkers(frame_re, corners, ids)
    # Draw the markers on the frame
    frame_with_markers = aruco.drawDetectedMarkers(frame_re, corners, ids)

    # Display the frame with the markers
    cv2.imshow("Camera", frame_with_markers)

    # Press 'q' to exit
    key = cv2.waitKey(1)
    if key%256 == 32:
        # SPACE is pressed: save test images 
        img_num += 1
        img_file_name = test_dir + '/test_' + str(img_num) + '.png'
        cv2.imwrite(img_file_name, frame_with_markers) 
    if key & 0xFF == ord('q'):
        # 'q' key is pressed: exit the loop
        break

# Release the VideoCapture object
cap.release()
cv2.destroyAllWindows()

