import os
import cv2
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))

# Camera Stream
w_original, h_original, framerate_original = 3280, 2464, 21
w_new, h_new, framerate_new = 1280, 720, 60

# cap = cv2.VideoCapture(0)  # Open the default camera
CSI_camera_params = f'nvarguscamerasrc sensor-id=1 ! video/x-raw(memory:NVMM), width={w_new}, height={h_new}, format=(string)NV12, framerate=(fraction){framerate_new}/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink'
# CSI_camera_params = f'nvarguscamerasrc sensor-id=1 ! video/x-raw(memory:NVMM), width={w_original}, height={h_original}, format=(string)NV12, framerate=(fraction){framerate_original}/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink'

cap = cv2.VideoCapture(CSI_camera_params, cv2.CAP_GSTREAMER)


# 1. Get camera internal and external parameters from exist file
camera_internal_file = dir_path + "/camera2_calibration_data.npz"
internal_data = np.load(camera_internal_file)
camera_matrix = internal_data["camera_matrix"]
dist_coeffs = internal_data["dist_coeffs"]
print("Camera Matrix: \n", camera_matrix)
print("Distortion Coefficients: \n", dist_coeffs)

camera_external_file = dir_path + "/cam2_external_parameters_0424.npz"
T_world_camera = np.load(camera_external_file)["T_world_camera"]
print("T_world_camera:\n", T_world_camera)

# Parameters from transfering function(calculate_mode_transfer.py)
scale_x = 0.50049686
scale_y = 0.5006737
offset_x = 361.1
offset_y = 513.5

def calculate_world_3D(u, v, Zc = 0.6731): # Zc is the distance from the camera to the target(tag-cam distance get from T_world_camera 0.69846859)
    """
    Calculate the 3D coordinates of the target in the world frame
    Input:
        u: x coordinate of the target in the image
        v: y coordinate of the target in the image
        Zc: depth of the target in the camera frame (distance from the camera to the target)
    Output:
        world_point: 3D coordinates of the target in the world frame
    """
    # 1. Convert Mode 4 (u, v) to Mode 0 coordinates
    u_mode0 = u / scale_x + offset_x
    v_mode0 = v / scale_y + offset_y

    # 2. Get the parameters we need: focal length, principal point, and rotation vector and translation vector
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    Rotation_matrix = T_world_camera[0:3, 0:3]
    Translation_vector = T_world_camera[0:3, 3].reshape(3, 1)

    # 4. Compute the 3D coordinates of the target in the camera frame
    Xc = (u_mode0 - cx) * Zc / fx
    Yc = (v_mode0 - cy) * Zc / fy
    camera_point = np.array([[Xc], [Yc], [Zc]])

    # 5. Compute the 3D coordinates of the target in the world frame
    # Method 1: external parameters is T_world_camera
    world_point = Rotation_matrix @ camera_point + Translation_vector

    # Method 2: external parameters is T_camera_world
    # world_point = Rotation_matrix.T @ (camera_point - Translation_vector)

    return world_point

points_with_labels = []
def mouse_callback(event, u, v, flags, param):
    global points_with_labels
    if event == cv2.EVENT_LBUTTONDOWN:
        # print the pixel coordinates u, v
        print("Pixel coordinates: ({}, {})".format(u, v))
        
        # Get the 3D coordinates of the target in the world frame
        P_world = calculate_world_3D(u, v)
        P_world = P_world.flatten()

        # Create a label for the point
        label = "({:.2f}, {:.2f}, {:.2f})".format(P_world[0], P_world[1], P_world[2])
        
        # Store the point and label in the list
        points_with_labels.append((u, v, label))


cv2.namedWindow("Camera")
cv2.setMouseCallback("Camera", mouse_callback)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # frame_re = cv2.resize(frame,(1920, 1080)) # resize the frame image
    frame_re = frame
 
    # Undistort the frame if needed
    # frame_re = cv2.undistort(frame, camera_matrix, dist_coeffs)
    
    # Draw all the points with a circle and a label on the image
    for (x, y, label) in points_with_labels:
        # Print a circle
        cv2.circle(frame_re, (x, y), 5, (0, 255, 0), -1)
        # Add a label
        cv2.putText(frame_re, label, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0, 255, 0), 2)
    
    cv2.imshow("Camera", frame_re)
    key = cv2.waitKey(1)
    if key == 27:  # Esc key to exit
        break

cap.release()
cv2.destroyAllWindows()