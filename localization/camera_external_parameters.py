import os
import cv2
import numpy as np
import cv2.aruco as aruco
from scipy.spatial.transform import Rotation as R

# Find the current working directory
dir_path = os.path.dirname(os.path.realpath(__file__))

# Least square method for calculating the camera pose
def cal_cam_pose_lsq(camera_poses):
    """
    Input:
      camera_poses: a list, each element is a 4x4 matrix (Camera Pose Matrix at world coordinate from every marker)。
    Output:
      T_avg: fusion of all camera poses。
    """
    N = len(camera_poses)
    if N == 0:
        raise ValueError("There are no valid camera poses provided.")
    
    # Separate translations and rotations
    translations = np.array([T[0:3, 3] for T in camera_poses])
    rotations = np.array([T[0:3, 0:3] for T in camera_poses])
    
    # Translations: Mean
    t_avg = np.mean(translations, axis=0)
    
    # Rotations: Quaternion
    r = R.from_matrix(rotations)
    
    # calculate average quaternion (Note: Scipy > 1.8 required !!!)
    r_mean = r.mean()
    R_avg = r_mean.as_matrix()
    
    # Construct the final 4x4 transformation matrix
    T_avg = np.eye(4)
    T_avg[0:3, 0:3] = R_avg
    T_avg[0:3, 3] = t_avg
    
    return T_avg

# 1. get camera internal parameters from exist file'
camera_calibration_file = dir_path + "/camera1_calibration_data.npz"
data = np.load(camera_calibration_file)
camera_matrix = data["camera_matrix"]
dist_coeffs = data["dist_coeffs"]
print("Camera Matrix: \n", camera_matrix)
print("Distortion Coefficients: \n", dist_coeffs)

# 2. Define the ArUco dictionary
aruco_dict = aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()


# 3. set the world coordinates of the markers
# Assuming we are using markers with IDs 1 and 2
# Define the world transformation matrix (4x4) for each marker
marker_world_poses = {
    1: np.array([[1, 0, 0, 0.5],    # marker 1 is located at (0.5, 0, 0) meters
                 [0, 1, 0, 0.0],
                 [0, 0, 1, 0.0],
                 [0, 0, 0, 1]]),
    2: np.array([[1, 0, 0, -0.5],   # marker 2 is located at (-0.5, 0, 0) meters
                 [0, 1, 0, 0.0],
                 [0, 0, 1, 0.0],
                 [0, 0, 0, 1]])
}

# 4. Read the image and detect ArUco markers
img_file = dir_path + "/test_camera_0.png"
img = cv2.imread(img_file)  # change the path to your image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
corners, ids, _ = aruco.detectMarkers(img, aruco_dict, parameters=parameters)
if ids is None:
    print("No ArUco markers detected in the image.")
    exit()
else:
    print("ArUco markers' ID detected in the image: ", ' '.join(str(item) for row in ids for item in row))

img_with_markers = aruco.drawDetectedMarkers(img, corners, ids)


# 5. Estimate the poses of the markers, and compute the world transformation matrix of camera for each marker
marker_length = 0.023  # length of each ArUco marker in meters
camera_poses = []

for i, marker_id in enumerate(ids.flatten()):
    # Estimate the pose of the marker(Rotation vector and Translation vector)
    rvec, tvec, _ = aruco.estimatePoseSingleMarkers([corners[i]], marker_length, camera_matrix, dist_coeffs)
    
    # change the rotation vector to rotation matrix
    R_ct, _ = cv2.Rodrigues(rvec[0])
    T_ct = tvec[0].reshape(3, 1)
    
    # Construct the 4x4 transformation matrix T_camera_marker（from marker frame to camera frame）
    T_camera_marker = np.eye(4)
    T_camera_marker[0:3, 0:3] = R_ct
    T_camera_marker[0:3, 3] = T_ct.flatten()
    
    # Compute the inverse，obtain the transformation T_marker_camera（from camera frame to marker frame）
    T_marker_camera = np.linalg.inv(T_camera_marker)
    
    # Get the world transformation matrix T_world_marker of the marker
    if marker_id in marker_world_poses:
        T_world_marker = marker_world_poses[marker_id]     
        # Compute the world transformation matrix T_world_camera
        # T_world_camera = T_world_marker * T_marker_camera
        T_world_camera = np.dot(T_world_marker, T_marker_camera)
        camera_poses.append(T_world_camera)        
        print(f"T_world_camera from Marker ID {marker_id}:\n", T_world_camera)
    else:
        print(f"Marker ID {marker_id} not found in marker_world_poses.")


# 6. Calculate the camera pose(external parameters) 
if camera_poses:
    # # Method 1: Calculate the average of the camera poses
    # T_avg = np.mean(np.array(camera_poses), axis=0)
    # Method 2: Least square method, using quaternion for rotation
    T_avg = cal_cam_pose_lsq(camera_poses)
    print("\nT_world_camera:\n", T_avg)
else:
    print("No valid camera poses found.")


# 7. Save the camera pose to a file
save_file = dir_path + "/camera_external_parameters.npz"
np.savez(save_file, T_world_camera=T_avg)

