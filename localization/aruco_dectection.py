import os
import cv2

# Open the default camera, use id=0 for the default one
cam_id = 0
cap = cv2.VideoCapture(cam_id)
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
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()


img_num = 0
while True:
    # Read the frame from the camera, if frame is read correctly ret is True
    ret, frame = cap.read()
    # if reading the frame was not successful
    if not ret:
        print("Can't receive frame. Exiting ...")
        break

    # Detect the markers in the frame
    corners, ids, rejected = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
    # Draw the markers on the frame
    frame_with_markers = cv2.aruco.drawDetectedMarkers(frame, corners, ids)

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

