import cv2

# Replace '/dev/video0' with the correct video device if it's different (use 'v4l2-ctl --list-devices' to check)
sensor_id = 0

# Open the camera
# cap = cv2.VideoCapture(camera_device, cv2.CAP_V4L2) # V4L2 can only support 8bit bayer, DONT USE!!! https://forums.developer.nvidia.com/t/green-screen-issue-on-nvidia-jetson-nano-and-raspberry-pi-v2-1-camera/220945/7
# cap = cv2.VideoCapture('nvarguscamerasrc ! video/x-raw(memory:NVMM), width=3280, height=2464, format=(string)NV12, framerate=(fraction)21/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink', cv2.CAP_GSTREAMER)
cap = cv2.VideoCapture('nvarguscamerasrc sensor-id={} ! video/x-raw(memory:NVMM), width=3280, height=2464, format=(string)NV12, framerate=(fraction)21/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink'.format(sensor_id), cv2.CAP_GSTREAMER)
# cap = cv2.VideoCapture(0, cv2.CAP_FFMPEG)

# Check if the camera is opened correctly
if not cap.isOpened():
    print("Error: Camera could not be opened.")
    exit()

# # Set camera resolution (optional, you can change it based on your camera capabilities)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH,300)  # Set frame width
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT,200)  # Set frame height

print("Streaming... Press 'q' to quit")

img_num = 20
while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture image.")
        break

    frame_re = cv2.resize(frame,(1920, 1080))
    # Display the frame
    cv2.imshow('Camera Stream', frame)

    key = cv2.waitKey(1)
    if key%256 == 32:
        # SPACE is pressed: save images 
        img_num += 1
        img_file_name = "test_img_{}.png".format(img_num)
        cv2.imwrite(img_file_name, frame) 
    if key & 0xFF == ord('q'):
        # 'q' key is pressed: exit the loop
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
