import cv2

# Replace '/dev/video0' with the correct video device if it's different (use 'v4l2-ctl --list-devices' to check)
camera_device = '/dev/video0'

# Open the camera
# cap = cv2.VideoCapture(camera_device)
cap = cv2.VideoCapture(camera_device, cv2.CAP_V4L2)


# Check if the camera is opened correctly
if not cap.isOpened():
    print("Error: Camera could not be opened.")
    exit()

# # Set camera resolution (optional, you can change it based on your camera capabilities)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH,300)  # Set frame width
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT,200)  # Set frame height

print("Streaming... Press 'q' to quit")

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture image.")
        break

    # Display the frame
    cv2.imshow('Camera Stream', frame)

    # Wait for 'q' key to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
