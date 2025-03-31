import cv2
import numpy as np

def nothing(x):
    pass

def get_masked_image(img, hsv_min, hsv_max):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_min, hsv_max)
    masked_image = cv2.bitwise_and(img, img, mask=mask)
    return mask, masked_image

# Create a window for the trackbars
cv2.namedWindow('Trackbars')

# Create 6 trackbars：H_min, S_min, V_min, H_max, S_max, V_max
cv2.createTrackbar('H_min', 'Trackbars', 0, 179, nothing)
cv2.createTrackbar('S_min', 'Trackbars', 0, 255, nothing)
cv2.createTrackbar('V_min', 'Trackbars', 0, 255, nothing)
cv2.createTrackbar('H_max', 'Trackbars', 179, 179, nothing)
cv2.createTrackbar('S_max', 'Trackbars', 255, 255, nothing)
cv2.createTrackbar('V_max', 'Trackbars', 255, 255, nothing)

# Read the image from a file
img = cv2.imread("localization/test_camera_0.png")
if img is None:
    print("Error: Image could not be read.")
    exit()

while True:
    # Get the current positions of the trackbars
    h_min = cv2.getTrackbarPos('H_min', 'Trackbars')
    s_min = cv2.getTrackbarPos('S_min', 'Trackbars')
    v_min = cv2.getTrackbarPos('V_min', 'Trackbars')
    h_max = cv2.getTrackbarPos('H_max', 'Trackbars')
    s_max = cv2.getTrackbarPos('S_max', 'Trackbars')
    v_max = cv2.getTrackbarPos('V_max', 'Trackbars')
    
    # Create the HSV range
    lower_hsv = np.array([h_min, s_min, v_min])
    upper_hsv = np.array([h_max, s_max, v_max])
    
    # Get the masked image
    mask, masked_img = get_masked_image(img, lower_hsv, upper_hsv)
    
    # Stack all 3 images into one, original image, the mask and the masked image
    images = np.hstack((img, masked_img))
    
    # Show the images
    cv2.imshow("Original Image & Masked Image", images)
    cv2.imshow("Mask", mask)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
