import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread("colormask_images/grape_demo.png")  # Change to your image path
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for proper display

# Convert to HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define color range for masking (adjust as needed)
lower_bound = np.array([120, 30, 30])  # Relaxed: Purple lower bound
upper_bound = np.array([180, 255, 255])  # Relaxed: Purple upper bound

# Create the mask (black and white)
mask = cv2.inRange(hsv, lower_bound, upper_bound)

# Compute average coordinates of the masked area
coordinates = np.column_stack(np.where(mask > 0))  # Get all nonzero pixel locations
if len(coordinates) > 0:
    avg_y, avg_x = np.mean(coordinates, axis=0).astype(int)  # Compute average coordinates
    print(f"Average object location: ({avg_x}, {avg_y})")
else:
    avg_x, avg_y = -1, -1  # Default if nothing is detected

# Apply the mask to the original image
masked_image = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)

# Convert mask to 3-channel for visualization
mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

# Draw a red marker at the average location
if avg_x >= 0 and avg_y >= 0:
    cv2.drawMarker(mask_bgr, (avg_x, avg_y), (255, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

# Display the images side by side
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(image_rgb)
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(masked_image)
axes[1].set_title("Masked Image")
axes[1].axis("off")

axes[2].imshow(mask_bgr)
axes[2].set_title("Black & White Mask with Marker")
axes[2].axis("off")

plt.show()
