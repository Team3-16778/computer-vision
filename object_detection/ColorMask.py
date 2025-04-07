import cv2
import numpy as np
import glob
import os

class MockCamera:
    def __init__(self):
        self.camera_matrix = np.eye(3)
        self.camera_matrix[0, 0] = 1000
        self.camera_matrix[1, 1] = 1000
        self.camera_matrix[0, 2] = 640
        self.camera_matrix[1, 2] = 480
        self.Rotation_matrix = np.eye(3)
        self.Translation_vector = np.zeros((3, 1))

def calculate_world_3D(camera, u, v, Zc=0.5):
    fx = camera.camera_matrix[0, 0]
    fy = camera.camera_matrix[1, 1]
    cx = camera.camera_matrix[0, 2]
    cy = camera.camera_matrix[1, 2]
    R = camera.Rotation_matrix
    T = camera.Translation_vector

    Xc = (u - cx) * Zc / fx
    Yc = (v - cy) * Zc / fy
    camera_point = np.array([[Xc], [Yc], [Zc]])
    world_point = R @ camera_point + T
    return world_point.flatten()

class DarkContourTester:
    def __init__(self, image_dir, camera=None):
        self.image_dir = image_dir
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))[:3]
        self.camera = camera if camera else MockCamera()

    def colormask(self, image):
        blurred = cv2.GaussianBlur(image, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([179, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masked_image = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 100:
                x, y, w, h = cv2.boundingRect(largest_contour)
                cx, cy = x + w // 2, y + h // 2

                cv2.rectangle(mask_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.drawMarker(mask_bgr, (cx, cy), (255, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

                print(f"Detected object center: ({cx}, {cy})")
                world_coords = calculate_world_3D(self.camera, x, y)
                label = "({:.2f}, {:.2f}, {:.2f})".format(*world_coords)
                cv2.putText(image_rgb, label, (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                print(f"World coords: X={world_coords[0]:.2f}, Y={world_coords[1]:.2f}, Z={world_coords[2]:.2f}")

        return image_rgb, masked_image, mask_bgr

    def run(self):
        for i, image_path in enumerate(self.image_paths):
            image = cv2.imread(image_path)
            if image is None:
                print(f"Could not read {image_path}")
                continue

            print(f"\nProcessing {image_path}")
            _, output_image, mask_bgr = self.colormask(image)

            stacked = np.hstack((
                cv2.resize(image, (640, 480)),
                cv2.resize(output_image, (640, 480)),
                cv2.resize(mask_bgr, (640, 480))
            ))
            cv2.imshow(f"Result {i+1}: {os.path.basename(image_path)}", stacked)
            cv2.waitKey(0)

        cv2.destroyAllWindows()

if __name__ == "__main__":
    image_dir = "/Users/meesh/Desktop/CMU/MechDesign/computer-vision/object_detection/images/ribcage"  # Update this path
    tester = DarkContourTester(image_dir)
    tester.run()