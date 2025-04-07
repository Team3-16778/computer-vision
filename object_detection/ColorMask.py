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

class ColorMaskTuner:
    def __init__(self, window_name="Live Mask Tuning"):  # <-- unified window name
        self.window_name = window_name
        self.lower1 = [0, 100, 100]
        self.upper1 = [10, 255, 255]
        self.lower2 = [160, 100, 100]
        self.upper2 = [179, 255, 255]

        cv2.namedWindow(self.window_name)
        self._create_trackbars()

    def _create_trackbars(self):
        ranges = [
            ("[Range 1] Low H", self.lower1[0], 179),
            ("[Range 1] Low S", self.lower1[1], 255),
            ("[Range 1] Low V", self.lower1[2], 255),
            ("[Range 1] High H", self.upper1[0], 179),
            ("[Range 1] High S", self.upper1[1], 255),
            ("[Range 1] High V", self.upper1[2], 255),
            ("", 0, 1),  # spacer
            ("[Range 2] Low H", self.lower2[0], 179),
            ("[Range 2] Low S", self.lower2[1], 255),
            ("[Range 2] Low V", self.lower2[2], 255),
            ("[Range 2] High H", self.upper2[0], 179),
            ("[Range 2] High S", self.upper2[1], 255),
            ("[Range 2] High V", self.upper2[2], 255),
        ]

        for label, default, max_val in ranges:
            # If it's a spacer, use a dummy no-op trackbar
            if label == "":
                cv2.createTrackbar(" ", self.window_name, 0, 1, lambda x: None)
            else:
                cv2.createTrackbar(label, self.window_name, default, max_val, lambda x: None)



    def _get_trackbar_values(self):
        names = [
            "[Range 1] Low H", "[Range 1] Low S", "[Range 1] Low V",
            "[Range 1] High H", "[Range 1] High S", "[Range 1] High V",
            "[Range 2] Low H", "[Range 2] Low S", "[Range 2] Low V",
            "[Range 2] High H", "[Range 2] High S", "[Range 2] High V"
        ]
        vals = [cv2.getTrackbarPos(name, self.window_name) for name in names]
        return vals[:3], vals[3:6], vals[6:9], vals[9:12]


    def apply_mask(self, image):
        blurred = cv2.GaussianBlur(image, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        self.lower1, self.upper1, self.lower2, self.upper2 = self._get_trackbar_values()

        mask1 = cv2.inRange(hsv, np.array(self.lower1), np.array(self.upper1))
        mask2 = cv2.inRange(hsv, np.array(self.lower2), np.array(self.upper2))
        mask = cv2.bitwise_or(mask1, mask2)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        masked_image = cv2.bitwise_and(image, image, mask=mask)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        return masked_image, mask_bgr

class ColorMask:
    def __init__(self, image_dir, camera=None, tuner_enabled=False):
        self.image_dir = image_dir
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))[:3]
        self.camera = camera if camera else MockCamera()
        self.tuner = ColorMaskTuner() if tuner_enabled else None

    def colormask(self, image):
        blurred = cv2.GaussianBlur(image, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        if self.tuner:
            l1, u1, l2, u2 = self.tuner._get_trackbar_values()
        else:
            l1 = [0, 100, 100]
            u1 = [10, 255, 255]
            l2 = [160, 100, 100]
            u2 = [179, 255, 255]

        mask1 = cv2.inRange(hsv, np.array(l1), np.array(u1))
        mask2 = cv2.inRange(hsv, np.array(l2), np.array(u2))
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
            key = cv2.waitKey(0)
            if key == 27:  # ESC to break early
                break
        cv2.destroyAllWindows()

    def run_live_tuning(self):
        if not self.image_paths:
            print("No images found in directory.")
            return

        index = 0  # start with first image

        while index < len(self.image_paths):
            image = cv2.imread(self.image_paths[index])
            if image is None:
                print(f"Could not read {self.image_paths[index]}")
                index += 1
                continue

            while True:
                display_img = image.copy()
                _, output_image, mask_bgr = self.colormask(display_img)

                stacked = np.hstack((
                    cv2.resize(image, (640, 480)),
                    cv2.resize(output_image, (640, 480)),
                    cv2.resize(mask_bgr, (640, 480))
                ))

                cv2.imshow(self.tuner.window_name, stacked)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC to exit completely
                    cv2.destroyAllWindows()
                    return
                elif key == 32:  # SPACE to go to next image
                    break  # break inner loop, advance index

            index += 1

        cv2.destroyAllWindows()




if __name__ == "__main__":
    image_dir = "/Users/meesh/Desktop/CMU/MechDesign/computer-vision/object_detection/images/ribcage"  # Update this path
    tester = ColorMask(image_dir, tuner_enabled=True)
    tester.run_live_tuning()

