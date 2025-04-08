import sys
import cv2
import numpy as np
import os
import glob
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QGridLayout,
    QSlider, QSpinBox, QPushButton
)
from PyQt6.QtCore import Qt, QTimer, QPoint
from PyQt6.QtGui import QImage, QPixmap, QMouseEvent

class ColorMask(QWidget):
    def __init__(self, image_dir):
        super().__init__()
        self.setWindowTitle("Color Mask Tuner with Eyedropper")
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, '*.png')))
        self.index = 0

        self.label = QLabel()
        self.label.setMouseTracking(True)
        self.label.mousePressEvent = self.pick_color
        self.display_img = None

        self.sliders = {}
        self.init_ui()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def init_ui(self):
        layout = QVBoxLayout()
        layout.addWidget(self.label)

        # Button to go to next image
        next_btn = QPushButton("Next Image")
        next_btn.clicked.connect(self.next_image)
        layout.addWidget(next_btn)

        # HSV Sliders
        slider_layout = QGridLayout()
        names = ["Low H", "Low S", "Low V", "High H", "High S", "High V"]
        default_values = [0, 100, 100, 179, 255, 255]

        for i, (name, val) in enumerate(zip(names, default_values)):
            lbl = QLabel(name)
            sld = QSlider(Qt.Orientation.Horizontal)
            sld.setRange(0, 255 if 'S' in name or 'V' in name else 179)
            sld.setValue(val)
            spn = QSpinBox()
            spn.setRange(0, 255 if 'S' in name or 'V' in name else 179)
            spn.setValue(val)
            sld.valueChanged.connect(spn.setValue)
            spn.valueChanged.connect(sld.setValue)
            sld.valueChanged.connect(self.update_frame)

            self.sliders[name] = spn
            slider_layout.addWidget(lbl, i, 0)
            slider_layout.addWidget(sld, i, 1)
            slider_layout.addWidget(spn, i, 2)

        layout.addLayout(slider_layout)
        self.setLayout(layout)

    def next_image(self):
        self.index = (self.index + 1) % len(self.image_paths)

    def get_hsv_bounds(self):
        l = [self.sliders[f"Low H"].value(), self.sliders[f"Low S"].value(), self.sliders[f"Low V"].value()]
        u = [self.sliders[f"High H"].value(), self.sliders[f"High S"].value(), self.sliders[f"High V"].value()]
        return np.array(l), np.array(u)

    def update_frame(self):
        if not self.image_paths:
            return

        image = cv2.imread(self.image_paths[self.index])
        if image is None:
            return

        blurred = cv2.GaussianBlur(image, (7, 7), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        lower, upper = self.get_hsv_bounds()
        mask = cv2.inRange(hsv, lower, upper)

        # Morphological cleanup
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Blob detection
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        overlay = image.copy()
        masked = cv2.bitwise_and(image, image, mask=mask)

        if contours:
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            cx, cy = x + w // 2, y + h // 2

            # Draw on overlay
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(overlay, (cx, cy), 5, (255, 0, 0), -1)  # blue center

            # Draw on masked
            cv2.rectangle(masked, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(masked, (cx, cy), 5, (255, 0, 0), -1)  # blue center

            print(f"Target blob at ({cx}, {cy})")
            # world_coords = calculate_world_3D(self.camera, cx, cy)
            # print(f"World coords: X={world_coords[0]:.2f}, Y={world_coords[1]:.2f}, Z={world_coords[2]:.2f}")

        # Final side-by-side view
        display = np.hstack((image, masked, overlay))
        display = cv2.resize(display, (960, 320))
        display_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)

        self.display_img = display_rgb
        h, w, ch = display_rgb.shape
        qimg = QImage(display_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(qimg))

    def pick_color(self, event: QMouseEvent):
        if self.display_img is None or not self.image_paths:
            return

        # Click coordinates in the display (resized + side-by-side layout)
        x_disp = int(event.position().x())
        y_disp = int(event.position().y())

        # Define original image resolution
        orig_image = cv2.imread(self.image_paths[self.index])
        if orig_image is None:
            return
        h_orig, w_orig = orig_image.shape[:2]

        # Assume 3 panels side-by-side, equally divided (original, masked, mask)
        panel_width = self.display_img.shape[1] // 3
        if x_disp > panel_width:
            return  # Only allow picking on the first (original) panel

        # Scale coordinates to original image resolution
        x_ratio = w_orig / panel_width
        y_ratio = h_orig / self.display_img.shape[0]
        x = int(x_disp * x_ratio)
        y = int(y_disp * y_ratio)

        # Convert to HSV
        hsv = cv2.cvtColor(orig_image, cv2.COLOR_BGR2HSV)

        # Sample 5x5 patch and use median
        radius = 2
        x1, x2 = max(0, x - radius), min(w_orig, x + radius + 1)
        y1, y2 = max(0, y - radius), min(h_orig, y + radius + 1)
        patch = hsv[y1:y2, x1:x2]
        median = np.median(patch.reshape(-1, 3), axis=0)

        margin = 20
        low = np.clip(median - margin, [0, 0, 0], [179, 255, 255]).astype(int)
        high = np.clip(median + margin, [0, 0, 0], [179, 255, 255]).astype(int)

        self.sliders["Low H"].setValue(low[0])
        self.sliders["Low S"].setValue(low[1])
        self.sliders["Low V"].setValue(low[2])
        self.sliders["High H"].setValue(high[0])
        self.sliders["High S"].setValue(high[1])
        self.sliders["High V"].setValue(high[2])

        self.update_frame()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    cm = ColorMask("/Users/meesh/Desktop/CMU/MechDesign/computer-vision/object_detection/images/ribcage")
    cm.show()
    sys.exit(app.exec())
