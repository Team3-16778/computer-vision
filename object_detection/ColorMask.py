import sys
import cv2
import numpy as np
import os
import glob
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QGridLayout,
    QSlider, QSpinBox, QPushButton, QHBoxLayout
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap, QMouseEvent

class ColorMask(QWidget):
    def __init__(self, image_dir):
        super().__init__()
        self.setWindowTitle("Color Mask Tuner with Dual Hue Support")
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

        next_btn = QPushButton("Next Image")
        next_btn.clicked.connect(self.next_image)
        layout.addWidget(next_btn)

        grid = QGridLayout()
        hsv_names = ["Low H", "Low S", "Low V", "High H", "High S", "High V"]
        default_values1 = [0, 100, 100, 10, 255, 255]
        default_values2 = [160, 100, 100, 179, 255, 255]

        for col, (suffix, defaults) in enumerate(zip(['1', '2'], [default_values1, default_values2])):
            for row, (name, val) in enumerate(zip(hsv_names, defaults)):
                full_name = f"{name}{suffix}"
                lbl = QLabel(full_name)
                sld = QSlider(Qt.Orientation.Horizontal)
                sld.setRange(0, 255 if 'S' in name or 'V' in name else 179)
                sld.setValue(val)
                spn = QSpinBox()
                spn.setRange(0, 255 if 'S' in name or 'V' in name else 179)
                spn.setValue(val)
                sld.valueChanged.connect(spn.setValue)
                spn.valueChanged.connect(sld.setValue)
                sld.valueChanged.connect(self.update_frame)

                self.sliders[full_name] = spn
                grid.addWidget(lbl, row, col * 3 + 0)
                grid.addWidget(sld, row, col * 3 + 1)
                grid.addWidget(spn, row, col * 3 + 2)

        layout.addLayout(grid)
        self.setLayout(layout)

    def next_image(self):
        self.index = (self.index + 1) % len(self.image_paths)

    def get_hsv_bounds(self):
        l1 = np.array([self.sliders[f"Low H1"].value(), self.sliders[f"Low S1"].value(), self.sliders[f"Low V1"].value()])
        u1 = np.array([self.sliders[f"High H1"].value(), self.sliders[f"High S1"].value(), self.sliders[f"High V1"].value()])
        l2 = np.array([self.sliders[f"Low H2"].value(), self.sliders[f"Low S2"].value(), self.sliders[f"Low V2"].value()])
        u2 = np.array([self.sliders[f"High H2"].value(), self.sliders[f"High S2"].value(), self.sliders[f"High V2"].value()])
        return l1, u1, l2, u2

    def update_frame(self):
        if not self.image_paths:
            return

        image = cv2.imread(self.image_paths[self.index])
        if image is None:
            return

        blurred = cv2.GaussianBlur(image, (7, 7), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        l1, u1, l2, u2 = self.get_hsv_bounds()

        mask1 = cv2.inRange(hsv, l1, u1)
        mask2 = cv2.inRange(hsv, l2, u2)
        mask = cv2.bitwise_or(mask1, mask2)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        overlay = image.copy()
        masked = cv2.bitwise_and(image, image, mask=mask)

        if contours:
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            cx, cy = x + w // 2, y + h // 2
            for img in [overlay, masked]:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(img, (cx, cy), 5, (255, 0, 0), -1)
            print(f"Target blob at ({cx}, {cy})")

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

        x_disp = int(event.position().x())
        y_disp = int(event.position().y())

        orig_image = cv2.imread(self.image_paths[self.index])
        if orig_image is None:
            return
        h_orig, w_orig = orig_image.shape[:2]

        panel_width = self.display_img.shape[1] // 3
        if x_disp > panel_width:
            return

        x_ratio = w_orig / panel_width
        y_ratio = h_orig / self.display_img.shape[0]
        x = int(x_disp * x_ratio)
        y = int(y_disp * y_ratio)

        hsv = cv2.cvtColor(orig_image, cv2.COLOR_BGR2HSV)

        radius = 2
        x1, x2 = max(0, x - radius), min(w_orig, x + radius + 1)
        y1, y2 = max(0, y - radius), min(h_orig, y + radius + 1)
        patch = hsv[y1:y2, x1:x2]
        median = np.median(patch.reshape(-1, 3), axis=0)

        margin = 20
        low = np.clip(median - margin, [0, 0, 0], [179, 255, 255]).astype(int)
        high = np.clip(median + margin, [0, 0, 0], [179, 255, 255]).astype(int)

        if low[0] <= high[0]:
            # Normal range
            self.sliders["Low H1"].setValue(int(low[0]))
            self.sliders["Low S1"].setValue(int(low[1]))
            self.sliders["Low V1"].setValue(int(low[2]))
            self.sliders["High H1"].setValue(int(high[0]))
            self.sliders["High S1"].setValue(int(high[1]))
            self.sliders["High V1"].setValue(int(high[2]))
            self.sliders["Low H2"].setValue(0)
            self.sliders["High H2"].setValue(0)
        else:
            # Wraparound case
            self.sliders["Low H1"].setValue(0)
            self.sliders["High H1"].setValue(int(high[0]))
            self.sliders["Low H2"].setValue(int(low[0]))
            self.sliders["High H2"].setValue(179)

        self.sliders["Low S1"].setValue(int(low[1]))
        self.sliders["Low V1"].setValue(int(low[2]))
        self.sliders["High S1"].setValue(int(high[1]))
        self.sliders["High V1"].setValue(int(high[2]))
        self.sliders["Low S2"].setValue(int(low[1]))
        self.sliders["Low V2"].setValue(int(low[2]))
        self.sliders["High S2"].setValue(int(high[1]))
        self.sliders["High V2"].setValue(int(high[2]))

        self.update_frame()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    cm = ColorMask("/Users/meesh/Desktop/CMU/MechDesign/computer-vision/object_detection/images/ribcage")
    cm.show()
    sys.exit(app.exec())