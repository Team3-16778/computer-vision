# import sys
# import cv2
# import numpy as np
# import glob
# import os
# from PyQt6.QtWidgets import (
#     QApplication, QWidget, QSlider, QLabel, QVBoxLayout, QHBoxLayout,
#     QGroupBox, QGridLayout, QSpinBox, QMainWindow, QPushButton
# )
# from PyQt6.QtCore import Qt, QTimer
# from PyQt6.QtGui import QImage, QPixmap

# class MockCamera:
#     def __init__(self):
#         self.camera_matrix = np.eye(3)
#         self.camera_matrix[0, 0] = 1000
#         self.camera_matrix[1, 1] = 1000
#         self.camera_matrix[0, 2] = 640
#         self.camera_matrix[1, 2] = 480
#         self.Rotation_matrix = np.eye(3)
#         self.Translation_vector = np.zeros((3, 1))

# def calculate_world_3D(camera, u, v, Zc=0.5):
#     fx = camera.camera_matrix[0, 0]
#     fy = camera.camera_matrix[1, 1]
#     cx = camera.camera_matrix[0, 2]
#     cy = camera.camera_matrix[1, 2]
#     R = camera.Rotation_matrix
#     T = camera.Translation_vector

#     Xc = (u - cx) * Zc / fx
#     Yc = (v - cy) * Zc / fy
#     camera_point = np.array([[Xc], [Yc], [Zc]])
#     world_point = R @ camera_point + T
#     return world_point.flatten()

# class ColorMaskTuner(QWidget):
#     def __init__(self):
#         super().__init__()
#         self.lower1 = [0, 100, 100]
#         self.upper1 = [10, 255, 255]
#         self.lower2 = [160, 100, 100]
#         self.upper2 = [179, 255, 255]

#         self.sliders = {}
#         self.init_ui()

#     def init_ui(self):
#         layout = QGridLayout()

#         ranges = [
#             ("Low H1", 0, 179), ("Low S1", 100, 255), ("Low V1", 100, 255),
#             ("High H1", 10, 179), ("High S1", 255, 255), ("High V1", 255, 255),
#             ("Low H2", 160, 179), ("Low S2", 100, 255), ("Low V2", 100, 255),
#             ("High H2", 179, 179), ("High S2", 255, 255), ("High V2", 255, 255),
#         ]

#         for i, (label, val, max_val) in enumerate(ranges):
#             lbl = QLabel(label)
#             slider = QSlider(Qt.Orientation.Horizontal)
#             slider.setRange(0, max_val)
#             slider.setValue(val)
#             spin = QSpinBox()
#             spin.setRange(0, max_val)
#             spin.setValue(val)
#             slider.valueChanged.connect(spin.setValue)
#             spin.valueChanged.connect(slider.setValue)
#             layout.addWidget(lbl, i, 0)
#             layout.addWidget(slider, i, 1)
#             layout.addWidget(spin, i, 2)
#             self.sliders[label] = spin

#         self.setLayout(layout)

#     def get_values(self):
#         l1 = [self.sliders[f"Low H1"].value(), self.sliders[f"Low S1"].value(), self.sliders[f"Low V1"].value()]
#         u1 = [self.sliders[f"High H1"].value(), self.sliders[f"High S1"].value(), self.sliders[f"High V1"].value()]
#         l2 = [self.sliders[f"Low H2"].value(), self.sliders[f"Low S2"].value(), self.sliders[f"Low V2"].value()]
#         u2 = [self.sliders[f"High H2"].value(), self.sliders[f"High S2"].value(), self.sliders[f"High V2"].value()]
#         return l1, u1, l2, u2

# class ColorMaskApp(QWidget):
#     def __init__(self, image_dir):
#         super().__init__()
#         self.setWindowTitle("Color Mask Tuning")
#         self.image_paths = sorted(glob.glob(os.path.join(image_dir, '*.png')))
#         self.index = 0
#         self.camera = MockCamera()

#         self.tuner = ColorMaskTuner()
#         self.label = QLabel()

#         layout = QVBoxLayout()
#         layout.addWidget(self.label)
#         layout.addWidget(self.tuner)
#         self.setLayout(layout)

#         self.timer = QTimer()
#         self.timer.timeout.connect(self.update_frame)
#         self.timer.start(30)

#     def keyPressEvent(self, event):
#         if event.key() == Qt.Key.Key_Space:
#             self.index += 1
#             if self.index >= len(self.image_paths):
#                 self.index = 0

#     def update_frame(self):
#         if not self.image_paths:
#             return

#         image = cv2.imread(self.image_paths[self.index])
#         if image is None:
#             return

#         blurred = cv2.GaussianBlur(image, (11, 11), 0)
#         hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

#         l1, u1, l2, u2 = self.tuner.get_values()
#         mask1 = cv2.inRange(hsv, np.array(l1), np.array(u1))
#         mask2 = cv2.inRange(hsv, np.array(l2), np.array(u2))
#         mask = cv2.bitwise_or(mask1, mask2)

#         kernel = np.ones((5, 5), np.uint8)
#         mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
#         mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

#         masked = cv2.bitwise_and(image, image, mask=mask)
#         stacked = np.hstack((image, masked, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)))
#         stacked = cv2.resize(stacked, (640, 213))  # Smaller output
#         stacked = cv2.cvtColor(stacked, cv2.COLOR_BGR2RGB)

#         h, w, ch = stacked.shape
#         img_qt = QImage(stacked.data, w, h, ch * w, QImage.Format.Format_RGB888)
#         self.label.setPixmap(QPixmap.fromImage(img_qt))

# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     window = ColorMaskApp("/Users/meesh/Desktop/CMU/MechDesign/computer-vision/object_detection/images/ribcage")
#     window.show()
#     sys.exit(app.exec())


import sys
import cv2
import numpy as np
import os
import glob
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QSlider, QSpinBox, QGridLayout
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap

class ColorMask(QWidget):
    def __init__(self, image_dir):
        super().__init__()
        self.setWindowTitle("Simple Color Mask Tuner")
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, '*.png')))
        self.index = 0

        self.label = QLabel()
        self.sliders = {}
        self.init_ui()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def init_ui(self):
        layout = QVBoxLayout()
        layout.addWidget(self.label)

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

        result = cv2.bitwise_and(image, image, mask=mask)
        display = np.hstack((image, result, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)))
        display = cv2.resize(display, (960, 320))
        display = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)

        h, w, ch = display.shape
        qimg = QImage(display.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(qimg))

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space:
            self.index = (self.index + 1) % len(self.image_paths)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    cm = ColorMask("/Users/meesh/Desktop/CMU/MechDesign/computer-vision/object_detection/images/ribcage")
    cm.show()
    sys.exit(app.exec())
