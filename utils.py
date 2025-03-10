import cv2
import numpy as np


retval, corners = cv2.findChessboardCorners(image, patternSize, flags)
