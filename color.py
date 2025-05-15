# color.py
import numpy as np
import cv2

class Color:
    def __init__(self, name: str, lower_bound: np.ndarray, upper_bound: np.ndarray):
        self.name = name
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def get_lower_bound(self):
        """Return the lower bound of the color."""
        return self.lower_bound

    def get_upper_bound(self):
        """Return the upper bound of the color."""
        return self.upper_bound

    def get_name(self):
        """Return the name of the color."""
        return self.name


    def initialize_kalman_filter(self):
        kf = cv2.KalmanFilter(4, 2, 0)
        kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                        [0, 1, 0, 1],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]], np.float32)
        kf.measurementMatrix = np.eye(2, 4, dtype=np.float32)
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
        return kf
