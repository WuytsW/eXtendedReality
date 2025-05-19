import numpy as np
import cv2
class HomographyCalibrationFSM:
    def __init__(self):
        self.state = "IDLE"
        self.points = []
        self.homography = None

    def start(self):
        self.state = "COLLECT_POINTS"
        self.points = []
        self.homography = None

    def add_point(self, point):
        if self.state == "COLLECT_POINTS" and len(self.points) < 4:
            self.points.append(point)
            if len(self.points) == 4:
                self.state = "CALCULATE"

    def remove_last_point(self):
        if self.state == "COLLECT_POINTS" and self.points:
            self.points.pop()

    def calculate(self, target_square):
        if self.state == "CALCULATE":
            import numpy as np
            src_pts = np.array(self.points, dtype=np.float32)
            dst_pts = np.array(target_square, dtype=np.float32)
            self.homography, _ = cv2.findHomography(src_pts, dst_pts)
            self.state = "SHOW_WARP"

    def reset(self):
        self.state = "IDLE"
        self.points = []
        self.homography = None

    def is_done(self):
        return self.state == "SHOW_WARP"