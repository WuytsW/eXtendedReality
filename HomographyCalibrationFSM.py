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
            # Sort points: 0=top, 1=right, 2=bottom, 3=left
            pts = np.array(self.points, dtype=np.float32)
            center = np.mean(pts, axis=0)
            # Calculate angles from center to each point
            angles = np.arctan2(pts[:,1] - center[1], pts[:,0] - center[0])
            # Map: top = smallest positive angle (closest to -pi/2), right = 0, bottom = pi/2, left = pi or -pi
            # We'll sort by angle: top (-pi/2), right (0), bottom (pi/2), left (pi)
            # But since y increases downward, top is the smallest y
            idx_top = np.argmin(pts[:,1])
            idx_bottom = np.argmax(pts[:,1])
            # For right/left, exclude top/bottom
            idxs_side = [i for i in range(4) if i not in [idx_top, idx_bottom]]
            if pts[idxs_side[0],0] > pts[idxs_side[1],0]:
                idx_right = idxs_side[0]
                idx_left = idxs_side[1]
            else:
                idx_right = idxs_side[1]
                idx_left = idxs_side[0]
            ordered = [pts[idx_top], pts[idx_right], pts[idx_bottom], pts[idx_left]]
            src_pts = np.array(ordered, dtype=np.float32)
            dst_pts = np.array(target_square, dtype=np.float32)
            self.homography, _ = cv2.findHomography(src_pts, dst_pts)
            self.state = "SHOW_WARP"

    def reset(self):
        self.state = "IDLE"
        self.points = []
        self.homography = None

    def is_done(self):
        return self.state == "SHOW_WARP"
