import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
import HomographyCalibrationFSM
from calibration_helper import generate_square_points, draw_playing_area, calibrate_playing_area
import numpy as np

# Camera native resolution
camera_width = 1920
camera_height = 1080

# Calibration display resolution (fits twice on most screens, 16:9)
calib_disp_width = 960
calib_disp_height = 540

def save_calibration(homography, play_center, play_radius, calibration_resolution, diagonal_m, margin_m, filename="calibration_data.npz"):
    np.savez(
        filename,
        homography=homography,
        play_center=play_center,
        play_radius=play_radius,
        calibration_resolution=calibration_resolution,
        square_diagonal_m=diagonal_m,
        margin_m=margin_m
    )

def get_reference_square(screen_width, screen_height, angle=-45, margin_percent=0.05):
    """
    Generate a square with a margin (as a percentage of width/height) such that,
    after rotation, the square fits within the frame with the specified margin.
    """
    margin_x = screen_width * margin_percent
    margin_y = screen_height * margin_percent

    # The largest square that fits after rotation with margin
    # The bounding box of a rotated square is sqrt(2) * side_length at 45deg
    max_side_x = (screen_width - 2 * margin_x) / (2 ** 0.5)
    max_side_y = (screen_height - 2 * margin_y) / (2 ** 0.5)
    side_length = min(max_side_x, max_side_y)
    center = (screen_width // 2, screen_height // 2)
    return generate_square_points(center, side_length, angle)

class HomographyCalibrationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Homography Calibration")
        self.fsm = HomographyCalibrationFSM.HomographyCalibrationFSM()
        self.cap = cv2.VideoCapture(0)
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
        self.frame = None
        self.display_width = calib_disp_width
        self.display_height = calib_disp_height

        # Top frame for state label and buttons
        self.top_frame = tk.Frame(root)
        self.top_frame.pack(side=tk.TOP, fill=tk.X)

        self.state_label = tk.Label(self.top_frame, text="State: IDLE")
        self.state_label.pack(side=tk.LEFT, padx=10)

        self.confirm_btn = tk.Button(self.top_frame, text="Confirm", command=self.on_confirm)
        self.confirm_btn.pack(side=tk.LEFT, padx=10, pady=10)

        self.back_btn = tk.Button(self.top_frame, text="Back", command=self.on_back)
        self.back_btn.pack(side=tk.LEFT, padx=10, pady=10)

        self.save_btn = tk.Button(self.top_frame, text="Save Calibration", command=self.on_save)
        self.save_btn.pack(side=tk.LEFT, padx=10, pady=10)

        # Diagonal slider (0.01m to 20m)
        self.diagonal_m = tk.DoubleVar(value=2.828)  # Default: sqrt(2^2 + 2^2) ≈ 2.828m for 2m side
        self.diagonal_slider = tk.Scale(
            self.top_frame, from_=0.01, to=20.0, resolution=0.01, orient=tk.HORIZONTAL,
            label="Square Diagonal (meters)", variable=self.diagonal_m, length=200
        )
        self.diagonal_slider.pack(side=tk.LEFT, padx=10)

        # Margin slider (-1m to 3m)
        self.margin_m = tk.DoubleVar(value=0.2)
        self.margin_slider = tk.Scale(
            self.top_frame, from_=-1.0, to=3.0, resolution=0.01, orient=tk.HORIZONTAL,
            label="Margin (meters)", variable=self.margin_m, length=200
        )
        self.margin_slider.pack(side=tk.LEFT, padx=10)

        self.label = tk.Label(root, text="Press Confirm to start calibration.")
        self.label.pack()

        self.canvas = tk.Canvas(root, width=self.display_width * 2, height=self.display_height)
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        self.update_frame()

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.root.after(10, self.update_frame)
            return
        self.frame = frame.copy()

        # Resize frame for display
        disp_frame = cv2.resize(self.frame, (self.display_width, self.display_height), interpolation=cv2.INTER_AREA)

        if self.fsm.state == "SHOW_WARP" and self.fsm.homography is not None:
            orig = disp_frame.copy()
            scale_x = self.display_width / camera_width
            scale_y = self.display_height / camera_height
            for idx, pt in enumerate(self.fsm.points):
                disp_pt = (int(pt[0] * scale_x), int(pt[1] * scale_y))
                cv2.circle(orig, disp_pt, 5, (0, 255, 0), -1)
                cv2.putText(orig, str(idx + 1), disp_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            h, w = orig.shape[:2]
            angle = -45
            diagonal_m = self.diagonal_m.get()
            if diagonal_m > 0:
                real_square_size_m = diagonal_m / (2 ** 0.5)
            else:
                real_square_size_m = 0.0
            # Use new reference square with 5% margin, ensuring fit after rotation
            target_square = get_reference_square(w, h, angle, margin_percent=0.05)
            if len(self.fsm.points) == 4:
                src_pts = np.array(self.fsm.points, dtype=np.float32)
                dst_pts = np.array([
                    (pt[0] * scale_x, pt[1] * scale_y) for pt in self.fsm.points
                ], dtype=np.float32)
                disp_homography, _ = cv2.findHomography(dst_pts, np.array(target_square, dtype=np.float32))
                warped = cv2.warpPerspective(orig, disp_homography, (w, h))
            else:
                warped = orig.copy()
            for idx, pt in enumerate(target_square):
                cv2.circle(warped, pt, 5, (0, 0, 255), -1)
                cv2.putText(warped, str(idx + 1), pt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            margin_m = self.margin_m.get()
            play_center, play_radius = calibrate_playing_area(
                target_square, real_square_size_m, margin_m
            )
            draw_playing_area(warped, play_center, play_radius)
            combined = cv2.hconcat([orig, warped])
            img = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.config(width=self.display_width * 2, height=self.display_height)
            self.canvas.imgtk = imgtk
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        else:
            # Draw points on display frame
            scale_x = self.display_width / camera_width
            scale_y = self.display_height / camera_height
            for idx, pt in enumerate(self.fsm.points):
                disp_pt = (int(pt[0] * scale_x), int(pt[1] * scale_y))
                cv2.circle(disp_frame, disp_pt, 5, (0, 255, 0), -1)
                cv2.putText(disp_frame, str(idx + 1), disp_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            img = cv2.cvtColor(disp_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.config(width=self.display_width, height=self.display_height)
            self.canvas.imgtk = imgtk
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)

        self.update_label()
        self.root.after(30, self.update_frame)

    def update_label(self):
        # Update the main instruction label
        match self.fsm.state:
            case "IDLE":
                self.label.config(text="Press Confirm to start calibration.")
            case "COLLECT_POINTS":
                self.label.config(text=f"Select 4 corners. Points selected: {len(self.fsm.points)}/4")
            case "CALCULATE":
                self.label.config(text="Ready to calculate homography. Press Confirm.")
            case "SHOW_WARP":
                self.label.config(text="Showing original and warped frame. Press Confirm to restart.")
            case _:
                self.label.config(text="Unknown state.")
        # Update the FSM state label
        self.state_label.config(text=f"State: {self.fsm.state}")

    def on_canvas_click(self, event):
        # Map click to original camera resolution
        if self.fsm.state == "COLLECT_POINTS" and len(self.fsm.points) < 4:
            x = int(event.x * camera_width / self.display_width)
            y = int(event.y * camera_height / self.display_height)
            self.fsm.add_point((x, y))

    def on_confirm(self):
        match self.fsm.state:
            case "IDLE":
                self.fsm.start()
            case "COLLECT_POINTS":
                pass
            case "CALCULATE":
                h, w = self.frame.shape[:2]
                angle = -45
                # Use new reference square with 5% margin, ensuring fit after rotation
                target_square = get_reference_square(w, h, angle, margin_percent=0.05)
                self.fsm.calculate(target_square)
            case "SHOW_WARP":
                self.fsm.reset()
            case _:
                pass

    def on_back(self):
        match self.fsm.state:
            case "COLLECT_POINTS":
                self.fsm.remove_last_point()
            case "CALCULATE":
                self.fsm.state = "COLLECT_POINTS"
                self.fsm.remove_last_point()
            case "SHOW_WARP":
                self.fsm.reset()
            case _:
                pass

    def on_closing(self):
        self.cap.release()
        self.root.destroy()

    def on_save(self):
        if self.fsm.homography is not None:
            h, w = camera_height, camera_width
            angle = -45
            diagonal_m = self.diagonal_m.get()
            if diagonal_m > 0:
                real_square_size_m = diagonal_m / (2 ** 0.5)
            else:
                real_square_size_m = 0.1
            # Use new reference square with 5% margin, ensuring fit after rotation
            target_square = get_reference_square(w, h, angle, margin_percent=0.05)
            margin_m = self.margin_m.get()
            play_center, play_radius = calibrate_playing_area(target_square, real_square_size_m, margin_m)
            calibration_resolution = (camera_width, camera_height)  # Save camera resolution, not display
            # Store diagonal_m and margin_m in calibration file
            save_calibration(self.fsm.homography, play_center, play_radius, calibration_resolution, diagonal_m, margin_m)

if __name__ == "__main__":
    root = tk.Tk()
    app = HomographyCalibrationApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
