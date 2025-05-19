import cv2
import numpy as np
from color_ranges import green, purple, yellow, blue, white
from matplotlib.colors import CSS4_COLORS
import matplotlib
import paho.mqtt.client as mqtt
import time
import json

# Define expected area (in pixels)
EXPECTED_AREA = 150  # Adjust this value based on your use case

# Load calibration data
data = np.load("calibration_data.npz")
homography = data["homography"]
play_center = tuple(data["play_center"])
play_radius = int(data["play_radius"])
calibration_resolution = tuple(data["calibration_resolution"]) if "calibration_resolution" in data else None
square_diagonal_m = float(data["square_diagonal_m"]) if "square_diagonal_m" in data else None

for key in data.files:
    print(f"{key}:")
    print(data[key])
    print()

def color_name_to_rgb(color_name):
    """Convert a color name to an RGB tuple (0-255 range)."""
    hex_color = CSS4_COLORS.get(color_name.lower())
    if hex_color:
        rgb = matplotlib.colors.hex2color(hex_color)  # Returns values in 0-1 range
        return tuple(int(c * 255) for c in rgb)
    else:
        raise ValueError(f"Color '{color_name}' not found in CSS4_COLORS")


def detect_color(frame_hsv, color):
    """Detect contours for a given color."""
    mask = cv2.inRange(frame_hsv, color.get_lower_bound(), color.get_upper_bound())
    kernel = np.ones((5, 5), np.uint8)
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def detect_color_morpho(frame_hsv, color):  # inconsisent lighting
    """Detect contours for a given color."""
    mask = cv2.inRange(frame_hsv, color.get_lower_bound(), color.get_upper_bound())
    kernel = np.ones((5, 5), np.uint8)
    mask_open_closed = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask_open_closed = cv2.morphologyEx(mask_open_closed, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask_open_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def adaptive_detect_color_morpho(frame_hsv, adaptive_hsv_bounds):
    """Detect contours for a given color with adaptive HSV mask."""
    lower, upper = adaptive_hsv_bounds
    mask = cv2.inRange(frame_hsv, lower, upper)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def get_closest_blob_to_position(contours, previous_position):
    """Find the blob closest to the previous position."""
    closest_blob = None
    min_distance = float('inf')
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            distance = np.sqrt((cx - previous_position[0]) ** 2 + (cy - previous_position[1]) ** 2)
            if distance < min_distance:
                min_distance = distance
                closest_blob = contour
    return closest_blob


def get_closest_blob_to_area(contours, expected_area):
    """Find the blob with an area closest to the expected area."""
    closest_blob = None
    min_area_diff = float('inf')
    for contour in contours:
        area = cv2.contourArea(contour)
        area_diff = abs(area - expected_area)
        if area_diff < min_area_diff:
            min_area_diff = area_diff
            closest_blob = contour
    return closest_blob


def get_blob_centroid(contour):
    """Calculate the centroid of a given contour."""
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx, cy)
    return None


## KF FUNCTIONS
def get_closest_blob_to_prediction(contours, pred_x, pred_y):
    """Get closest blob to prediction."""
    best_cx, best_cy = None, None
    min_dist = float('inf')

    for contour in contours:
        if cv2.contourArea(contour) > 100:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = float(M["m10"] / M["m00"])
                cy = float(M["m01"] / M["m00"])
                dist = np.sqrt((cx - pred_x) ** 2 + (cy - pred_y) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    best_cx, best_cy = cx, cy
    return min_dist, best_cx, best_cy


def initialize_kalman_filter():
    kf = cv2.KalmanFilter(4, 2, 0)
    kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)
    kf.measurementMatrix = np.eye(2, 4, dtype=np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2 * 5
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
    return kf


# MQTT settings
broker = "broker.hivemq.com"
port = 1883
topic = "XRCatAndMouse/1111"
client = mqtt.Client()
client.connect(broker, port, 60)

SEND_FREQUENCY_HZ = 10  # Adjust how often data is sent (Hz)
SEND_INTERVAL = 1.0 / SEND_FREQUENCY_HZ
last_send_time = 0

def transform_to_play_area_meters(pos, play_center, play_radius, square_diagonal_m):
    """
    Transform pixel coordinates to play area coordinates in meters:
    - Center is (0,0)
    - Range is -diagonal/2 to diagonal/2 in both x and y (in meters)
    """
    if pos is None or pos["x"] is None or pos["y"] is None or square_diagonal_m is None:
        return {"x": None, "y": None}
    # Pixels per meter along the diagonal
    pixels_per_meter = (play_radius * 2) / square_diagonal_m
    # Shift to center, then convert to meters
    dx = (pos["x"] - play_center[0]) / pixels_per_meter
    dy = (pos["y"] - play_center[1]) / pixels_per_meter
    return {
        "x": round(dx, 4),
        "y": round(dy, 4)
    }

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 60)

    all_tracked_colors = [green, purple, yellow, blue]

    ## DEBUG ## Color print in BGR
    print_color = {"Green": (0, 255, 0), "Purple": (255, 0, 255), "Yellow": (0, 200, 255), "Blue": (255, 135, 0),
                   "White": (255, 255, 255)}
    print_compl_color = {"Green": (0, 0, 255), "Purple": (0, 255, 255), "Yellow": (255, 120, 175),
                         "Blue": (0, 135, 255), "White": (0, 0, 0)}

    ## PARAMETER for Detection AND prediction 
    decay_factor = 0.5
    contour_threshold_list = {"Green": 2, "Purple": 2, "Yellow": 2, "Blue": 2, "White": 2}
    reset_threshold_list = {"Green": 50, "Purple": 50, "Yellow": 50, "Blue": 50, "White": 50}
    adaptive_hsv_bounds = {color.get_name(): None for color in all_tracked_colors}
    ## Initialize KF
    kalman_filters = {}
    initialized_flags = {}
    for color in all_tracked_colors:
        kalman_filters[color.get_name()] = initialize_kalman_filter()
        initialized_flags[color.get_name()] = False

    # Set display resolution (fits twice on most screens, 16:9)
    display_width = 960
    display_height = 540

    # For MQTT sending
    global last_send_time

    # Actual loop
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize to calibration resolution if present
            if calibration_resolution is not None:
                frame = cv2.resize(frame, calibration_resolution, interpolation=cv2.INTER_AREA)

            # Apply homography transformation
            warped = cv2.warpPerspective(frame, homography, (frame.shape[1], frame.shape[0]))

            # Use the warped frame for color detection
            frame_hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)

            # Prepare detected positions for MQTT
            detected_positions = {}

            for color in all_tracked_colors:
                # KF
                name = color.get_name()
                kf = kalman_filters[name]
                initialized = initialized_flags[name]

                contour_threshold = contour_threshold_list[name]
                reset_threshold = reset_threshold_list[name]
                best_cx, best_cy = None, None
                min_dist = float('inf')

                if adaptive_hsv_bounds[name] is not None:
                    ## Adaptive HSV
                    contours = adaptive_detect_color_morpho(frame_hsv, adaptive_hsv_bounds[name])                
                else:
                    ## Predefined HSV
                    contours = detect_color_morpho(frame_hsv, color)

                # KF prediction
                prediction = kf.predict()
                pred_x, pred_y = int(prediction[0]), int(prediction[1])

                if contours:  # If a contour detected
                    ## Function for closest blob to prediction
                    min_dist, best_cx, best_cy = get_closest_blob_to_prediction(contours, pred_x, pred_y)

                    if best_cx is not None and best_cy is not None:
                        detected_positions[name] = {"x": float(best_cx), "y": float(best_cy)}
                        # INITIALIZE KALMAN FILTER only first time
                        if not initialized:
                            kf.statePost = np.array([[best_cx], [best_cy], [0], [0]], np.float32)
                            initialized = True

                        # MEASUREMNT
                        if min_dist < contour_threshold:
                            measurement = np.array([[best_cx], [best_cy]], np.float32)
                            est = kf.correct(measurement)
                            est_x, est_y = int(est[0]), int(est[1])
                            cv2.circle(warped, (est_x, est_y), 5, print_color[name], -1)
                            cv2.putText(warped, f"Measuremnt for {name}: ({est_x},{est_y})", (est_x + 10, est_y),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, print_color[name], 1)
                        elif min_dist < reset_threshold:
                            cv2.circle(warped, (pred_x, pred_y), 5, print_compl_color[name], -1)
                            cv2.putText(warped, f"KF PREDICTION for {name}: bad measurement", (pred_x + 10, pred_y + 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, print_compl_color[name], 1)
                        else:
                            # Preserve velocity from the last known state
                            vx = kf.statePost[2][0]
                            vy = kf.statePost[3][0]

                            # Optionally apply damping to avoid runaway drift
                            vx *= decay_factor
                            vy *= decay_factor

                            # Reinitialize position but keep velocity
                            kf.statePost = np.array([[best_cx], [best_cy], [vx], [vy]], np.float32)

                            cv2.circle(warped, (pred_x, pred_y), 5, print_compl_color[name], -1)
                            cv2.putText(warped, f"KF PREDICTION for {name}: VELOCITY", (pred_x + 10, pred_y + 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, print_compl_color[name], 1)
                            print(f"[{name}] Reinitialized with preserved velocity. dist = {min_dist:.2f}")            
                    else:
                        detected_positions[name] = None
                else:
                    detected_positions[name] = None

            # Resize warped frame for display
            warped_display = cv2.resize(warped, (display_width, display_height), interpolation=cv2.INTER_AREA)

            # Show the warped (homography-transformed) camera view instead of the normal frame
            cv2.imshow('Color Tracking', warped_display)

            # MQTT sending logic
            now = time.time()
            if now - last_send_time >= SEND_INTERVAL:
                # Example: send green as "cat", blue as "mouse"
                cat_pixel = detected_positions.get("Green", {"x": None, "y": None})
                mouse_pixel = detected_positions.get("Blue", {"x": None, "y": None})
                cat = transform_to_play_area_meters(cat_pixel, play_center, play_radius, square_diagonal_m)
                mouse = transform_to_play_area_meters(mouse_pixel, play_center, play_radius, square_diagonal_m)
                message = {
                    "message": "coordinates",
                    "timestamp": int(now * 1000),
                    "cat": cat,
                    "mouse": mouse
                }
                json_message = json.dumps(message)
                client.publish(topic, json_message)
                # print(f"Published: {json_message}")
                last_send_time = now

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        client.disconnect()


if __name__ == "__main__":
    main()

