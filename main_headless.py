import cv2
import numpy as np
from color import Color
from matplotlib.colors import CSS4_COLORS
import matplotlib
import paho.mqtt.client as mqtt
import time
import json
import os

# Define expected area (in pixels)
EXPECTED_AREA = 150  # Adjust this value based on your use case

# Load calibration data
data = np.load("calibration_data.npz")
homography = data["homography"]
play_center = tuple(data["play_center"])
play_radius = int(data["play_radius"])
calibration_resolution = tuple(data["calibration_resolution"]) if "calibration_resolution" in data else None
square_diagonal_m = float(data["square_diagonal_m"]) if "square_diagonal_m" in data else None

# Print homography calibration settings before starting
print("=== Homography Calibration Settings ===")
print(f"homography:\n{homography}")
print(f"play_center: {play_center}")
print(f"play_radius: {play_radius}")
print(f"calibration_resolution: {calibration_resolution}")
print(f"square_diagonal_m: {square_diagonal_m}")
if "margin_m" in data:
    print(f"margin_m: {data['margin_m']}")
print("=======================================")

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
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def adaptive_detect_color(frame_hsv, adaptive_hsv_bounds):
    """Detect contours for a given color with adaptive HSV mask."""
    lower, upper = adaptive_hsv_bounds
    mask = cv2.inRange(frame_hsv, lower, upper)
    kernel = np.ones((2, 2), np.uint8)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
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
        if cv2.contourArea(contour) > 1:
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

SEND_FREQUENCY_HZ = 30  # Adjust how often data is sent (Hz)
SEND_INTERVAL = 1.0 / SEND_FREQUENCY_HZ
last_send_time = 0

COLOR_COLLECTION_FILE = "color_collection.npz"

def load_colors_for_condition(condition):
    if not os.path.exists(COLOR_COLLECTION_FILE):
        print("No color collection file found.")
        return []
    collection = dict(np.load(COLOR_COLLECTION_FILE, allow_pickle=True))
    colors = []
    for name, data in collection.items():
        v = data.item() if hasattr(data, "item") else data
        if v.get("condition", "") == condition:
            colors.append(Color(
                name=name,
                lower_bound=v["lower_bound"],
                upper_bound=v["upper_bound"],
                condition=v.get("condition", "")
            ))
    return colors

def select_condition(previous_condition=None):
    # Find all unique conditions in the color collection
    if not os.path.exists(COLOR_COLLECTION_FILE):
        print("No color collection file found.")
        return None
    collection = dict(np.load(COLOR_COLLECTION_FILE, allow_pickle=True))
    conditions = set()
    for v in collection.values():
        v = v.item() if hasattr(v, "item") else v
        cond = v.get("condition", "")
        if cond:
            conditions.add(cond)
    if not conditions:
        print("No conditions found in color collection.")
        return None
    print("Available conditions:")
    sorted_conditions = sorted(conditions)
    for idx, cond in enumerate(sorted_conditions):
        print(f"{idx+1}: {cond}")
    if previous_condition:
        print(f"Press Enter to use previous condition: '{previous_condition}'")
    while True:
        sel = input("Select condition by number: ")
        if sel.strip() == "" and previous_condition:
            return previous_condition
        try:
            sel_int = int(sel)
            if 1 <= sel_int <= len(sorted_conditions):
                return sorted_conditions[sel_int-1]
        except Exception:
            pass
        print("Invalid selection.")

def select_cat_mouse_names(color_names, prev_cat=None, prev_mouse=None):
    print("\nAvailable colors:")
    for idx, name in enumerate(color_names):
        print(f"{idx+1}: {name}")
    # Cat selection
    if prev_cat and prev_cat in color_names:
        print(f"Press Enter to use previous cat: '{prev_cat}'")
    while True:
        sel = input("Select cat color by number: ")
        if sel.strip() == "" and prev_cat and prev_cat in color_names:
            cat = prev_cat
            break
        try:
            sel_int = int(sel)
            if 1 <= sel_int <= len(color_names):
                cat = color_names[sel_int-1]
                break
        except Exception:
            pass
        print("Invalid selection.")
    # Mouse selection
    if prev_mouse and prev_mouse in color_names:
        print(f"Press Enter to use previous mouse: '{prev_mouse}'")
    while True:
        sel = input("Select mouse color by number: ")
        if sel.strip() == "" and prev_mouse and prev_mouse in color_names:
            mouse = prev_mouse
            break
        try:
            sel_int = int(sel)
            if 1 <= sel_int <= len(color_names):
                mouse = color_names[sel_int-1]
                if mouse == cat:
                    print("Mouse and cat cannot be the same color.")
                    continue
                break
        except Exception:
            pass
        print("Invalid selection.")
    return cat, mouse

def transform_to_play_area_meters(pos, play_center, play_radius, square_diagonal_m):
    """
    Transform pixel coordinates to play area coordinates in meters:
    - Center is (0,0)
    - Top of the frame is positive Y, right is positive X
    - Range is -diagonal/2 to diagonal/2 in both x and y (in meters)
    """
    if pos is None or pos["x"] is None or pos["y"] is None or square_diagonal_m is None:
        return {"x": None, "y": None}
    # Pixels per meter along the diagonal
    pixels_per_meter = (play_radius * 2) / square_diagonal_m
    # Shift to center, then convert to meters
    dx = (pos["x"] - play_center[0]) / pixels_per_meter
    dy = (play_center[1] - pos["y"]) / pixels_per_meter  # Y axis: top is positive
    return {
        "x": round(dx, 4),
        "y": round(dy, 4)
    }

def main():
    # --- Select condition and load colors ---
    previous_condition = None
    previous_cat = None
    previous_mouse = None
    condition = select_condition(previous_condition)
    if not condition:
        print("No valid condition selected. Exiting.")
        return
    previous_condition = condition
    all_tracked_colors = load_colors_for_condition(condition)
    if not all_tracked_colors:
        print(f"No colors found for condition '{condition}'. Exiting.")
        return

    # --- Load camera settings from color collection ---
    if os.path.exists(COLOR_COLLECTION_FILE):
        collection = dict(np.load(COLOR_COLLECTION_FILE, allow_pickle=True))
        # Find the first color with matching condition to get settings
        for data in collection.values():
            v = data.item() if hasattr(data, "item") else data
            if v.get("condition", "") == condition:
                exposure = v.get("exposure", -4)
                wb_temp = v.get("wb_temp", 4500)
                webcam_name = v.get("webcam", "Unknown")
                focus = v.get("focus", 0)
                print(f"Using camera settings from color '{v.get('name', 'Unknown Color')}':")
                print(f"  Webcam: {webcam_name}")
                print(f"  Exposure: {exposure}")
                print(f"  White Balance: {wb_temp}")
                print(f"  Focus: {focus}")
                break
        else:
            exposure = -4
            wb_temp = 4500
            focus = 0
            print("No matching color found in collection. Using default camera settings.")
    else:
        exposure = -4
        wb_temp = 4500
        focus = 0
        print("No color collection file found. Using default camera settings.")

    color_names = [c.get_name() for c in all_tracked_colors]
    cat_name, mouse_name = select_cat_mouse_names(color_names, previous_cat, previous_mouse)
    previous_cat = cat_name
    previous_mouse = mouse_name

    cap = cv2.VideoCapture(0)
        ## START LONG WAITING TIME
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    # cap.set(cv2.CAP_PROP_FPS, 60)
        ## END LONG WAITING TIME
    #cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Manual mode
    #cap.set(cv2.CAP_PROP_EXPOSURE, exposure)         # Adjust this based on trial
    #cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    #cap.set(cv2.CAP_PROP_WB_TEMPERATURE, wb_temp) # Between 4000–6000 K for neutral
    #cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    #cap.set(cv2.CAP_PROP_FOCUS, focus)

    ## DEBUG ## Color print in BGR
    print_color = {color.get_name(): (0, 255, 0) for color in all_tracked_colors}
    print_compl_color = {color.get_name(): (0, 0, 255) for color in all_tracked_colors}

    ## PARAMETER for Detection AND prediction 
    decay_factor = 0.9
    contour_threshold_list = {color.get_name(): 2 for color in all_tracked_colors}
    reset_threshold_list = {color.get_name(): 4 for color in all_tracked_colors}
    ## Adaptive HSV parameters
    adaptive_hsv_bounds = {color.get_name(): None for color in all_tracked_colors}
    previous_hsv = {color.get_name(): None for color in all_tracked_colors}
    ## Kalman Filter param
    kalman_filters = {}
    initialized_flags = {}
    for color in all_tracked_colors:
        kalman_filters[color.get_name()] = initialize_kalman_filter()
        initialized_flags[color.get_name()] = False

    display_width = 960
    display_height = 960

    # MQTT settings
    broker = "broker.hivemq.com"
    port = 1883
    topic = "XRCatAndMouse/1112"
    # topic = "catmouse/coordinates"
    client = mqtt.Client()
    client.connect(broker, port, 60)

    global last_send_time

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if calibration_resolution is not None:
                frame = cv2.resize(frame, calibration_resolution, interpolation=cv2.INTER_AREA)

            # # Apply homography transformation
            warped = cv2.warpPerspective(frame, homography, (frame.shape[1], frame.shape[0]))

            # # Use the warped frame for color detection
            # frame_hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)

            # Apply LAB equalization before converting to HSV
            lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_eq = clahe.apply(l)
            lab_eq = cv2.merge([l_eq, a, b])
            frame_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
            frame_hsv = cv2.cvtColor(frame_eq, cv2.COLOR_BGR2HSV)


            detected_positions = {}

            for color in all_tracked_colors:
                name = color.get_name()
                kf = kalman_filters[name]
                initialized = initialized_flags[name]

                contour_threshold = contour_threshold_list[name]
                reset_threshold = reset_threshold_list[name]
                best_cx, best_cy = None, None
                min_dist = float('inf')

                if adaptive_hsv_bounds[name] is not None:
                    ## Adaptive HSV
                    contours = adaptive_detect_color(frame_hsv, adaptive_hsv_bounds[name])                
                    # contours = detect_color(frame_hsv, color)
                else:
                    ## Predefined HSV
                    contours = detect_color(frame_hsv, color)

                prediction = kf.predict()
                pred_x, pred_y = int(prediction[0]), int(prediction[1])

                if contours:
                    min_dist, best_cx, best_cy = get_closest_blob_to_prediction(contours, pred_x, pred_y)
                    if best_cx is not None and best_cy is not None:
                        detected_positions[name] = {"x": float(best_cx), "y": float(best_cy)}
                        if not initialized:
                            kf.statePost = np.array([[best_cx], [best_cy], [0], [0]], np.float32)
                            initialized = True

                        if min_dist < contour_threshold:
                            measurement = np.array([[best_cx], [best_cy]], np.float32)
                            est = kf.correct(measurement)
                            est_x, est_y = int(est[0]), int(est[1])
                            '''
                            cv2.circle(warped, (est_x, est_y), 5, print_color[name], -1)
                            # cv2.putText(warped, f"Measuremnt for {name}: ({est_x},{est_y})", (est_x + 10, est_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, print_color[name], 1)
                            '''

                            # Adaptive HSV sampling
                            sample_size = 4
                            cx, cy = int(best_cx), int(best_cy)
                            patch = frame_hsv[max(0, cy-sample_size):cy+sample_size, max(0, cx-sample_size):cx+sample_size]
                            if patch.size > 0:
                                avg_hsv = patch.mean(axis=(0, 1)).astype(np.int16)
                                H, S, V = avg_hsv
                                # Compare with previous hsv
                                prev_hsv = previous_hsv[name]
                                if prev_hsv is None or (
                                        abs(int(H) - int(prev_hsv[0])) < 20 and
                                        abs(int(S) - int(prev_hsv[1])) < 25 and
                                        abs(int(V) - int(prev_hsv[2])) < 20
                                ):
                                    lower = np.array([max(H - 5, 0), max(S - 20, 0), max(V - 20, 0)]).astype(np.uint8)
                                    upper = np.array([min(H + 5, 179), min(S + 20, 255), min(V + 20, 255)]).astype(
                                        np.uint8)
                                    H = np.uint8(H)
                                    S = np.uint8(S)
                                    V = np.uint8(V)
                                    adaptive_hsv_bounds[name] = (lower, upper)
                                    previous_hsv[name] = (H, S, V)

                            ## DEBUG HSV adaptive cap
                                    # if prev_hsv is not None:
                                    #     Xh = abs(int(H) - int(prev_hsv[0]))
                                    #     Xs = abs(int(S) - int(prev_hsv[1]))
                                    #     Xv = abs(int(V) - int(prev_hsv[2]))
                                    #     print(f"{name} H: {Xh} S: {Xs}  V:  {Xv}" )

                        elif min_dist < reset_threshold:
                            print(f"KF PREDICTION for {name}: bad measurement", (pred_x + 10, pred_y + 20),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, print_compl_color[name], 1)
                            '''cv2.circle(warped, (pred_x, pred_y), 5, print_compl_color[name], -1)
                            # cv2.putText(warped, f"KF PREDICTION for {name}: bad measurement", (pred_x + 10, pred_y + 20),
                            #            cv2.FONT_HERSHEY_SIMPLEX, 0.4, print_compl_color[name], 1)
                            '''
                        else:
                            vx = kf.statePost[2][0]
                            vy = kf.statePost[3][0]
                            vx *= decay_factor
                            vy *= decay_factor
                            kf.statePost = np.array([[best_cx], [best_cy], [vx], [vy]], np.float32)
                            '''cv2.circle(warped, (pred_x, pred_y), 5, print_compl_color[name], -1)
                            # cv2.putText(warped, f"KF PREDICTION for {name}: VELOCITY", (pred_x + 10, pred_y + 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, print_compl_color[name], 1)
                            # print(f"[{name}] Reinitialized with preserved velocity. dist = {min_dist:.2f}")   
                            '''
                    else:
                        detected_positions[name] = None
                else:
                    detected_positions[name] = None                        

            # warped_display = cv2.resize(warped, (display_width, display_height), interpolation=cv2.INTER_AREA)
            # cv2.imshow('Color Tracking', warped_display)

            now = time.time()
            if now - last_send_time >= SEND_INTERVAL:
                # Use selected cat and mouse names
                cat_pixel = detected_positions.get(cat_name, {"x": None, "y": None})
                mouse_pixel = detected_positions.get(mouse_name, {"x": None, "y": None})
                cat = transform_to_play_area_meters(cat_pixel, play_center, play_radius, square_diagonal_m)
                mouse = transform_to_play_area_meters(mouse_pixel, play_center, play_radius, square_diagonal_m)
                # Apply rounding as in mqtt_example
                message = {
                    "message": "coordinates",
                    "timestamp": round(now * 1000, 3),
                    "cat": {
                        "x": round(cat["x"], 1) if cat["x"] is not None else None,
                        "y": round(cat["y"], 1) if cat["y"] is not None else None
                    },
                    "mouse": {
                        "x": round(mouse["x"], 1) if mouse["x"] is not None else None,
                        "y": round(mouse["y"], 1) if mouse["y"] is not None else None
                    }
                }
                json_message = json.dumps(message)
                client.publish(topic, json_message)
                print(f"Sent data: {json_message}")  # Show sent data in the console
                last_send_time = now

            #if cv2.waitKey(1) & 0xFF == ord('q'):
            #    break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        client.disconnect()

if __name__ == "__main__":
    main()

