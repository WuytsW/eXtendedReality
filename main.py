import cv2
import numpy as np
from color_ranges import green, purple, yellow, blue, white
from matplotlib.colors import CSS4_COLORS
import matplotlib

# Define expected area (in pixels)
EXPECTED_AREA = 150  # Adjust this value based on your use case

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

def get_closest_blob_to_position(contours, previous_position):
    """Find the blob closest to the previous position."""
    closest_blob = None
    min_distance = float('inf')
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            distance = np.sqrt((cx - previous_position[0])**2 + (cy - previous_position[1])**2)
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

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 60)

    all_tracked_colors = [green, purple, yellow, blue, white]
    previous_positions = {color.get_name(): None for color in all_tracked_colors}

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            for color in all_tracked_colors:
                contours = detect_color(frame_hsv, color)
                previous_position = previous_positions[color.get_name()]
                if previous_position is not None:
                    closest_blob = get_closest_blob_to_position(contours, previous_position)
                else:
                    closest_blob = get_closest_blob_to_area(contours, EXPECTED_AREA)

                if closest_blob is not None:
                    centroid = get_blob_centroid(closest_blob)
                    if centroid:
                        cx, cy = centroid
                        previous_positions[color.get_name()] = (cx, cy)

                        # Draw a white dot
                        cv2.circle(frame, (cx, cy), 5, (255, 255, 255), -1)

                        # Use black for the text
                        cv2.putText(frame, f"{color.get_name()} ({cx}, {cy})", (cx + 10, cy),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            cv2.imshow('Color Tracking', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()