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
    raise ValueError(f"Color '{color_name}' not found in CSS4_COLORS")


def detect_color_cuda(gpu_hsv, color):
    """CUDA-accelerated color detection pipeline."""
    if not hasattr(color, '_cuda_morph'):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        color._cuda_morph = cv2.cuda.createMorphologyFilter(
            cv2.MORPH_CLOSE, cv2.CV_8UC1, kernel)

    # Convert bounds to integer tuples
    lower = tuple(int(x) for x in color.get_lower_bound())
    upper = tuple(int(x) for x in color.get_upper_bound())

    # Perform inRange with scalar tuples
    gpu_mask = cv2.cuda.inRange(gpu_hsv, lower, upper)

    # Apply morphology
    gpu_mask_closed = color._cuda_morph.apply(gpu_mask)
    return color._cuda_morph.apply(gpu_mask_closed)


def get_closest_blob(contours, previous_position, expected_area):
    """RTX-optimized blob selection with error handling."""
    if not contours:
        return None

    try:
        if previous_position is not None:
            return min(contours, key=lambda c:
            np.sqrt((cv2.moments(c)['m10'] / cv2.moments(c)['m00'] - previous_position[0]) ** 2 +
                    (cv2.moments(c)['m01'] / cv2.moments(c)['m00'] - previous_position[1]) ** 2))
        return min(contours, key=lambda c: abs(cv2.contourArea(c) - expected_area))
    except:
        return None

def main():
    # Verify CUDA compatibility for sm_86
    if cv2.cuda.getCudaEnabledDeviceCount() == 0:
        raise RuntimeError("No CUDA device detected")

    # Initialize video capture with GStreamer backend fallback
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)  # Force V4L2 for Linux
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2160)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
    cap.set(cv2.CAP_PROP_FPS, 60)

    # CUDA resources with sm_86 optimization
    stream = cv2.cuda.Stream()
    gpu_frame = cv2.cuda.GpuMat()

    # Warm-up CUDA context
    dummy = cv2.cuda.GpuMat()
    dummy.upload(np.zeros((2160, 2160, 3), dtype=np.uint8))

    tracked_colors = [green, purple, yellow, blue, white]
    positions = {color.get_name(): None for color in tracked_colors}

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Async upload and processing
            gpu_frame.upload(frame, stream)
            gpu_hsv = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2HSV, stream=stream)
            stream.waitForCompletion()

            for color in tracked_colors:
                gpu_mask = detect_color_cuda(gpu_hsv, color)
                mask = gpu_mask.download()

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                blob = get_closest_blob(contours, positions[color.get_name()], EXPECTED_AREA)

                if blob is not None:  # First check for None
                    M = cv2.moments(blob)
                    if M["m00"] > 0:
                        cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                        positions[color.get_name()] = (cx, cy)
                        cv2.circle(frame, (cx, cy), 5, (255, 255, 255), -1)
                        cv2.putText(frame, f"{color.get_name()} ({cx}, {cy})",
                                (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            cv2.imshow('CUDA Colour Tracking', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()