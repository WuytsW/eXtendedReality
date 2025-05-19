import time

import cv2
import numpy as np
from color_ranges import green, purple, yellow, blue, white
import matplotlib

# Define expected area (in pixels)
EXPECTED_AREA = 150  # Adjust this value based on your use case

def detect_color_cuda(gpu_hsv, color):
    """CUDA-accelerated color detection pipeline."""
    if not hasattr(color, '_cuda_morph'):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        color._cuda_morph = cv2.cuda.createMorphologyFilter(
            cv2.MORPH_CLOSE, cv2.CV_8UC1, kernel)

    lower = tuple(int(x) for x in color.get_lower_bound())
    upper = tuple(int(x) for x in color.get_upper_bound())
    gpu_mask = cv2.cuda.inRange(gpu_hsv, lower, upper)
    gpu_mask_closed = color._cuda_morph.apply(gpu_mask)
    return color._cuda_morph.apply(gpu_mask_closed)

def get_closest_blob(contours, previous_position, expected_area):
    """Blob selection with error handling."""
    if not contours:
        return None
    try:
        if previous_position is not None:
            return min(contours, key=lambda c:
                np.sqrt((cv2.moments(c)['m10']/cv2.moments(c)['m00'] - previous_position[0])**2 +
                        (cv2.moments(c)['m01']/cv2.moments(c)['m00'] - previous_position[1])**2))
        return min(contours, key=lambda c: abs(cv2.contourArea(c) - expected_area))
    except:
        return None

def main():
    if cv2.cuda.getCudaEnabledDeviceCount() == 0:
        raise RuntimeError("No CUDA device detected")

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2160)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
    cap.set(cv2.CAP_PROP_FPS, 60)

    stream = cv2.cuda.Stream()
    gpu_frame = cv2.cuda.GpuMat()
    tracked_colors = [green, purple, yellow, blue, white]
    positions = {color.get_name(): None for color in tracked_colors}

    # Frame timing variables
    frame_count = 0
    start_time = time.perf_counter()

    try:
        while True:
            frame_start = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                break

            gpu_frame.upload(frame, stream)
            gpu_hsv = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2HSV, stream=stream)
            stream.waitForCompletion()

            results = {}
            for color in tracked_colors:
                gpu_mask = detect_color_cuda(gpu_hsv, color)
                mask = gpu_mask.download()
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                blob = get_closest_blob(contours, positions[color.get_name()], EXPECTED_AREA)

                if blob is not None:
                    M = cv2.moments(blob)
                    if M["m00"] > 0:
                        cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                        positions[color.get_name()] = (cx, cy)
                        results[color.get_name()] = (cx, cy)

            # Calculate timing metrics
            processing_time = (time.perf_counter() - frame_start) * 1000  # Convert to ms
            frame_count += 1

            # Print results with timing
            print(f"Frame {frame_count}:")
            for color, coords in results.items():
                print(f"  {color}: {coords}")
            print(f"  Processing time: {processing_time:.2f}ms")
            print(f"  Estimated FPS: {1000 / processing_time:.1f}" if processing_time > 0 else "")


    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        cap.release()

if __name__ == "__main__":
    main()