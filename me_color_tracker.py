import cv2
import numpy as np
import time

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    last_print_time = 0  # Track the last time the HSV value was printed

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to HSV
            frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Define circular ROI
            height, width, _ = frame.shape
            center_x, center_y = width // 2, height // 2
            radius = 20
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.circle(mask, (center_x, center_y), radius, 255, -1)

            # Extract HSV values within the circle
            hsv_values = frame_hsv[mask == 255]
            avg_hsv = np.mean(hsv_values, axis=0)

            # Print average HSV value once per second
            current_time = time.time()
            if current_time - last_print_time >= 1:
                print(f"Average HSV: {avg_hsv}")
                last_print_time = current_time

            # Draw the circle on the frame
            cv2.circle(frame, (center_x, center_y), radius, (0, 255, 0), 2)

            # Display the frame
            cv2.imshow("Calibration Tool", frame)

            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()