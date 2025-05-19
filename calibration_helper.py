import cv2
import numpy as np
from sympy.physics.units import length


def calibrate_playing_area(points, square_side_meters, margin_meters):
    """
    Takes in 4 points that form a square (any rotation) and returns
    the center and radius of a circle encompassing the square minus margin.
    - points: list of 4 (x, y) tuples in pixel coordinates
    - square_side_meters: real-world side length (meters)
    - margin_meters: margin to subtract from the radius (meters)
    Returns: (center_x, center_y), radius (in pixels)
    """
    # All calculations are done in the original camera resolution.
    pts = np.array(points)

    # Calculate average side length in pixels
    side_lengths = [
        np.linalg.norm(pts[i] - pts[(i + 1) % 4]) for i in range(4)
    ]
    avg_side = np.mean(side_lengths)

    # Pixels per meter
    pixels_per_meter = avg_side / square_side_meters

    # Calculate diagonal in pixels
    diagonal_pixels = avg_side * (2 ** 0.5)


    radius_pixels = diagonal_pixels / 2 - margin_meters * pixels_per_meter

    radius_pixels = max(0, radius_pixels)  # Ensure radius is never below 0

    center_x_pixels = int(np.mean(pts[:, 0]))
    center_y_pixels = int(np.mean(pts[:, 1]))

    return (center_x_pixels, center_y_pixels), int(radius_pixels)

def draw_playing_area(warped_frame, center, radius):
    """Draws a circular playing area on the frame."""
    # Calculate the center and radius of the circle

    # Draw the circle on the frame
    cv2.circle(warped_frame, center, radius, (255, 0, 0), 2)

def hsv_to_hsv_range(hsv_value,lower_margin,upper_margin):
    """
    Takes in an HSV value and returns the lower and upper bounds for color detection.
    - hsv_value: tuple of (H, S, V) in 0-255 range
    - lower_margin: margin to subtract from each channel
    - upper_margin: margin to add to each channel
    Returns: (lower_bound, upper_bound)
    """
    lower_bound = np.array([max(0, hsv_value[0] - lower_margin), max(0, hsv_value[1] - lower_margin), max(0, hsv_value[2] - lower_margin)])
    upper_bound = np.array([min(179, hsv_value[0] + upper_margin), min(255, hsv_value[1] + upper_margin), min(255, hsv_value[2] + upper_margin)])

    return lower_bound, upper_bound

def rotate_frame(frame, angle):
    """
    Rotates the frame by a given angle.
    - frame: the input frame
    - angle: the angle to rotate (in degrees)
    Returns: the rotated frame
    """
    # Get the center of the frame
    center = (frame.shape[1] // 2, frame.shape[0] // 2)

    # Get the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Rotate the frame
    rotated_frame = cv2.warpAffine(frame, rotation_matrix, (frame.shape[1], frame.shape[0]))

    return rotated_frame

def homography_transform(frame, homography_matrix):
    """
    Applies a homography transformation to the frame.
    - frame: the input frame
    - homography_matrix: the homography matrix
    Returns: the transformed frame
    """
    # Get the dimensions of the frame
    h, w = frame.shape[:2]

    # Apply the homography transformation
    transformed_frame = cv2.warpPerspective(frame, homography_matrix, (w, h))

    return transformed_frame

def get_hsv_at_pixel(frame, x, y):
    """
    Gets the HSV value at the pixel position.
    - frame: the input frame
    - x: x-coordinate of the cursor
    - y: y-coordinate of the cursor
    Returns: the HSV value at the cursor position
    """
    # Get the pixel value at the cursor position
    pixel_value = frame[y, x]

    # Convert BGR to HSV
    hsv_value = cv2.cvtColor(np.uint8([[pixel_value]]), cv2.COLOR_BGR2HSV)[0][0]

    return hsv_value

def generate_square_points(center, side_length,angle):
    """
    Generates 4 points that form a square around the center point.
    - center: tuple of (x, y) coordinates of the center
    - side_length: length of the sides of the square
    - angle: angle to rotate the square (in degrees)
    Returns: list of 4 points [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    """
    # Calculate half side length
    half_side = side_length / 2

    # Calculate the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)

    # Generate the 4 points of the square
    points = [
        (-half_side, -half_side),
        (half_side, -half_side),
        (half_side, half_side),
        (-half_side, half_side)
    ]

    # Rotate and translate the points
    rotated_points = []
    for point in points:
        x_rotated = int(point[0] * rotation_matrix[0][0] + point[1] * rotation_matrix[0][1] + center[0])
        y_rotated = int(point[0] * rotation_matrix[1][0] + point[1] * rotation_matrix[1][1] + center[1])
        rotated_points.append((x_rotated, y_rotated))

    return rotated_points


