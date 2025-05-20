from color import Color
import numpy as np

# Define color ranges using the Color class
green = Color("Green", np.array([50, 70, 107]), np.array([81, 132, 255]))
purple = Color("Purple", np.array([126, 141, 87]), np.array([146, 241, 187]))
yellow = Color("Yellow", np.array([16, 160, 200]), np.array([31, 230, 255]))
blue = Color("Blue", np.array([80, 200, 130]), np.array([120, 255, 190]))
white = Color("White", np.array([20, 20, 205]), np.array([40, 40, 235]))
marker = Color("White", np.array([0, 0, 150]), np.array([180, 30, 220]))
test_marker = Color("Blue", np.array([100, 145, 50]), np.array([130, 175, 160]))