from color import Color
import numpy as np

# Define color ranges using the Color class
green = Color("Green", np.array([50, 70, 107]), np.array([81, 132, 255]), "atrium_sunny")
purple = Color("Purple", np.array([126, 141, 87]), np.array([146, 241, 187]), "atrium_sunny")
yellow = Color("Yellow", np.array([16, 160, 200]), np.array([31, 230, 255]), "atrium_sunny")
blue = Color("Blue", np.array([80, 200, 130]), np.array([120, 255, 190]), "atrium_sunny")
white = Color("White", np.array([20, 20, 205]), np.array([40, 40, 235]), "atrium_sunny")
marker = Color("White", np.array([0, 0, 150]), np.array([180, 30, 220]), "atrium_sunny")
test_marker = Color("Blue", np.array([100, 145, 50]), np.array([130, 175, 160]), "atrium_sunny")