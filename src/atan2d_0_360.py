"""
Author: Ryan Gast
Date: 12/29/2023
Calculation of tan^-1(y/x) to lie in the range 0°to 360°. 
    Converted From Matlab to Python from: 
    "Orbital Mechanics for Engineering Students" (2013), 
    Appendix D, D.19.
Version: 1.0
Contact: ryan.d.gast@gmail.com
"""

import numpy as np

def atan2d_0_360(y, x):
    """
    Calculates the angle (in degrees) between the positive x-axis and the point (x, y) in the range [0, 360].

    Parameters:
    - y: The y-coordinate of the point.
    - x: The x-coordinate of the point.

    Returns:
    - t: The angle (in degrees) between the positive x-axis and the point (x, y) in the range [0, 360].
    """
    if x == 0:
        if y == 0:
            t = 0
        elif y > 0:
            t = 90
        else:
            t = 270
    elif x > 0:
        if y >= 0:
            t = np.degrees(np.arctan(y/x))
        else:
            t = np.degrees(np.arctan(y/x)) + 360
    elif x < 0:
        if y == 0:
            t = 180
        else:
            t = np.degrees(np.arctan(y/x)) + 180
    return t

