"""
Author: Ryan Gast
Date: 12/28/2023
Obtains the yaw, pitch, and roll angles from a DCM.
    Converted From Matlab to Python from: 
    "Orbital Mechanics for Engineering Students" (2013), 
    Appendix D, D.21.
Version: 1.0
Contact: ryan.d.gast@gmail.com
"""

import numpy as np
from atan2d_0_360 import atan2d_0_360

def dcm_to_ypr(Q):
    """
    Convert a Direction Cosine Matrix (DCM) to yaw, pitch, and roll angles.

    Parameters:
    Q (numpy.ndarray): The Direction Cosine Matrix (DCM) as a 3x3 numpy array.

    Returns:
    tuple: A tuple containing the yaw, pitch, and roll angles in degrees.
    """
    yaw = atan2d_0_360(Q[0, 1], Q[0, 0])
    pitch = np.degrees(np.arcsin(-Q[0, 2]))
    roll = atan2d_0_360(Q[1, 2], Q[2, 2])
    return yaw, pitch, roll

# Test the function
if __name__ == '__main__':
    # wild input example
    Q3 = np.array([[0.5, 0.5, 0.7071], [0.5, 0.5, -0.7071], [-0.7071, 0.7071, 0]])
    yaw3, pitch3, roll3 = dcm_to_ypr(Q3)
    print(f'yaw: {yaw3}, pitch: {pitch3}, roll: {roll3}')
    Q4 = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    yaw4, pitch4, roll4 = dcm_to_ypr(Q4)
    print(f'yaw: {yaw4}, pitch: {pitch4}, roll: {roll4}')