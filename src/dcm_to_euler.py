"""
Author: Ryan Gast
Date: 12/28/2023
Obtains the classical Euler angle sequence from a DCM.
    Converted From Matlab to Python from: 
    "Orbital Mechanics for Engineering Students" (2013), 
    Appendix D, D.20.
Version: 1.0
Contact: ryan.d.gast@gmail.com
"""

import numpy as np
from atan2d_0_360 import atan2d_0_360

def dcm_to_euler(Q):
    """
    Convert a Direction Cosine Matrix (DCM) to Euler angles.

    Parameters:
    Q (numpy.ndarray): The 3x3 DCM matrix.

    Returns:
    tuple: A tuple containing the Euler angles (alpha, beta, gamma).
    """
    print(Q)
    alpha = atan2d_0_360(Q[2, 0], -Q[2, 1])
    beta = np.degrees(np.arccos(Q[2, 2]))
    gamma = atan2d_0_360(Q[0, 2], Q[1, 2])
    return alpha, beta, gamma

# Test the function
if __name__ == '__main__':
    # wild input example
    Q3 = np.array([[0.5, 0.5, 0.7071], [0.5, 0.5, -0.7071], [-0.7071, 0.7071, 0]])
    alpha3, beta3, gamma3 = dcm_to_euler(Q3)
    print(f'alpha: {alpha3}, beta: {beta3}, gamma: {gamma3}')
    Q4 = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    alpha4, beta4, gamma4 = dcm_to_euler(Q4)
    print(f'alpha: {alpha4}, beta: {beta4}, gamma: {gamma4}')