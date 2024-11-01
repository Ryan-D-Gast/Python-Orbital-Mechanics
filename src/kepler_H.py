"""
Author: Ryan Gast
Date: 12/27/2023
Computes solution of Keplers equation for the hyperbola using
Newtons method.
    Converted From Matlab to Python from: 
    "Orbital Mechanics for Engineering Students" (2013), 
    Appendix D, D.12.
Version: 1.0
Contact: ryan.d.gast@gmail.com
"""

import math

def kepler_H(e, M):
    """
    Solve Kepler's equation for the hyperbolic case using the Newton-Raphson method.

    Parameters:
    e (float): Eccentricity of the orbit.
    M (float): Mean anomaly.

    Returns:
    float: True anomaly.
    """
    # Set an error tolerance:
    error = 1.e-8

    # Starting value for F:
    F = M

    # Iterate on Equation 3.45 until F is determined to within
    # the error tolerance:
    ratio = 1
    while abs(ratio) > error:
        ratio = (e*math.sinh(F) - F - M)/(e*math.cosh(F) - 1)
        F = F - ratio

    return F

if __name__ == '__main__':
    e = 0.5
    M = 1.0
    print(kepler_H(e, M))