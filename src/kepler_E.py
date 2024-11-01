"""
Author: Ryan Gast
Date: 12/27/2023
Computes solutions of Keplers equation by Newtons method.
    Converted From Matlab to Python from: 
    "Orbital Mechanics for Engineering Students" (2013), 
    Appendix D, D.11.
Version: 1.0
Contact: ryan.d.gast@gmail.com
"""

import math

def kepler_E(e, M):
    """
    Solve Kepler's equation for eccentric anomaly (E) given eccentricity (e) and mean anomaly (M).

    Parameters:
    e (float): Eccentricity of the orbit.
    M (float): Mean anomaly.

    Returns:
    float: Eccentric anomaly (E).
    """
    # Set an error tolerance:
    error = 1.e-8

    # Select a starting value for E:
    if M < math.pi:
        E = M + e/2
    else:
        E = M - e/2

    # Iterate on Equation 3.17 until E is determined to within
    # the error tolerance:
    ratio = 1
    while abs(ratio) > error:
        ratio = (E - e*math.sin(E) - M)/(1 - e*math.cos(E))
        E = E - ratio

    return E

if __name__ == '__main__':
    e = 1.5
    M = 1.0
    print(kepler_E(e, M))