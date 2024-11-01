"""
Author: Ryan Gast
Date: 12/28/2023
Calculates the Lagrange coefficient derivatives 
f and g in terms of change in true anomaly.
    Converted From Matlab to Python from: 
    "Orbital Mechanics for Engineering Students" (2013), 
    Appendix D, D.15.
Version: 1.0
Contact: ryan.d.gast@gmail.com
"""

import math
from stumpSC import stumpS, stumpC

def fDot_and_gDot(x, r, ro, a, mu=398600):
    """
    Calculate the derivatives fdot and gdot for the given parameters.

    Parameters:
    x (float): Independent variable.
    r (float): Distance from the central body.
    ro (float): Initial distance from the central body.
    a (float): Semi-major axis of the orbit.

    Returns:
    tuple: A tuple containing the derivatives fdot and gdot.
    """
    z = a * x ** 2
    # Equation 3.69c:
    fdot = math.sqrt(mu) / r / ro * (z * stumpS(z) - 1) * x
    # Equation 3.69d:
    gdot = 1 - x ** 2 / r * stumpC(z)
    return fdot, gdot


# Test
if __name__ == '__main__':
    x = 0.5
    r = 0.5
    ro = 1.0
    a = 1.0
    mu = 1.0
    fdot, gdot = fDot_and_gDot(x, r, ro, a)
    print(fdot, gdot)