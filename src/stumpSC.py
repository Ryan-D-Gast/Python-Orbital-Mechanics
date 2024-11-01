"""
Author: Ryan Gast
Date: 12/27/2023
Calculates the Stumpff function S(z) and C(z)
    Converted From Matlab to Python from: 
    "Orbital Mechanics for Engineering Students" (2013), 
    Appendix D, D.13.
Version: 1.0
Contact: ryan.d.gast@gmail.com
"""

import math

def stumpS(z):
    """
    Calculate the value of the Stumpff function S(z) for a given input z.

    Parameters:
    z (float): The input value for the Stumpff function.

    Returns:
    float: The value of the Stumpff function S(z).
    """
    if z > 0:
        s = (math.sqrt(z) - math.sin(math.sqrt(z)))/(math.sqrt(z))**3
    elif z < 0:
        s = (math.sinh(math.sqrt(-z)) - math.sqrt(-z))/(math.sqrt(-z))**3
    else:
        s = 1/6
    return s

def stumpC(z):
    """
    Calculate the value of the Stumpff function C(z) for a given input z.

    Parameters:
    z (float): The input value.

    Returns:
    float: The value of the Stumpff function C(z).
    """
    if z > 0:
        c = (1 - math.cos(math.sqrt(z)))/z
    elif z < 0:
        c = (math.cosh(math.sqrt(-z)) - 1)/(-z)
    else:
        c = 1/2
    return c

# test
if __name__ == "__main__":
    z = -1.0
    s = stumpS(z)
    c = stumpC(z)
    print(s, c)
    