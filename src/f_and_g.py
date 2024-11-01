"""
Author: Ryan Gast
Date: 12/28/2023
Calculation of the Lagrange coefficients f and g and their
time derivatives in terms of change in universal anomaly.
    Converted From Matlab to Python from: 
    "Orbital Mechanics for Engineering Students" (2013), 
    Appendix D, D.15.
Version: 1.0
Contact: ryan.d.gast@gmail.com
"""

import math
from stumpSC import stumpS, stumpC

def f_and_g(x, t, ro, a, mu=398600):
    """
    Calculate the values of f and g based on the given parameters.

    Parameters:
    x (float): The value of x.
    t (float): The value of t.
    ro (float): The value of ro.
    a (float): The value of a.

    Returns:
    tuple: A tuple containing the values of f and g.
    """
    z = a*x**2
    # Equation 3.69a:
    f = 1 - x**2/ro*stumpC(z)
    # Equation 3.69b:
    g = t - 1/math.sqrt(mu)*x**3*stumpS(z)
    return f, g

if __name__ == '__main__':
    x = 0.5
    t = 0.5
    ro = 1.0
    a = 1.0
    mu = 1.0
    f, g = f_and_g(x, t, ro, a, mu)
    print(f, g)