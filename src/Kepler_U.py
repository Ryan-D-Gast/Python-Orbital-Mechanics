"""
Author: Ryan Gast
Date: 12/27/2023
Computes  Solution of the universal Keplers equation using 
Newtons method.
    Converted From Matlab to Python from: 
    "Orbital Mechanics for Engineering Students" (2013), 
    Appendix D, D.14.
Version: 1.0
Contact: ryan.d.gast@gmail.com
"""

import math
from stumpSC import stumpS
from stumpSC import stumpC

def kepler_U(dt, ro, vro, a, mu=398600):
    """
    Solve Kepler's equation using the Universal Variable formulation.

    Parameters:
    dt (float): Time since periapsis passage [s]
    ro (float): Initial position magnitude [km]
    vro (float): Initial velocity magnitude [km/s]
    a (float): Semi-major axis of the orbit [km]
    (Universal Kepler's requires the reciprocal of semimajor axis)

    Returns:
    float: Value of x that solves Kepler's equation

    Raises:
    None

    """
    # Set an error tolerance and a limit on the number of iterations:
    error = 1.e-8
    nMax = 1000

    # Starting value for x:
    x = math.sqrt(mu)*abs(a)*dt

    # Iterate on Equation 3.65 until until convergence occurs within
    # the error tolerance:
    n = 0
    ratio = 1
    while abs(ratio) > error and n <= nMax:
        n += 1
        C = stumpC(a*x**2)
        S = stumpS(a*x**2)
        F = ro*vro/math.sqrt(mu) * x**2*C + (1 - a*ro)*x**3*S + ro*x - math.sqrt(mu)*dt
        dFdx = ro*vro/math.sqrt(mu)*x*(1 - a*x**2*S) + (1 - a*ro)*x**2*C + ro
        ratio = F/dFdx
        x = x - ratio

    # Deliver a value for x, but report that nMax was reached:
    if n > nMax:
        print(f'\n **No. iterations of Kepler\'s equation = {n}')
        print(f'\n F/dFdx = {F/dFdx}')

    return x

# example usage
if __name__ == '__main__':
    ro = 10000
    vro = 3.0752
    dt = 3600
    a = -19655
    test = kepler_U(dt, ro, vro, 1/a)
    print(test)