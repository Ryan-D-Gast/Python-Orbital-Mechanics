"""
Author: Ryan Gast
Date: 12/27/2023
Calculation of the state vector given the initial state vector
and the time lapse delta t.
    Converted From Matlab to Python from: 
    "Orbital Mechanics for Engineering Students" (2013), 
    Appendix D, D.16.
Version: 1.0
Contact: ryan.d.gast@gmail.com
"""

import numpy as np
from Kepler_U import kepler_U
from f_and_g import f_and_g
from fDot_and_gDot import fDot_and_gDot

def rv_from_r0v0(R0, V0, t, mu=398600):
    """
    Compute the position vector R and velocity vector V at a given time t,
    given the initial position vector R0, initial velocity vector V0, and
    the time of interest.

    Parameters:
    R0 (numpy.ndarray): Initial position vector.
    V0 (numpy.ndarray): Initial velocity vector.
    t (float): Time of interest.

    Returns:
    R (numpy.ndarray): Final position vector at time t.
    V (numpy.ndarray): Final velocity vector at time t.
    """
    # Magnitudes of R0 and V0:
    r0 = np.linalg.norm(R0)
    v0 = np.linalg.norm(V0)
    # Initial radial velocity:
    vr0 = np.dot(R0, V0)/r0
    # Reciprocal of the semimajor axis (from the energy equation):
    alpha = 2/r0 - v0**2/mu
    # Compute the universal anomaly:
    x = kepler_U(t, r0, vr0, alpha)
    # Compute the f and g functions:
    f, g = f_and_g(x, t, r0, alpha)
    # Compute the final position vector:
    R = f*R0 + g*V0
    # Compute the magnitude of R:
    r = np.linalg.norm(R)
    # Compute the derivatives of f and g:
    fdot, gdot = fDot_and_gDot(x, r, r0, alpha)
    # Compute the final velocity:
    V = fdot*R0 + gdot*V0
    return R, V

# example usage
if __name__ == "__main__":
    R0 = np.array([7000, -12124, 0])
    V0 = np.array([2.6679, 4.6210, 0])
    t = 3600

    R, V = rv_from_r0v0(R0, V0, t)
    print(R)
    print(V)