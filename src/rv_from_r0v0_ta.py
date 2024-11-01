"""
Author: Ryan Gast
Date: 12/26/2023
Calculate the state vector given the initial state vector and
the change in true anomaly.
    Converted From Matlab to Python from: 
    "Orbital Mechanics for Engineering Students" (2013), 
    Appendix D, D.8.
Version: 1.0
Contact: ryan.d.gast@gmail.com
"""

import numpy as np
from f_and_g_ta import f_and_g_ta
from fDot_and_gDot_ta import fDot_and_gDot_ta

def rv_from_r0v0_ta(r0, v0, dt, mu):
    """
    Compute the final position and velocity vectors after a given time interval.

    Parameters:
    r0 (numpy.ndarray): Initial position vector.
    v0 (numpy.ndarray): Initial velocity vector.
    dt (float): Time interval.
    mu (float): Gravitational parameter.

    Returns:
    tuple: Final position vector (r) and final velocity vector (v).
    """
    #...Compute the f and g functions and their derivatives:
    f, g = f_and_g_ta(r0, v0, dt, mu)
    fdot, gdot = fDot_and_gDot_ta(r0, v0, dt, mu)
    #...Compute the final position and velocity vectors:
    r = f*r0 + g*v0
    v = fdot*r0 + gdot*v0
    return r, v

if __name__ == '__main__':
    #...Test the function: 
    r0 = np.array([8000, 0, 6000]) #km
    v0 = np.array([0, 7, 0]) #km/s
    dt = 45 #degrees
    mu = 398600 #km^3/s^2
    r, v = rv_from_r0v0_ta(r0, v0, dt, mu)
    print(f"\n r = {r} km")
    print(f"\n v = {v} km/s")