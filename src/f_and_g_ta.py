"""
Author: Ryan Gast
Date: 12/28/2023
Calculates the Lagrange coefficients f and g in terms 
of change in true anomaly.
    Converted From Matlab to Python from: 
    "Orbital Mechanics for Engineering Students" (2013), 
    Appendix D, D.7.
Version: 1.0
Contact: ryan.d.gast@gmail.com
"""

import numpy as np

def f_and_g_ta(r0, v0, dt, mu=398600):
    """
    This function calculates the Lagrange f and g coefficients from the
    change in true anomaly since time t0.
    mu - gravitational parameter (km^3/s^2)
    dt - change in true anomaly (degrees)
    r0 - position vector at time t0 (km)
    v0 - velocity vector at time t0 (km/s)
    h - angular momentum (km^2/s)
    vr0 - radial component of v0 (km/s)
    r - radial position after the change in true anomaly
    f - the Lagrange f coefficient (dimensionless)
    g - the Lagrange g coefficient (s)
    """
    h = np.linalg.norm(np.cross(r0,v0))
    vr0 = np.dot(v0,r0)/np.linalg.norm(r0)
    r0 = np.linalg.norm(r0)
    s = np.sin(np.deg2rad(dt))
    c = np.cos(np.deg2rad(dt))
    #...Equation 2.152:
    r = h**2/mu/(1 + (h**2/mu/r0 - 1)*c - h*vr0*s/mu)
    #...Equations 2.158a & b:
    f = 1 - mu*r*(1 - c)/h**2
    g = r*r0*s/h
    return f, g

if __name__ == '__main__':
    r0 = np.array([1.0, 0.0, 0.0])
    v0 = np.array([0.0, 1.0, 0.0])
    dt = 90.0
    mu = 1.0
    f, g = f_and_g_ta(r0, v0, dt, mu)
    print(f, g)
    