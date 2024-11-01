"""
Author: Ryan Gast
Date: 12/28/2023
Calculates the Lagrange coefficient derivatives 
f and g in terms of change in true anomaly.
    Converted From Matlab to Python from: 
    "Orbital Mechanics for Engineering Students" (2013), 
    Appendix D, D.7.
Version: 1.0
Contact: ryan.d.gast@gmail.com
"""

import numpy as np

def fDot_and_gDot_ta(r0, v0, dt, mu=398600):
    """
    This function calculates the time derivatives of the Lagrange
    f and g coefficients from the change in true anomaly since time t0.
    mu - gravitational parameter (km^3/s^2)
    dt - change in true anomaly (degrees)
    r0 - position vector at time t0 (km)
    v0 - velocity vector at time t0 (km/s)
    h - angular momentum (km^2/s)
    vr0 - radial component of v0 (km/s)
    fdot - time derivative of the Lagrange f coefficient (1/s)
    gdot - time derivative of the Lagrange g coefficient (dimensionless)
    """
    h = np.linalg.norm(np.cross(r0,v0))
    vr0 = np.dot(v0,r0)/np.linalg.norm(r0)
    r0 = np.linalg.norm(r0)
    c = np.cos(np.deg2rad(dt))
    s = np.sin(np.deg2rad(dt))
    #...Equations 2.158c & d:
    fdot = mu/h*(vr0/h*(1 - c) - s/r0)
    gdot = 1 - mu*r0/h**2*(1 - c)
    return fdot, gdot
    
if __name__ == '__main__':
    r0 = np.array([1.0, 0.0, 0.0])
    v0 = np.array([0.0, 1.0, 0.0])
    dt = 90.0
    mu = 1.0
    fdot, gdot = fDot_and_gDot_ta(r0, v0, dt, mu)
    print(fdot, gdot)