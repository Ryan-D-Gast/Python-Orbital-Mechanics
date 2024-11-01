"""
Author: Ryan Gast
Date: 1/14/2024
This function computes the state vector of the moon at a given time
    Converted From Matlab to Python from: 
    "Orbital Mechanics for Engineering Students" (2013), 
    Appendix D, D.37.
Version: 1.0
Contact: ryan.d.gast@gmail.com
"""

import numpy as np

def simpsons_lunar_ephemeris(jd):
    """
    David G. Simpson, "An Alternative Ephemeris Model for
    On-Board Flight Software Use," Proceedings of the 1999 Flight Mechanics
    Symposium, NASA Goddard Space Flight Center, pp. 175 - 184.
    
    This function computes the state vector of the moon at a given time
    relative to the earth's geocentric equatorial frame using a curve fit
    to JPL's DE200 (1982) ephemeris model.
    
    Args:
        jd (float): Julian date (days)
    
    Returns:
        tuple: A tuple containing the position vector (pos) and velocity vector (vel) of the moon.
            pos (numpy.ndarray): Position vector (km)
            vel (numpy.ndarray): Velocity vector (km/s)
    
    Notes:
        - The function requires the numpy library to be imported.
        - The function uses a matrix of amplitudes (a), frequencies (b), and phase angles (c)
          to compute the position and velocity vectors.
        - The time is calculated in centuries since J2000.
        - The tfac variable represents the number of seconds in a Julian century (36525 days).
    """
    import numpy as np
    
    tfac = 36525*3600*24
    t = (jd - 2451545.0)/36525
    a = np.array([[383.0, 31.5, 10.6, 6.2, 3.2, 2.3, 0.8],
                  [351.0, 28.9, 13.7, 9.7, 5.7, 2.9, 2.1],
                  [153.2, 31.5, 12.5, 4.2, 2.5, 3.0, 1.8]]) * 1.e3
    b = np.array([[8399.685, 70.990, 16728.377, 1185.622, 7143.070, 15613.745, 8467.263],
                  [8399.687, 70.997, 8433.466, 16728.380, 1185.667, 7143.058, 15613.755],
                  [8399.672, 8433.464, 70.996, 16728.364, 1185.645, 104.881, 8399.116]])
    c = np.array([[5.381, 6.169, 1.453, 0.481, 5.017, 0.857, 1.010],
                  [3.811, 4.596, 4.766, 6.165, 5.164, 0.300, 5.565],
                  [3.807, 1.629, 4.595, 6.162, 5.167, 2.555, 6.248]])
    pos = np.zeros(3)
    vel = np.zeros(3)
    for i in range(3):
        for j in range(7):
            pos[i] += a[i,j]*np.sin(b[i,j]*t + c[i,j])
            vel[i] += a[i,j]*np.cos(b[i,j]*t + c[i,j])*b[i,j]
        vel[i] = vel[i]/tfac
    return pos, vel

# Test
if __name__ == "__main__":
    jd = 2451545.0
    pos, vel = simpsons_lunar_ephemeris(jd)
    print(f"Position: {pos}")
    print(f"Velocity: {vel}")