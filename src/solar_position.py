"""
Author: Ryan Gast
Date: 1/18/2024
Calculates the solar position (ecliptic longitude, obliquity of the ecliptic,
and geocentric position vector) for a given Julian day.
    Converted From Matlab to Python from: 
    "Orbital Mechanics for Engineering Students" (2013), 
    Appendix D, D.43.
Version: 1.0
Contact: ryan.d.gast@gmail.com
"""

import numpy as np

def solar_position(jd):
    """
    Calculate the solar position (ecliptic longitude, obliquity of the ecliptic, and geocentric position vector) for a given Julian day.

    Parameters:
    jd (float): Julian day

    Returns:
    tuple: A tuple containing the ecliptic longitude (lamda), obliquity of the ecliptic (eps), and geocentric position vector (r_S).
    """

    # Astronomical unit (km):
    AU = 149597870.691
    # Julian days since J2000:
    n = jd - 2451545
    # Julian centuries since J2000:
    cy = n/36525
    # Mean anomaly (deg):
    M = (357.528 + 0.9856003*n) % 360
    # Mean longitude (deg):
    L = (280.460 + 0.98564736*n) % 360
    # Apparent ecliptic longitude (deg):
    lamda = (L + 1.915*np.sin(np.radians(M)) + 0.020*np.sin(np.radians(2*M))) % 360
    # Obliquity of the ecliptic (deg):
    eps = 23.439 - 0.0000004*n
    # Unit vector from earth to sun:
    u = np.array([np.cos(np.radians(lamda)), np.sin(np.radians(lamda))*np.cos(np.radians(eps)), np.sin(np.radians(lamda))*np.sin(np.radians(eps))])
    # Distance from earth to sun (km):
    rS = (1.00014 - 0.01671*np.cos(np.radians(M)) - 0.000140*np.cos(np.radians(2*M)))*AU
    # Geocentric position vector (km):
    r_S = rS*u
    return lamda, eps, r_S

if __name__ == "__main__":
    # Test the function:
    jd = 2451545.0
    lamda, eps, r_S = solar_position(jd)
    print(f"Julian day: {jd}")
    print(f"Ecliptic longitude (deg): {lamda}")
    print(f"Obliquity of the ecliptic (deg): {eps}")
    print(f"Geocentric position vector (km): {r_S}")