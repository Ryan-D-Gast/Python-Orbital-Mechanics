"""
Author: Ryan Gast
Date: 1/18/2024
Determines if satellite is in earth's shadow.
    Converted From Matlab to Python from: 
    "Orbital Mechanics for Engineering Students" (2013), 
    Appendix D, D.44.
Version: 1.0
Contact: ryan.d.gast@gmail.com
"""

import numpy as np

def los(r_sat, r_sun):
    """
    Determines whether a line from the sun to the satellite intersects the earth.

    Parameters:
    r_sat (numpy.ndarray): Position vector of the satellite.
    r_sun (numpy.ndarray): Position vector of the sun.

    Returns:
    int: 0 if the line intersects the earth, 1 otherwise.
    """
    RE = 6378  # Earth’s radius (km)
    rsat = np.linalg.norm(r_sat)
    rsun = np.linalg.norm(r_sun)
    # Angle between sun and satellite position vectors:
    theta = np.degrees(np.arccos(np.dot(r_sat, r_sun) / (rsat * rsun)))
    # Angle between the satellite position vector and the radial to the point
    # of tangency with the earth of a line from the satellite:
    theta_sat = np.degrees(np.arccos(RE / rsat))
    # Angle between the sun position vector and the radial to the point
    # of tangency with the earth of a line from the sun:
    theta_sun = np.degrees(np.arccos(RE / rsun))
    # Determine whether a line from the sun to the satellite
    # intersects the earth:
    if theta_sat + theta_sun <= theta:
        light_switch = 0  # yes
    else:
        light_switch = 1  # no
    return light_switch

# Test
if __name__ == '__main__':
    rsat = np.array([0, 0, 10000])
    rsun = np.array([0, 0, 100000])
    print(los(rsat, rsun))