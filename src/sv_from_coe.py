"""
Author: Ryan Gast
Date: 12/28/2023
Calculates of the state vector from the orbital elements.
    Converted From Matlab to Python from: 
    "Orbital Mechanics for Engineering Students" (2013), 
    Appendix D, D.22.
Version: 1.0
Contact: ryan.d.gast@gmail.com
"""

import numpy as np

def sv_from_coe(coe, mu):
    """
    Calculate the position and velocity vectors of an object in space
    given its classical orbital elements (coe) and the gravitational
    parameter (mu).

    Parameters:
    coe (list): List of classical orbital elements [h, e, RA, incl, w, TA]
    mu - gravitational parameter (km^3/s^2)
    coe - orbital elements [h e RA incl w TA]
        h = angular momentum (km^2/s)
        e = eccentricity
        RA = right ascension of the ascending node (rad)
        incl = inclination of the orbit (rad)
        w = argument of perigee (rad)
        TA = true anomaly (rad)
    mu (float): Gravitational parameter of the central body

    Variables:
    R3_w - Rotation matrix about the z-axis through the angle w
    R1_i - Rotation matrix about the x-axis through the angle i
    R3_W - Rotation matrix about the z-axis through the angle RA
    Q_pX - Matrix of the transformation from perifocal to geocentric
    equatorial frame
    rp - position vector in the perifocal frame (km)
    vp - velocity vector in the perifocal frame (km/s)
    r - position vector in the geocentric equatorial frame (km)
    v - velocity vector in the geocentric equatorial frame (km/s)
    
    Returns:
    tuple: Tuple containing the position vector (r) and velocity vector (v)
    """
    h = coe[0]
    e = coe[1]
    RA = coe[2]
    incl = coe[3]
    w = coe[4]
    TA = coe[5]

    rp = (h**2/mu) * (1/(1 + e*np.cos(TA))) * (np.cos(TA)*np.array([1,0,0]) + np.sin(TA)*np.array([0,1,0]))
    vp = (mu/h) * (-np.sin(TA)*np.array([1,0,0]) + (e + np.cos(TA))*np.array([0,1,0]))

    R3_W = np.array([[np.cos(RA), np.sin(RA), 0], [-np.sin(RA), np.cos(RA), 0], [0, 0, 1]])
    R1_i = np.array([[1, 0, 0], [0, np.cos(incl), np.sin(incl)], [0, -np.sin(incl), np.cos(incl)]])
    R3_w = np.array([[np.cos(w), np.sin(w), 0], [-np.sin(w), np.cos(w), 0], [0, 0, 1]])

    Q_pX = np.transpose(np.dot(np.dot(R3_w, R1_i), R3_W))

    r = np.dot(Q_pX, rp)
    v = np.dot(Q_pX, vp)

    r = np.transpose(r)
    v = np.transpose(v)

    return r, v

# example usage
if __name__ == "__main__":
    h = 80000
    e = 1.4
    RA = 40
    incl = 30
    w = 60
    TA = 30
    mu = 398600.4418  # Gravitational parameter for Earth

    # Convert angles from degrees to radians
    RA_rad = np.radians(RA)
    incl_rad = np.radians(incl)
    w_rad = np.radians(w)
    TA_rad = np.radians(TA)

    coe = [h, e, RA_rad, incl_rad, w_rad, TA_rad]

    # Call the function with the coe and mu as arguments
    r, v = sv_from_coe(coe, mu)
    print(r, v)