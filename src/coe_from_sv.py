"""
Author: Ryan Gast
Date: 12/29/2023
Calculates of the orbital elements from the state vector.
    Converted From Matlab to Python from: 
    "Orbital Mechanics for Engineering Students" (2013), 
    Appendix D, D.18.
Version: 1.0
Contact: ryan.d.gast@gmail.com
"""

import numpy as np

def coe_from_sv(R, V, mu):
    """
    Calculates the classical orbital elements (COE) from the position vector (R),
    velocity vector (V), and gravitational parameter (mu).

    Parameters:
    - R (numpy.ndarray): Position vector in Cartesian coordinates.
    - V (numpy.ndarray): Velocity vector in Cartesian coordinates.
    - mu (float): Gravitational parameter of the central body.

    Returns:
    - coe (list): List containing the classical orbital elements [h, e, RA, incl, w, TA, a].

    """
    # Set tolerance
    eps = 1.e-10
    
    # Calculate the magnitude of position vector and velocity vector
    r = np.linalg.norm(R)
    v = np.linalg.norm(V)
    
    # Calculate the dot product of position vector and velocity vector
    vr = np.dot(R, V) / r
    
    # Calculate the angular momentum vector
    H = np.cross(R, V)
    h = np.linalg.norm(H)
    
    # Calculate the inclination angle
    incl = np.arccos(H[2] / h)
    
    # Calculate the node vector
    N = np.cross([0, 0, 1], H)
    n = np.linalg.norm(N)
    
    # Calculate the right ascension of the ascending node
    if n != 0:
        RA = np.arccos(N[0] / n)
        if N[1] < 0:
            RA = 2 * np.pi - RA
    else:
        RA = 0
    
    # Calculate the eccentricity vector
    E = 1 / mu * ((v**2 - mu / r) * R - r * vr * V)
    e = np.linalg.norm(E)
    
    # Calculate the argument of periapsis
    if n != 0:
        if e > eps:
            w = np.arccos(np.dot(N, E) / n / e)
            if E[2] < 0:
                w = 2 * np.pi - w
        else:
            w = 0
    else:
        w = 0
    
    # Calculate the true anomaly
    if e > eps:
        TA = np.arccos(np.dot(E, R) / e / r)
        if vr < 0:
            TA = 2 * np.pi - TA
    else:
        cp = np.cross(N, R)
        if cp[2] >= 0:
            TA = np.arccos(np.dot(N, R) / n / r)
        else:
            TA = 2 * np.pi - np.arccos(np.dot(N, R) / n / r)
    
    # Calculate the semi-major axis
    a = h**2 / mu / (1 - e**2)
    
    # Return the classical orbital elements
    coe = [h, e, RA, incl, w, TA, a]
    return coe

# example usage
if __name__ == '__main__':
    mu = 398600
    r = np.array([-6045, -3490, 2500])
    v = np.array([-3.457, 6.618, 2.533])
    result = coe_from_sv(r, v, mu)
    # Values in pure from
    print(coe_from_sv(r, v, mu))


    # Results similar to fprintf statements
    print(f"\nAngular momentum (km^2/s) = {result[0]}")
    print(f"Eccentricity = {result[1]}")
    print(f"Right ascension (deg) = {result[2] / np.deg2rad(1)}")
    print(f"Inclination (deg) = {result[3] / np.deg2rad(1)}")
    print(f"Argument of perigee (deg) = {result[4] / np.deg2rad(1)}")
    print(f"True anomaly (deg) = {result[5] / np.deg2rad(1)}")
    print(f"Semimajor axis (km): = {result[6]}")

    # If the orbit is an ellipse, output its period
    if result[1] < 1:
        T = 2 * np.pi / np.sqrt(mu) * result[6] ** 1.5
        print("\nPeriod:")
        print(f"Seconds = {T}")
        print(f"Minutes = {T / 60}")
        print(f"Hours = {T / 3600}")
        print(f"Days = {T / 24 / 3600}")