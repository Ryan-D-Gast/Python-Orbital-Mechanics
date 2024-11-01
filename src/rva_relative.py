"""
Author: Ryan Gast
Date: 1/6/2023
Finds the position, velocity, and acceleration of B relative to
A's comoving frame.
    Converted From Matlab to Python from: 
    "Orbital Mechanics for Engineering Students" (2013), 
    Appendix D, D.31.
Version: 1.0
Contact: ryan.d.gast@gmail.com
"""

import numpy as np

def rva_relative(rA, vA, rB, vB, mu=398600):
    """
    Calculates the relative position, velocity, and acceleration vectors between two objects A and B.

    Parameters:
    rA (numpy.ndarray): Position vector of object A.
    vA (numpy.ndarray): Velocity vector of object A.
    rB (numpy.ndarray): Position vector of object B.
    vB (numpy.ndarray): Velocity vector of object B.
    mu (float): Gravitational parameter (km^3/s^2), default value is Earth's gravitational parameter.

    Returns:
    r_rel_x (numpy.ndarray): Relative position vector in the transformed coordinate system.
    v_rel_x (numpy.ndarray): Relative velocity vector in the transformed coordinate system.
    a_rel_x (numpy.ndarray): Relative acceleration vector in the transformed coordinate system.
    """
    # Calculate the vector hA
    hA = np.cross(rA, vA)

    # Calculate the unit vectors i, j and k
    i = rA / np.linalg.norm(rA)
    k = hA / np.linalg.norm(hA)
    j = np.cross(k, i)

    # Calculate the transformation matrix Qxx
    QXx = np.array([i, j, k])

    # Calculate Omega and Omega_dot
    Omega = hA / np.linalg.norm(rA)**2  # Equation 7.5
    Omega_dot = -2 * np.dot(rA, vA) / np.linalg.norm(rA)**2 * Omega  # Equation 7.6

    # Calculate the accelerations aA and aB
    aA = -mu * rA / np.linalg.norm(rA)**3
    aB = -mu * rB / np.linalg.norm(rB)**3

    # Calculate r_rel
    r_rel = rB - rA

    # Calculate v_rel
    v_rel = vB - vA - np.cross(Omega, r_rel)

    # Calculate a_rel
    a_rel = aB - aA - np.cross(Omega_dot, r_rel) - np.cross(Omega, np.cross(Omega, r_rel)) - 2 * np.cross(Omega, v_rel)

    # Calculate r_rel_x, v_rel_x and a_rel_x
    r_rel_x = np.dot(QXx, r_rel)
    v_rel_x = np.dot(QXx, v_rel)
    a_rel_x = np.dot(QXx, a_rel)

    return r_rel_x, v_rel_x, a_rel_x

# example usage
if __name__ == '__main__':
    from sv_from_coe import sv_from_coe
    mu = 398600  # gravitational parameter (km^3/s^2), you may need to adjust this value
    deg = np.pi / 180  # conversion factor from degrees to radians

    # Spacecraft A:
    h_A = 52059
    e_A = 0.025724
    i_A = 60 * deg
    RAAN_A = 40 * deg
    omega_A = 30 * deg
    theta_A = 40 * deg

    # Spacecraft B:
    h_B = 52362
    e_B = 0.0072696
    i_B = 50 * deg
    RAAN_B = 40 * deg
    omega_B = 120 * deg
    theta_B = 40 * deg

    # Compute the initial state vectors of A and B using sv_from_coe function
    # The sv_from_coe function is not provided, so we assume it exists and returns r and v
    rA, vA = sv_from_coe([h_A, e_A, RAAN_A, i_A, omega_A, theta_A], mu)
    rB, vB = sv_from_coe([h_B, e_B, RAAN_B, i_B, omega_B, theta_B], mu)

    # Compute relative position, velocity and acceleration using rva_relative function
    r, v, a = rva_relative(rA, vA, rB, vB)

    # Output
    print(f"\nOrbital parameters of spacecraft A:")
    print(f" angular momentum = {h_A} (km^2/s)")
    print(f" eccentricity = {e_A}")
    print(f" inclination = {i_A / deg} (deg)")
    print(f" RAAN = {RAAN_A / deg} (deg)")
    print(f" argument of perigee = {omega_A / deg} (deg)")
    print(f" true anomaly = {theta_A / deg} (deg)")

    print(f"\nState vector of spacecraft A:")
    print(f" r = {rA}")
    print(f" (magnitude = {np.linalg.norm(rA)})")
    print(f" v = {vA}")
    print(f" (magnitude = {np.linalg.norm(vA)})")

    print(f"\nOrbital parameters of spacecraft B:")
    print(f" angular momentum = {h_B} (km^2/s)")
    print(f" eccentricity = {e_B}")
    print(f" inclination = {i_B / deg} (deg)")
    print(f" RAAN = {RAAN_B / deg} (deg)")
    print(f" argument of perigee = {omega_B / deg} (deg)")
    print(f" true anomaly = {theta_B / deg} (deg)")

    print(f"\nState vector of spacecraft B:")
    print(f" r = {rB}")
    print(f" (magnitude = {np.linalg.norm(rB)})")
    print(f" v = {vB}")
    print(f" (magnitude = {np.linalg.norm(vB)})")

    print(f"\nIn the co-moving frame attached to A:")
    print(f" Position of B relative to A = {r}")
    print(f" (magnitude = {np.linalg.norm(r)})")
    print(f" Velocity of B relative to A = {v}")
    print(f" (magnitude = {np.linalg.norm(v)})")
    print(f" Acceleration of B relative to A = {a}")
    print(f" (magnitude = {np.linalg.norm(a)})")