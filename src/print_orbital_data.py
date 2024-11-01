"""
Author: Ryan Gast
Date: 12/29/2023
Computes the ground track of a satellite orbit.
    Converted From Matlab to Python from: 
    "Orbital Mechanics for Engineering Students" (2013), 
    Appendix D, D.23.
Version: 1.0
Contact: ryan.d.gast@gmail.com
"""

import numpy as np
from sv_from_coe import sv_from_coe

def print_orbital_data(h, e, Wo, incl, wpo, TAo, a, rP, rA, T, to, Wdot, wpdot, mu):
    """
    Prints the orbital data based on the given parameters.

    Parameters:
    - h: Angular momentum (km^2/s)
    - e: Eccentricity
    - Wo: Initial right ascension (rad)
    - incl: Inclination (rad)
    - wpo: Argument of perigee (rad)
    - TAo: Initial true anomaly (rad)
    - a: Semimajor axis (km)
    - rP: Perigee radius (km)
    - rA: Apogee radius (km)
    - T: Period (s)
    - to: Time since perigee (s)
    - Wdot: Right ascension rate (rad/s)
    - wpdot: Argument of perigee rate (rad/s)
    - mu: Gravitational parameter (km^3/s^2)
    """
    coe = [h, e, Wo, incl, wpo, TAo]
    ro, vo = sv_from_coe(coe, mu)
    deg = 180 / np.pi
    print("\n" + "-"*70)
    print("\nAngular momentum = {} km^2/s".format(h))
    print("\nEccentricity = {}".format(e))
    print("\nSemimajor axis = {} km".format(a))
    print("\nPerigee radius = {} km".format(rP))
    print("\nApogee radius = {} km".format(rA))
    print("\nPeriod = {} hours".format(T/3600))
    print("\nInclination = {} deg".format(incl/deg))
    print("\nInitial true anomaly = {} deg".format(TAo/deg))
    print("\nTime since perigee = {} hours".format(to/3600))
    print("\nInitial RA = {} deg".format(Wo/deg))
    print("\nRA_dot = {} deg/period".format(Wdot/deg*T))
    print("\nInitial wp = {} deg".format(wpo/deg))
    print("\nwp_dot = {} deg/period".format(wpdot/deg*T))
    print("\nr0 = [{}, {}, {}] (km)".format(ro[0], ro[1], ro[2]))
    print("\nmagnitude = {} km".format(np.linalg.norm(ro)))
    print("\nv0 = [{}, {}, {}] (km)".format(vo[0], vo[1], vo[2]))
    print("\nmagnitude = {} km".format(np.linalg.norm(vo)))
    print("\n" + "-"*70)