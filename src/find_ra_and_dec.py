"""
Author: Ryan Gast
Date: 12/29/2023
Computes the right ascension (RA) and declination (Dec) of an object in orbit.
    Converted From Matlab to Python from: 
    "Orbital Mechanics for Engineering Students" (2013), 
    Appendix D, D.23.
Version: 1.0
Contact: ryan.d.gast@gmail.com
"""

import numpy as np
from kepler_E import kepler_E
from ra_and_dec_from_r import ra_and_dec_from_r

def find_ra_and_dec(to, tf, h, e, T, Wo, wpo, incl, Wdot, wpdot, we, mu=398600):
    """
    Calculate the right ascension (RA) and declination (Dec) of an object in orbit.

    Parameters:
    - to (float): Initial time of the orbit (in seconds)
    - tf (float): Final time of the orbit (in seconds)
    - h (float): Specific angular momentum of the orbit (in m^2/s)
    - e (float): Eccentricity of the orbit
    - T (float): Period of the orbit (in seconds)
    - Wo (float): Initial argument of perigee (in radians)
    - wpo (float): Initial argument of perigee rate (in radians/second)
    - incl (float): Inclination of the orbit (in radians)
    - Wdot (float): Rate of change of the right ascension of the ascending node (in radians/second)
    - wpdot (float): Rate of change of the argument of perigee (in radians/second)
    - mu (float): Gravitational parameter of the central body (in m^3/s^2)
    - we (float): Angular velocity of the Earth (in radians/second)

    Returns:
    - ra (list): List of right ascension values at each time step
    - dec (list): List of declination values at each time step
    """
    times = np.linspace(to, tf, 1000)
    ra = []
    dec = []
    theta = 0
    for t in times:
        M = 2*np.pi/T*t
        E = kepler_E(e, M)
        TA = 2*np.arctan(np.tan(E/2)*np.sqrt((1+e)/(1-e)))
        r = h**2/mu/(1 + e*np.cos(TA))*np.array([np.cos(TA), np.sin(TA), 0])
        W = Wo + Wdot*t
        wp = wpo + wpdot*t
        R1 = np.array([[np.cos(W), np.sin(W), 0], [-np.sin(W), np.cos(W), 0], [0, 0, 1]])
        R2 = np.array([[1, 0, 0], [0, np.cos(incl), np.sin(incl)], [0, -np.sin(incl), np.cos(incl)]])
        R3 = np.array([[np.cos(wp), np.sin(wp), 0], [-np.sin(wp), np.cos(wp), 0], [0, 0, 1]])
        QxX = np.transpose(np.dot(np.dot(R3, R2), R1))
        R = np.dot(QxX, r)
        theta = we*(t - to)
        Q = np.array([[np.cos(theta), np.sin(theta), 0], [-np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
        r_rel = np.dot(Q, R)
        alpha, delta = ra_and_dec_from_r(r_rel)
        ra.append(alpha)
        dec.append(delta)
    return ra, dec

# test
if __name__ == '__main__':
    to = 0
    tf = 86164
    h = 1.5e11
    e = 0.5
    T = 86164
    Wo = 0
    wpo = 0
    incl = 0
    Wdot = 0
    wpdot = 0
    mu = 3.986e14
    we = 7.292e-5
    ra, dec = find_ra_and_dec(to, tf, h, e, T, Wo, wpo, incl, Wdot, wpdot, mu, we)
    print(len(ra))
    print(ra[-2], dec[-2])