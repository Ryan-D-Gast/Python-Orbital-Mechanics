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
from find_ra_and_dec import find_ra_and_dec
from form_seperate_curves import form_sep_curves
from plot_ground_track import plot_ground_track
from print_orbital_data import print_orbital_data

# Constants
deg = np.pi/180  # Conversion factor from degrees to radians
mu = 398600  # Gravitational parameter of Earth (km^3/s^2)
J2 = 0.00108263  # Second zonal harmonic coefficient
Re = 6378  # Mean radius of Earth (km)
we = (2*np.pi + 2*np.pi/365.26)/(24*3600)  # Earth's rotation rate (rad/s)

# Data declaration for Example 4.12:
rP = 6700  # Perigee radius (km)
rA = 10000  # Apogee radius (km)
TAo = 230*deg  # True anomaly at epoch (rad)
Wo = 270*deg  # Argument of perigee at epoch (rad)
incl = 60*deg  # Inclination at epoch (rad)
wpo = 45*deg  # Longitude of ascending node at epoch (rad)
n_periods = 3.25  # Number of orbital periods

def ground_track(rP, rA, TAo, Wo, incl, wpo, n_periods, mu, J2, Re, we):
    """
    Computes the ground track of a satellite orbit.

    Args:
        rP (float): Radius of perigee (km).
        rA (float): Radius of apogee (km).
        TAo (float): True anomaly at epoch (rad).
        Wo (float): Argument of perigee at epoch (rad).
        incl (float): Inclination of the orbit (rad).
        wpo (float): Longitude of perigee at epoch (rad).
        n_periods (int): Number of orbital periods to compute the ground track for.
        mu (float): Gravitational parameter of the central body (km^3/s^2).
        J2 (float): Second zonal harmonic coefficient of the central body.
        Re (float): Equatorial radius of the central body (km).
        we (float): Angular velocity of the central body (rad/s).

    Returns:
        None, but prints the orbital data and plots the ground track.
    """
    # Compute the initial time (since perigee) and the rates of node regression and perigee advance
    a = (rA + rP)/2  # Semi-major axis (km)
    T = 2*np.pi/np.sqrt(mu)*a**(3/2)  # Orbital period (s)
    e = (rA - rP)/(rA + rP)  # Eccentricity
    h = np.sqrt(mu*a*(1 - e**2))  # Specific angular momentum (km^2/s)
    Eo = 2*np.arctan(np.tan(TAo/2)*np.sqrt((1-e)/(1+e)))  # Eccentric anomaly at epoch (rad)
    Mo = Eo - e*np.sin(Eo)  # Mean anomaly at epoch (rad)
    to = Mo*(T/2/np.pi)  # Initial time since perigee (s)
    tf = to + n_periods*T  # Final time since perigee (s)
    fac = -3/2*np.sqrt(mu)*J2*Re**2/(1-e**2)**2/a**(7/2)  # Common factor for node regression and perigee advance
    Wdot = fac*np.cos(incl)  # Rate of node regression (rad/s)
    wpdot = fac*(5/2*np.sin(incl)**2 - 2)  # Rate of perigee advance (rad/s)

    # Get results
    ra, dec = find_ra_and_dec(to, tf, h, e, T, Wo, wpo, incl, Wdot, wpdot, mu, we)
    RA, Dec, n_curves = form_sep_curves(ra, dec)
    plot_ground_track(RA, Dec, ra, dec, n_curves)
    print_orbital_data(h, e, Wo, incl, wpo, TAo, a, rP, rA, T, to, Wdot, wpdot, mu)
    
ground_track(rP, rA, TAo, Wo, incl, wpo, n_periods, mu, J2, Re, we)