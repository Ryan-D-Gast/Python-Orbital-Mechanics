"""
Author: Ryan Gast
Date: 12/31/2023
This function calculates the geocentric equatorial position and
velocity vectors of an object from radar observations of range,
azimuth, elevation angle and their rates.
    Converted From Matlab to Python from: 
    "Orbital Mechanics for Engineering Students" (2013), 
    Appendix D, D.28.
Version: 1.0
Contact: ryan.d.gast@gmail.com
"""

import numpy as np

# Global constants
f = 0.0033528106647474805  # Earth's flattening factor
Re = 6378.137  # Earth's equatorial radius in km
wE = 7.2921159e-5  # Earth's angular velocity in rad/s

def rv_from_observe(rho, rhodot, A, Adot, a, adot, theta, phi, H):
    """
    Calculate the position vector (r) and velocity vector (v) of an observed object in a rotating Earth frame.

    Parameters:
    rho (float): Range from observer to object
    rhodot (float): Range rate from observer to object
    A (float): Azimuth angle of the object
    Adot (float): Rate of change of azimuth angle of the object
    a (float): Elevation angle of the object
    adot (float): Rate of change of elevation angle of the object
    theta (float): Observer's longitude
    phi (float): Observer's latitude
    H (float): Observer's altitude above the Earth's surface

    Returns:
    tuple: A tuple containing the position vector (r) and velocity vector (v) of the observed object.
    """

    deg = np.pi/180
    omega = np.array([0, 0, wE])

    # Convert angular quantities from degrees to radians
    A = A * deg
    Adot = Adot * deg
    a = a * deg
    adot = adot * deg
    theta = theta * deg
    phi = phi * deg

    # Equation 5.56
    R = np.array([
        (Re/np.sqrt(1-(2*f - f*f)*np.sin(phi)**2) + H)*np.cos(phi)*np.cos(theta),
        (Re/np.sqrt(1-(2*f - f*f)*np.sin(phi)**2) + H)*np.cos(phi)*np.sin(theta),
        (Re*(1 - f)**2/np.sqrt(1-(2*f - f*f)*np.sin(phi)**2) + H)*np.sin(phi)
    ])

    # Equation 5.66
    Rdot = np.cross(omega, R)

    # Equation 5.83a
    dec = np.arcsin(np.cos(phi)*np.cos(A)*np.cos(a) + np.sin(phi)*np.sin(a))

    # Equation 5.83b
    h = np.arccos((np.cos(phi)*np.sin(a) - np.sin(phi)*np.cos(A)*np.cos(a))/np.cos(dec))
    if (A > 0) & (A < np.pi):
        h = 2*np.pi - h

    # Equation 5.83c
    RA = theta - h

    # Equations 5.57
    Rho = np.array([np.cos(RA)*np.cos(dec), np.sin(RA)*np.cos(dec), np.sin(dec)])

    # Equation 5.63
    r = R + rho*Rho

    # Equation 5.84
    decdot = (-Adot*np.cos(phi)*np.sin(A)*np.cos(a) + adot*(np.sin(phi)*np.cos(a) - np.cos(phi)*np.cos(A)*np.sin(a)))/np.cos(dec)

    # Equation 5.85
    RAdot = wE + (Adot*np.cos(A)*np.cos(a) - adot*np.sin(A)*np.sin(a) + decdot*np.sin(A)*np.cos(a)*np.tan(dec)) / (np.cos(phi)*np.sin(a) - np.sin(phi)*np.cos(A)*np.cos(a))

    # Equations 5.69 and 5.72
    Rhodot = np.array([-RAdot*np.sin(RA)*np.cos(dec) - decdot*np.cos(RA)*np.sin(dec), RAdot*np.cos(RA)*np.cos(dec) - decdot*np.sin(RA)*np.sin(dec), decdot*np.cos(dec)])

    # Equation 5.64
    v = Rdot + rhodot*Rho + rho*Rhodot

    return r, v

# example usage
if __name__ == '__main__':
    from coe_from_sv import coe_from_sv
    # Constants
    deg = np.pi/180
    f = 1/298.256421867
    Re = 6378.13655
    wE = 7.292115e-5
    mu = 398600.4418

    # Data declaration for Example 5.10
    rho = 2551
    rhodot = 0
    A = 90
    Adot = 0.1130
    a = 30
    adot = 0.05651
    theta = 300
    phi = 60
    H = 0

    # Algorithm 5.4
    r, v = rv_from_observe(rho, rhodot, A, Adot, a, adot, theta, phi, H)

    # Algorithm 4.2
    coe = coe_from_sv(r, v, mu)
    h, e, RA, incl, w, TA, a = coe

    # Equation 2.40
    rp = h**2/mu/(1 + e)

    # Echo the input data and output the solution to the command window
    print('Example 5.10')
    print('Input data:')
    print(f'Slant range (km) = {rho}')
    print(f'Slant range rate (km/s) = {rhodot}')
    print(f'Azimuth (deg) = {A}')
    print(f'Azimuth rate (deg/s) = {Adot}')
    print(f'Elevation (deg) = {a}')
    print(f'Elevation rate (deg/s) = {adot}')
    print(f'Local sidereal time (deg) = {theta}')
    print(f'Latitude (deg) = {phi}')
    print(f'Altitude above sea level (km) = {H}')
    print('\nSolution:')
    print(f'State vector:\nr (km) = {r}\nv (km/s) = {v}')
    print('Orbital elements:')
    print(f'Angular momentum (km^2/s) = {h}')
    print(f'Eccentricity = {e}')
    print(f'Inclination (deg) = {incl/deg}')
    print(f'RA of ascending node (deg) = {RA/deg}')
    print(f'Argument of perigee (deg) = {w/deg}')
    print(f'True anomaly (deg) = {TA/deg}')
    print(f'Semimajor axis (km) = {a}')
    print(f'Perigee radius (km) = {rp}')

    # If the orbit is an ellipse, output its period
    if e < 1:
        T = 2*np.pi/np.sqrt(mu)*a**1.5
        print(f'Period:\nSeconds = {T}\nMinutes = {T/60}\nHours = {T/3600}\nDays = {T/24/3600}')