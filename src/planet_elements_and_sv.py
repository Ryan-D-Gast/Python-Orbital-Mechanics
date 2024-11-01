"""
Author: Ryan Gast
Date: 1/9/2023
Contains functions for calculating the planet elements 
and state vectors for a given planet at a specific date and time.
    Converted From Matlab to Python from: 
    "Orbital Mechanics for Engineering Students" (2013), 
    Appendix D, D.35.
Version: 1.0
Contact: ryan.d.gast@gmail.com
"""

import math
import numpy as np
from scipy.constants import gravitational_constant as G
from scipy.constants import astronomical_unit as au
from J0 import J0
from kepler_E import kepler_E
from sv_from_coe import sv_from_coe

def planet_elements_and_sv(planet_id, year, month, day, hour, minute, second):
    """
    Calculate the planet elements and state vectors for a given planet at a specific date and time.

    Parameters:
    planet_id (int): The ID of the planet.
    year (int): The year.
    month (int): The month.
    day (int): The day.
    hour (int): The hour.
    minute (int): The minute.
    second (int): The second.

    Returns:
    tuple: A tuple containing the planet elements, position vector, velocity vector, and Julian date.
    """
    mu = 1.327 * (10**11)  # Gravitational parameter of the sun in km^3/s^2
    deg = math.pi / 180
    j0 = J0(year, month, day)  # Calculate the Julian day at 0 hours UT
    ut = (hour + minute / 60 + second / 3600) / 24  # Calculate the fraction of the day
    jd = j0 + ut  # Calculate the Julian day
    J2000_coe, rates = planetary_elements(planet_id)  # Get the J2000 orbital elements and centennial rates
    t0 = (jd - 2451545) / 36525  # Calculate the number of centuries since J2000
    elements = J2000_coe + rates * t0  # Calculate the planet elements at the given date and time
    a = elements[0]  # Semi-major axis
    e = elements[1]  # Eccentricity
    h = math.sqrt(mu * a * (1 - e**2))  # Angular momentum
    incl = elements[2]  # Inclination
    RA = zero_to_360(elements[3])  # Right ascension of the ascending node
    w_hat = zero_to_360(elements[4])  # Argument of perihelion
    L = zero_to_360(elements[5])  # Mean longitude
    w = zero_to_360(w_hat - RA)  # Argument of latitude
    M = zero_to_360((L - w_hat))  # Mean anomaly
    E = kepler_E(e, M * deg)  # Eccentric anomaly
    TA = zero_to_360(2 * math.atan(math.sqrt((1 + e) / (1 - e)) * math.tan(E / 2)) / deg)  # True anomaly
    coe = [h, e, RA, incl, w, TA, a, w_hat, L, M, E / deg]  # Orbital elements
    r, v = sv_from_coe([h, e, RA * deg, incl * deg, w * deg, TA * deg], mu)  # State vectors
    return coe, r, v, jd

def planetary_elements(planet_id):
    """
    This function extracts a planet’s J2000 orbital elements and
    centennial rates from Table 8.1.

    Parameters:
    planet_id (int): The ID of the planet (1-9).

    Returns:
    tuple: A tuple containing the J2000 orbital elements and centennial rates.
           The J2000 orbital elements are represented as a 1D numpy array with 6 elements:
           [a, e, i, L, omega, Omega], where:
           - a: Semi-major axis (in km)
           - e: Eccentricity
           - i: Inclination (in degrees)
           - L: Mean longitude (in degrees)
           - omega: Argument of perihelion (in degrees)
           - Omega: Longitude of ascending node (in degrees)

           The centennial rates are represented as a 1D numpy array with 6 elements:
           [da/dt, de/dt, di/dt, dL/dt, domega/dt, dOmega/dt], where:
           - da/dt: Rate of change of semi-major axis (in km/century)
           - de/dt: Rate of change of eccentricity (in 1/century)
           - di/dt: Rate of change of inclination (in degrees/century)
           - dL/dt: Rate of change of mean longitude (in degrees/century)
           - domega/dt: Rate of change of argument of perihelion (in degrees/century)
           - dOmega/dt: Rate of change of longitude of ascending node (in degrees/century)
    """
    J2000_elements = np.array([
        [0.38709893, 0.20563069, 7.00487, 48.33167, 77.45645, 252.25084],
        [0.72333199, 0.00677323, 3.39471, 76.68069, 131.53298, 181.97973],
        [1.00000011, 0.01671022, 0.00005, -11.26064, 102.94719, 100.46435],
        [1.52366231, 0.09341233, 1.85061, 49.57854, 336.04084, 355.45332],
        [5.20336301, 0.04839266, 1.30530, 100.55615, 14.75385, 34.40438],
        [9.53707032, 0.05415060, 2.48446, 113.71504, 92.43194, 49.94432],
        [19.19126393, 0.04716771, 0.76986, 74.22988, 170.96424, 313.23218],
        [30.06896348, 0.00858587, 1.76917, 131.72169, 44.97135, 304.88003],
        [39.48168677, 0.24880766, 17.14175, 110.30347, 224.06676, 238.92881]
    ])

    cent_rates = np.array([
        [0.00000066, 0.00002527, -23.51, -446.30, 573.57, 538101628.29],
        [0.00000092, -0.00004938, -2.86, -996.89, -108.80, 210664136.06],
        [-0.00000005, -0.00003804, -46.94, -18228.25, 1198.28, 129597740.63],
        [-0.00007221, 0.00011902, -25.47, -1020.19, 1560.78, 68905103.78],
        [0.00060737, -0.00012880, -4.15, 1217.17, 839.93, 10925078.35],
        [-0.00301530, -0.00036762, 6.11, -1591.05, -1948.89, 4401052.95],
        [0.00152025, -0.00019150, -2.09, -1681.4, 1312.56, 1542547.79],
        [-0.00125196, 0.00002514, -3.64, -151.25, -844.43, 786449.21],
        [-0.00076912, 0.00006465, 11.07, -37.33, -132.25, 522747.90]
    ])

    J2000_coe = J2000_elements[planet_id - 1, :]
    rates = cent_rates[planet_id - 1, :]

    # Convert from AU to km:
    au = 149597871
    J2000_coe[0] = J2000_coe[0] * au
    rates[0] = rates[0] * au

    # Convert from arcseconds to fractions of a degree:
    rates[2:6] = rates[2:6] / 3600

    return J2000_coe, rates

def zero_to_360(x):
    """
    This function reduces an angle to lie in the range 0 - 360 degrees.

    Parameters:
    - x: The angle to be reduced (in degrees).

    Returns:
    - The reduced angle within the range 0 - 360 degrees.
    """
    x = x % 360
    if x < 0:
        x += 360
    return x

# example usage
if __name__ == '__main__':
    from month_planet_names import month_planet_names

    # Input data
    planet_id = 3
    year = 2003
    month = 8
    day = 27
    hour = 12
    minute = 0
    second = 0
    
    print(planetary_elements(planet_id))

    # Algorithm 8.1
    coe, r, v, jd = planet_elements_and_sv(planet_id, year, month, day, hour, minute, second)
    
    print(coe)
    print(r)
    print(v)
    print(jd)

    # Convert the planet_id and month numbers into names for output
    month_name, planet_name = month_planet_names(month, planet_id)

    # Echo the input data and output the solution to the command window
    print('Example 8.7')
    print('\nInput data:')
    print(f'\nPlanet: {planet_name}')
    print(f'Year : {year}')
    print(f'Month : {month_name}')
    print(f'Day : {day}')
    print(f'Hour : {hour}')
    print(f'Minute: {minute}')
    print(f'Second: {second}')
    print(f'\n\nJulian day: {jd:.3f}')
    print('\n\nOrbital elements:')
    print(f'\nAngular momentum (km^2/s) = {coe[0]}')
    print(f'Eccentricity = {coe[1]}')
    print(f'Right ascension of the ascending node (deg) = {coe[2]}')
    print(f'Inclination to the ecliptic (deg) = {coe[3]}')
    print(f'Argument of perihelion (deg) = {coe[4]}')
    print(f'True anomaly (deg) = {coe[5]}')
    print(f'Semimajor axis (km) = {coe[6]}')
    print(f'\nLongitude of perihelion (deg) = {coe[7]}')
    print(f'Mean longitude (deg) = {coe[8]}')
    print(f'Mean anomaly (deg) = {coe[9]}')
    print(f'Eccentric anomaly (deg) = {coe[10]}')
    print('\n\nState vector:')
    print(f'\nPosition vector (km) = [{r[0]} {r[1]} {r[2]}]')
    print(f'Magnitude = {np.linalg.norm(r)}')
    print(f'\nVelocity (km/s) = [{v[0]} {v[1]} {v[2]}]')
    print(f'Magnitude = {np.linalg.norm(v)}')