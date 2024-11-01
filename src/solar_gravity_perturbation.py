"""
Author: Ryan Gast
Date: 1/18/2024
Calculates the solar gravity perturbation effects on the orbital elements of a satellite.
    Converted From Matlab to Python from: 
    "Orbital Mechanics for Engineering Students" (2013), 
    Appendix D, D.48.
Version: 1.0
Contact: ryan.d.gast@gmail.com
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import savgol_filter
from sv_from_coe import sv_from_coe
from solar_position import solar_position

def solar_gravity_perturbation(n, a0, e0, w0, RA0, i0, TA0, JD0, mu_earth=398600, mu_sun=132.712e9):
    """
    Calculates the solar gravity perturbation effects on the orbital elements of a satellite.

    Parameters:
        n (int): Index of the figure to be plotted.
        a0 (float): Initial semi-major axis of the satellite's orbit (km).
        e0 (float): Initial eccentricity of the satellite's orbit.
        w0 (float): Initial argument of perigee of the satellite's orbit (rad).
        RA0 (float): Initial right ascension of the ascending node of the satellite's orbit (rad).
        i0 (float): Initial inclination of the satellite's orbit (rad).
        TA0 (float): Initial true anomaly of the satellite's orbit (rad).
        JD0 (float): Initial Julian Day.
        mu_earth (float, optional): Earth's gravitational parameter (km^3/s^2). Default is 398600.
        mu_sun (float, optional): Sun's gravitational parameter (km^3/s^2). Default is 132.712e9.

    Returns:
        tuple: A tuple containing the time vector, right ascension, inclination, and argument of perigee.
    """

    # Conversion factors:
    hours = 3600  # Hours to seconds
    days = 24 * hours  # Days to seconds
    deg = np.pi / 180  # Degrees to radians

    # Constants:
    mu_earth = 398600  # Earth’s gravitational parameter (km^3/s^2)
    mu_sun = 132.712e9  # Sun’s gravitational parameter (km^3/s^2)
    RE = 6378  # Earth’s radius (km)

    # Initial orbital parameters (calculated from the given data):
    h0 = np.sqrt(mu_earth * a0 * (1 - e0**2))  # angular momentum (km^2/s)
    T0 = 2 * np.pi / np.sqrt(mu_earth) * a0**1.5  # Period (s)
    rp0 = h0**2 / mu_earth / (1 + e0)  # perigee radius (km)
    ra0 = h0**2 / mu_earth / (1 - e0)  # apogee radius (km)

    # Store initial orbital elements (from above) in the vector coe0:
    coe0 = [h0, e0, RA0, i0, w0, TA0]

    # Use solve_ivp to integrate the Equations 12.84, the Gauss variational
    # equations with lunar gravity as the perturbation, from t0 to tf:
    t0 = 0
    tf = 720 * days
    y0 = coe0  # Initial orbital elements
    nout = 400  # Number of solution points to output
    tspan = np.linspace(t0, tf, nout)  # Integration time interval
    
    def rates(t, f):
        """
        Calculates the rates of change of the orbital elements at a given time.

        Parameters:
        - t: Time at which the rates are calculated.
        - f: Array of orbital elements [h, e, RA, i, w, TA].

        Returns:
        - dfdt: Array of rates of change of the orbital elements [hdot, edot, RAdot, idot, wdot, TAdot].
        """
        # The orbital elements at time t:
        h = f[0]
        e = f[1]
        RA = f[2]
        i = f[3]
        w = f[4]
        TA = f[5]
        phi = w + TA  # argument of latitude

        # Obtain the state vector at time t from Algorithm 4.5:
        coe = [h, e, RA, i, w, TA]
        R, V = sv_from_coe(coe, mu_earth)  # You need to define this function

        # Obtain the unit vectors of the rsw system:
        r = np.linalg.norm(R)
        ur = R / r  # radial
        H = np.cross(R, V)
        uh = H / np.linalg.norm(H)  # normal
        s = np.cross(uh, ur)
        us = s / np.linalg.norm(s)  # transverse

        # Update the Julian Day:
        JD = JD0 + t / days

        # Find and normalize the position vector of the sun:
        lamda, eps, R_S = solar_position(JD)  # You need to define this function
        r_S = np.linalg.norm(R_S)
        R_rel = R_S - R  # R_rel = position vector of sun wrt satellite
        r_rel = np.linalg.norm(R_rel)

        # See Appendix F:
        q = np.dot(R, (2 * R_S - R)) / r_S**2
        F = (q**2 - 3 * q + 3) * q / (1 + (1 - q)**1.5)

        # Gravitational perturbation of the sun (Equation 12.130):
        ap = mu_sun / r_rel**3 * (F * R_S - R)

        # Perturbation components in the rsw system:
        apr = np.dot(ap, ur)
        aps = np.dot(ap, us)
        aph = np.dot(ap, uh)

        # Gauss variational equations (Equations 12.84):
        hdot = r * aps
        edot = h / mu_earth * np.sin(TA) * apr + 1 / mu_earth / h * ((h**2 + mu_earth * r) * np.cos(TA) + mu_earth * e * r) * aps
        RAdot = r / h / np.sin(i) * np.sin(phi) * aph
        idot = r / h * np.cos(phi) * aph
        wdot = - h * np.cos(TA) / mu_earth / e * apr + (h**2 + mu_earth * r) / mu_earth / e / h * np.sin(TA) * aps - r * np.sin(phi) / h / np.tan(i) * aph
        TAdot = h / r**2 + 1 / e / h * (h**2 / mu_earth * np.cos(TA) * apr - (r + h**2 / mu_earth) * np.sin(TA) * aps)

        # Return rates to solve_ivp in the array dfdt:
        dfdt = [hdot, edot, RAdot, idot, wdot, TAdot]

        return dfdt

    sol = solve_ivp(rates, [t0, tf], y0, t_eval=tspan, rtol=1e-8, atol=1e-8)  # You need to define this function

    # Time histories of the right ascension, inclination and argument of perigee:
    RA = sol.y[2, :]
    i = sol.y[3, :]
    w = sol.y[4, :]

    # Smooth the data to eliminate short period variations:
    RA = savgol_filter(RA, 51, 3)
    i = savgol_filter(i, 51, 3)
    w = savgol_filter(w, 51, 3)

    plt.figure(n)
    plt.subplot(1, 3, 1)
    plt.plot(sol.t / days, (RA - RA0) / deg)
    plt.title('Right Ascension vs Time')
    plt.xlabel('t (days)')
    plt.ylabel('Omega (deg)')

    plt.subplot(1, 3, 2)
    plt.plot(sol.t / days, (i - i0) / deg)
    plt.title('Inclination vs Time')
    plt.xlabel('t (days)')
    plt.ylabel('i (deg)')

    plt.subplot(1, 3, 3)
    plt.plot(sol.t / days, (w - w0) / deg)
    plt.title('Argument of Perigee vs Time')
    plt.xlabel('t (days)')
    plt.ylabel('omega (deg)')

    plt.tight_layout()
    plt.show()
    
    return sol.t, RA, i, w
    
# example usage
if __name__ == '__main__':
    # Conversion factors:
    hours = 3600  # Hours to seconds
    days = 24 * hours  # Days to seconds
    deg = np.pi / 180  # Degrees to radians

    # Constants:
    mu_earth = 398600  # Earth’s gravitational parameter (km^3/s^2)
    mu_sun = 132.712e9  # Sun’s gravitational parameter (km^3/s^2)
    RE = 6378  # Earth’s radius (km)

    # Initial data for each of the three given orbits:
    types = ['GEO', 'HEO', 'LEO']

    # GEO
    n = 0
    a0 = 42164  # semimajor axis (km)
    e0 = 0.0001  # eccentricity
    w0 = 0  # argument of perigee (rad)
    RA0 = 0  # right ascension (rad)
    i0 = 1 * deg  # inclination (rad)
    TA0 = 0  # true anomaly (rad)
    JD0 = 2454283  # Julian Day
    solar_gravity_perturbation(n, a0, e0, w0, RA0, i0, TA0, JD0)

    # HEO
    n = 1
    a0 = 26553.4
    e0 = 0.741
    w0 = 270
    RA0 = 0
    i0 = 63.4 * deg
    TA0 = 0
    JD0 = 2454283
    solar_gravity_perturbation(n, a0, e0, w0, RA0, i0, TA0, JD0)

    # LEO
    n = 2
    a0 = 6678.136
    e0 = 0.01
    w0 = 0
    RA0 = 0
    i0 = 28.5 * deg
    TA0 = 0
    JD0 = 2454283
    solar_gravity_perturbation(n, a0, e0, w0, RA0, i0, TA0, JD0)