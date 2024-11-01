"""
Author: Ryan Gast
Date: 1/18/2024
Calculates the lunar gravity perturbation effects on the orbital elements of a satellite.
    Converted From Matlab to Python from: 
    "Orbital Mechanics for Engineering Students" (2013), 
    Appendix D, D.47.
Version: 1.0
Contact: ryan.d.gast@gmail.com
"""

import numpy as np
import matplotlib.pyplot as plt
from sv_from_coe import sv_from_coe
from lunar_position import lunar_position
from scipy.integrate import solve_ivp

def lunar_gravity_perturbation(n, a0, e0, w0, RA0, i0, TA0, JD0, mu_earth=398600, mu_moon=4903):
    """
    Calculates the lunar gravity perturbation effects on the orbital elements of a satellite.

    Parameters:
    - n: int, the number of solution points to output.
    - a0: float, the initial semi-major axis of the satellite's orbit (in km).
    - e0: float, the initial eccentricity of the satellite's orbit.
    - w0: float, the initial argument of perigee of the satellite's orbit (in radians).
    - RA0: float, the initial right ascension of the ascending node of the satellite's orbit (in radians).
    - i0: float, the initial inclination of the satellite's orbit (in radians).
    - TA0: float, the initial true anomaly of the satellite's orbit (in radians).
    - JD0: float, the initial Julian Day.
    - mu_earth: float, optional, the gravitational parameter of the Earth (default value is 398600 km^3/s^2).
    - mu_moon: float, optional, the gravitational parameter of the Earth-Moon system (default value is 4903 km^3/s^2).

    Returns:
        tuple: A tuple containing the time vector, right ascension, inclination, and argument of perigee.
    """
    
    # Conversion factors:
    hours = 3600  # Hours to seconds
    days = 24 * hours  # Days to seconds
    deg = np.pi / 180  # Degrees to radians

    # Initial orbital parameters (calculated from the given data):
    h0 = np.sqrt(mu_earth * a0 * (1 - e0 ** 2))  # angular momentum (km^2/s)
    T0 = 2 * np.pi / np.sqrt(mu_earth) * a0 ** 1.5  # Period (s)
    rp0 = h0 ** 2 / mu_earth / (1 + e0)  # perigee radius (km)
    ra0 = h0 ** 2 / mu_earth / (1 - e0)  # apogee radius (km)

    # Store initial orbital elements (from above) in the vector coe0:
    coe0 = [h0, e0, RA0, i0, w0, TA0]

    # Use odeint to integrate the Equations 12.84, the Gauss variational
    # equations with lunar gravity as the perturbation, from t0 to tf:
    t0 = 0
    tf = 60 * days
    y0 = coe0  # Initial orbital elements
    nout = 400  # Number of solution points to output
    tspan = np.linspace(t0, tf, nout)  # Integration time interval

    # Define the rates function
    def rates(t, f):
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

        # Find and normalize the position vector of the moon:
        R_m = lunar_position(JD)  # You need to define this function
        r_m = np.linalg.norm(R_m)
        R_rel = R_m - R  # R_rel = position vector of moon wrt satellite
        r_rel = np.linalg.norm(R_rel)

        # See Appendix F:
        q = np.dot(R, (2 * R_m - R)) / r_m ** 2
        F = (q ** 2 - 3 * q + 3) * q / (1 + (1 - q) ** 1.5)

        # Gravitational perturbation of the moon (Equation 12.117):
        ap = mu_moon / r_rel ** 3 * (F * R_m - R)

        # Perturbation components in the rsw system:
        apr = np.dot(ap, ur)
        aps = np.dot(ap, us)
        aph = np.dot(ap, uh)

        # Gauss variational equations (Equations 12.84):
        hdot = r * aps
        edot = h / mu_earth * np.sin(TA) * apr + 1 / mu_earth / h * ((h ** 2 + mu_earth * r) * np.cos(TA) + mu_earth * e * r) * aps
        RAdot = r / h / np.sin(i) * np.sin(phi) * aph
        idot = r / h * np.cos(phi) * aph
        wdot = - h * np.cos(TA) / mu_earth / e * apr + (h ** 2 + mu_earth * r) / mu_earth / e / h * np.sin(TA) * aps - r * np.sin(phi) / h / np.tan(i) * aph
        TAdot = h / r ** 2 + 1 / e / h * (h ** 2 / mu_earth * np.cos(TA) * apr - (r + h ** 2 / mu_earth) * np.sin(TA) * aps)

        # Return rates to odeint in the array dfdt:
        dfdt = [hdot, edot, RAdot, idot, wdot, TAdot]
        return dfdt

    options = {'rtol': 1e-8, 'atol': 1e-8}
    sol = solve_ivp(rates, [tspan[0], tspan[-1]], y0, t_eval=tspan, method='RK45', **options)
    t = sol.t
    y = sol.y.T

    # Time histories of the right ascension, inclination and argument of perigee:
    RA = y[:, 2]
    i = y[:, 3]
    w = y[:, 4]

    # Plotting
    plt.figure(n)
    plt.subplot(1, 3, 1)
    plt.plot(tspan / days, (RA - RA0) / deg)
    plt.title('Right Ascension vs Time')
    plt.xlabel('t (days)')
    plt.ylabel('Omega (deg)')

    plt.subplot(1, 3, 2)
    plt.plot(tspan / days, (i - i0) / deg)
    plt.title('Inclination vs Time')
    plt.xlabel('t (days)')
    plt.ylabel('i (deg)')

    plt.subplot(1, 3, 3)
    plt.plot(tspan / days, (w - w0) / deg)
    plt.title('Argument of Perigee vs Time')
    plt.xlabel('t (days)')
    plt.ylabel('omega (deg)')

    plt.tight_layout()
    plt.show()

    return t, RA, i, w
    
    
# example usage
if __name__ == '__main__':
    # Conversion factors
    hours = 3600  # Hours to seconds
    days = 24 * hours  # Days to seconds
    deg = np.pi / 180  # Degrees to radians

    # Constants:
    mu_earth = 398600  # Earth’s gravitational parameter (km^3/s^2)
    mu_moon = 4903  # Moon’s gravitational parameter (km^3/s^2)
    RE = 6378  # Earth’s radius (km)

    # Initial data for each of the three given orbits:
    types = ['GEO', 'HEO', 'LEO']

    # GEO
    n = 1
    a0 = 42164  # semimajor axis (km)
    e0 = 0.0001  # eccentricity
    w0 = 0  # argument of perigee (rad)
    RA0 = 0  # right ascension (rad)
    i0 = 1 * deg  # inclination (rad)
    TA0 = 0  # true anomaly (rad)
    JD0 = 2454283  # Julian Day
    t, RA, i, w = lunar_gravity_perturbation(n, a0, e0, w0, RA0, i0, TA0, JD0, mu_earth, mu_moon)
    print(t[-1])
    print(RA[-1])
    print(i[-1])
    print(w[-1])


    # HEO
    n = 2
    a0 = 26553.4
    e0 = 0.741
    w0 = 270
    RA0 = 0
    i0 = 63.4 * deg
    TA0 = 0
    JD0 = 2454283
    lunar_gravity_perturbation(n, a0, e0, w0, RA0, i0, TA0, JD0)

    # LEO
    n = 3
    a0 = 6678.136
    e0 = 0.01
    w0 = 0
    RA0 = 0
    i0 = 28.5 * deg
    TA0 = 0
    JD0 = 2454283
    lunar_gravity_perturbation(n, a0, e0, w0, RA0, i0, TA0, JD0)