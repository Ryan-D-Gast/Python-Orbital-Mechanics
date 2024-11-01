"""
Author: Ryan Gast
Date: 1/18/2024
Calculates the solar radiation perturbation on orbital elements over time.
    Converted From Matlab to Python from: 
    "Orbital Mechanics for Engineering Students" (2013), 
    Appendix D, D.45.
Version: 1.0
Contact: ryan.d.gast@gmail.com
"""

# WARNING
# RESULT PLOTS SLIGHTLY VARY FROM THE BOOK. THIS MIGHT BE DUE TO THE USE OF SAVGOL_FILTER INSTEAD OR RSMOOTH (FROM MATLAB)
# OR AN UNKNOWN BUG. BECAUSE OF THIS DON'T USE THIS CODE FOR REAL LIFE USE OR ACCURATE RESULTS THIS IS JUST FOR LEARNING PURPOSES.

import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from solar_position import solar_position
from sv_from_coe import sv_from_coe
from los import los
from scipy.integrate import solve_ivp
import pickle
import os.path

def solar_radiation_perturbation(CR, As, m, a0, e0, incl0, RA0, w0, TA0, S=1367, mu=398600, RE=6378, days=24*3600):
    """
    Calculates the solar radiation perturbation on orbital elements over time.

    Parameters:
    - CR (float): Radiation pressure coefficient
    - As (float): Solar panel area (m^2)
    - m (float): Spacecraft mass (kg)
    - a0 (float): Initial semi-major axis (km)
    - e0 (float): Initial eccentricity
    - incl0 (float): Initial inclination (degrees)
    - RA0 (float): Initial right ascension of the ascending node (degrees)
    - w0 (float): Initial argument of periapsis (degrees)
    - TA0 (float): Initial true anomaly (degrees)
    - S (float): Solar constant (W/m^2), default value is 1367
    - mu (float): Gravitational parameter of the central body (km^3/s^2), default value is 398600
    - RE (float): Equatorial radius of the central body (km), default value is 6378
    - days (float): Duration of integration in seconds, default value is 24*3600

    Returns:
    - None
    """
    
    # Conversion factors:
    hours = 3600  # Hours to seconds
    days = 24 * hours  # Days to seconds
    deg = np.pi / 180  # Degrees to radians
    
    # Constants:
    c = 2.998e8  # Speed of light (m/s)
    Psr = S / c  # Solar pressure (Pa)
    
    # Initial orbital parameters (inferred):
    h0 = np.sqrt(mu * a0 * (1 - e0 ** 2))  # Angular momentum (km^2/s)
    T0 = 2 * np.pi / np.sqrt(mu) * a0 ** 1.5  # Period (s)
    #rp0 = h0 ** 2 / mu / (1 + e0)  # Perigee radius (km)
    #ra0 = h0 ** 2 / mu / (1 - e0)  # Apogee radius (km)

    # Store initial orbital elements (from above) in the vector coe0:
    coe0 = [h0, e0, RA0, incl0, w0, TA0]

    # Use odeint to integrate Equations 12.106, the Gauss planetary equations
    # from t0 to tf:
    JD0 = 2438400.5  # Initial Julian date (6 January 1964 0 UT)
    t0 = 0  # Initial time (s)
    tf = 3 * 365 * days  # Final time (s)
    y0 = coe0  # Initial orbital elements
    nout = 4000  # Number of solution points to output
    tspan = np.linspace(t0, tf, nout)  # Integration time interval

    # Define the rates function
    def rates(t, f):
        # Update the Julian Date at time t:
        JD = JD0 + t / days
        # Compute the apparent position vector of the sun:
        lamda, eps, r_sun = solar_position(JD)
        # Convert the ecliptic latitude and the obliquity to radians:
        lamda = lamda * deg
        eps = eps * deg
        # Extract the orbital elements at time t
        h, e, RA, i, w, TA = f
        u = w + TA  # Argument of latitude
        # Compute the state vector at time t:
        coe = [h, e, RA, i, w, TA]
        R, V = sv_from_coe(coe, mu)
        # Calculate the magnitude of the radius vector:
        r = np.linalg.norm(R)
        # Compute the shadow function and the solar radiation perturbation:
        nu = los(R, r_sun)
        Psr = nu * (S / c) * CR * As / m / 1000
        # Calculate the trig functions in Equations 12.105.
        sl, cl, se, ce, sW, cW, si, ci, su, cu, sT, cT = np.sin([lamda, lamda, eps, eps, RA, RA, i, i, u, u, TA, TA])
        # Calculate the earth-sun unit vector components (Equations 12.105):
        ur = sl * ce * cW * ci * su + sl * ce * sW * cu - cl * sW * ci * su + cl * cW * cu + sl * se * si * su
        us = sl * ce * cW * ci * cu - sl * ce * sW * su - cl * sW * ci * cu - cl * cW * su + sl * se * si * cu
        uw = - sl * ce * cW * si + cl * sW * si + sl * se * ci
        # Calculate the time rates of the osculating elements from Equations 12.106:
        hdot = -Psr * r * us
        edot = -Psr * (h / mu * sT * ur + 1 / mu / h * ((h ** 2 + mu * r) * cT + mu * e * r) * us)
        TAdot = h / r ** 2 - Psr / e / h * (h ** 2 / mu * cT * ur - (r + h ** 2 / mu) * sT * us)
        RAdot = -Psr * r / h / si * su * uw
        idot = -Psr * r / h * cu * uw
        wdot = -Psr * (-1 / e / h * (h ** 2 / mu * cT * ur - (r + h ** 2 / mu) * sT * us) - r * su / h / si * ci * uw)
        # Return the rates to odeint:
        dfdt = [hdot, edot, RAdot, idot, wdot, TAdot]
        return dfdt

    # Define the options
    options = {
        'rtol': 1e-8,
        'atol': 1e-8,
        'first_step': T0 / 1000
    }

    # Solve the ODE
    sol = solve_ivp(rates, [tspan[0], tspan[-1]], y0, method='RK45', **options)

    # Extract the solution
    t = sol.t
    y = sol.y.T

    # Extract or compute the orbital elements’ time histories from the
    # solution vector y:
    h = y[:, 0]
    e = y[:, 1]
    RA = y[:, 2]
    incl = y[:, 3]
    w = y[:, 4]
    TA = y[:, 5]
    a = h ** 2 / mu / (1 - e ** 2)
    
    # Smooth the data to remove short period variations:
    h = savgol_filter(h, 51, 3)
    e = savgol_filter(e, 51, 3)
    RA = savgol_filter(RA, 51, 3)
    incl = savgol_filter(incl, 51, 3)
    w = savgol_filter(w, 51, 3)
    a = savgol_filter(a, 51, 3)

    # Plot the results
    plt.figure()

    # Plot Angular Momentum
    plt.subplot(3, 2, 1)
    plt.plot(t / days, h - h0)
    plt.title('Angular Momentum (km^2/s)')
    plt.xlabel('days')

    # Plot Eccentricity
    plt.subplot(3, 2, 2)
    plt.plot(t / days, e - e0)
    plt.title('Eccentricity')
    plt.xlabel('days')

    # Plot Right Ascension
    plt.subplot(3, 2, 3)
    plt.plot(t / days, RA - RA0)
    plt.title('Right Ascension')
    plt.xlabel('days')

    # Plot Inclination
    plt.subplot(3, 2, 4)
    plt.plot(t / days, incl - incl0)
    plt.title('Inclination')
    plt.xlabel('days')

    # Plot Argument of Periapsis
    plt.subplot(3, 2, 5)
    plt.plot(t / days, w - w0)
    plt.title('Argument of Periapsis')
    plt.xlabel('days')

    # Plot Semi-Major Axis
    plt.subplot(3, 2, 6)
    plt.plot(t / days, a - a0)
    plt.title('Semi-Major Axis')
    plt.xlabel('days')

    # Show the plot
    plt.tight_layout()
    plt.show()

    return h, e, RA, incl, w, TA, a

# example usage
if __name__ == '__main__':
    # Conversion factors:
    hours = 3600  # Hours to seconds
    days = 24 * hours  # Days to seconds
    deg = np.pi / 180  # Degrees to radians

    # Constants:
    mu = 398600  # Gravitational parameter (km^3/s^2)
    RE = 6378  # Earth’s radius (km)
    c = 2.998e8  # Speed of light (m/s)
    S = 1367  # Solar constant (W/m^2)
    Psr = S / c  # Solar pressure (Pa)

    # Satellite data:
    CR = 2  # Radiation pressure coefficient
    m = 100  # Mass (kg)
    As = 200  # Frontal area (m^2)

    # Initial orbital parameters (given):
    a0 = 10085.44  # Semimajor axis (km)
    e0 = 0.025422  # Eccentricity
    incl0 = 88.3924 * deg  # Inclination (radians)
    RA0 = 45.38124 * deg  # Right ascension of the node (radians)
    TA0 = 343.4268 * deg  # True anomaly (radians)
    w0 = 227.493 * deg  # Argument of perigee (radians)

    solar_radiation_perturbation(CR, As, m, a0, e0, incl0, RA0, w0, TA0, S=S, mu=mu, RE=RE, days=days)