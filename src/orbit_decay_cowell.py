"""
Author: Ryan Gast
Date: 1/16/2024
Calculates the decay of an orbit using Cowell's method.
    Converted From Matlab to Python from: 
    "Orbital Mechanics for Engineering Students" (2013), 
    Appendix D, D.40.
Version: 1.0
Contact: ryan.d.gast@gmail.com
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from sv_from_coe import sv_from_coe
from atmosphere import atmosphere
from scipy.signal import argrelextrema

def orbit_decay_cowell(rp, ra, RA, i, w, TA, e, a, h, T, mu, RE, wE, CD, m, A):
    """
    Simulates the decay of an orbit using Cowell's method.

    Parameters:
    rp (float): Perigee radius (km)
    ra (float): Apogee radius (km)
    RA (float): Right Ascension of the ascending node (degrees)
    i (float): Inclination (degrees)
    w (float): Argument of perigee (degrees)
    TA (float): True anomaly (degrees)
    e (float): Eccentricity
    a (float): Semi-major axis (km)
    h (float): Specific angular momentum (km^2/s)
    T (float): Orbital period (s)
    mu (float): Gravitational parameter of the central body (km^3/s^2)
    RE (float): Radius of the central body (km)
    wE (float): Angular velocity of the central body (rad/s)
    CD (float): Drag coefficient
    m (float): Mass of the spacecraft (kg)
    A (float): Cross-sectional area of the spacecraft (m^2)

    Returns:
    perigee (ndarray): Array of perigee altitudes and times (km, s)
    apogee (ndarray): Array of apogee altitudes and times (km, s)
    maxima (ndarray): Array of maximum altitudes and times (km, s)
    minima (ndarray): Array of minimum altitudes and times (km, s)
    sol (OdeResult): Solution of the equations of motion
    """
    def rates(t, f):
        """
        Calculates the rates of change of position and velocity vectors for orbit decay simulation.

        Parameters:
        t (float): Time parameter.
        f (ndarray): State vector containing position and velocity vectors.

        Returns:
        ndarray: Concatenation of velocity and acceleration vectors.
        """
        global alt
        R = f[:3]  # Position vector (km/s)
        r = np.linalg.norm(R)  # Distance from earth’s center (km)
        alt = r - RE  # Altitude (km)
        rho = atmosphere(alt)  # Air density from US Standard Model (kf/m^3)
        V = f[3:]  # Velocity vector (km/s)
        Vrel = V - np.cross(wE, R)  # Velocity relative to the atmosphere (km/s)
        vrel = np.linalg.norm(Vrel)  # Speed relative to the atmosphere (km/s)
        uv = Vrel / vrel  # Relative velocity unit vector
        ap = -CD * A / m * rho * (1000 * vrel) ** 2 / 2 * uv  # Acceleration due to drag (m/s^2)
        a0 = -mu * R / r ** 3  # Gravitational acceleration (km/s^2)
        a = a0 + ap / 1000  # Total acceleration (km/s^2)
        return np.concatenate((V, a))  # Velocity and the acceleration returned to solve_ivp

    def terminate(t, y):
        """
        Terminate the simulation if the altitude is below 100 km.

        Parameters:
        t (float): Current time of the simulation.
        y (numpy.ndarray): State vector containing the position and velocity.

        Returns:
        float: The difference between the altitude and 100 km.
        """
        global alt
        R = y[:3]  # Position vector (km)
        r = np.linalg.norm(R)  # Distance from earth’s center (km)
        alt = r - RE  # Altitude (km)
        return alt - 100  # Terminate if altitude is below 100 km

    terminate.terminal = True
    terminate.direction = -1

    # Conversion factors
    hours = 3600  # Hours to seconds
    days = 24 * hours  # Days to seconds
    deg = np.pi / 180  # Degrees to radians

    # Store initial orbital elements (from above) in the vector coe0
    coe0 = [h, e, RA, i, w, TA]

    # Obtain the initial state vector from Algorithm 4.5 (sv_from_coe)
    R0, V0 = sv_from_coe(coe0, mu)  # R0 is the initial position vector, V0 is the initial velocity vector
    r0 = np.linalg.norm(R0)
    v0 = np.linalg.norm(V0)

    # Use solve_ivp to integrate the equations of motion d/dt(R,V) = f(R,V) from t0 to tf
    t0 = 0
    tf = 120 * days  # Initial and final times (s)
    y0 = np.concatenate((R0, V0))  # Initial state vector
    nout = 40000  # Number of solution points to output
    tspan = np.linspace(t0, tf, nout)  # Integration time interval

    # Set error tolerances, initial step size, and termination event
    options = {'rtol': 1.e-8, 'atol': 1.e-8, 'first_step': T / 100, 'events': terminate}

    # Altitude
    alt = 0 

    # Integrate the equations of motion
    sol = solve_ivp(rates, (t0, tf), y0, t_eval=tspan, **options)

            
    # Extract the locally extreme altitudes
    altitude = np.sqrt(np.sum(sol.y[:3] ** 2, axis=0)) - RE  # Altitude at each time
    max_altitude = np.max(altitude)
    imax = np.argmax(altitude)
    min_altitude = np.min(altitude)
    imin = np.argmin(altitude)
    maxima = np.column_stack((sol.t[imax], max_altitude))  # Maximum altitudes and times
    minima = np.column_stack((sol.t[imin], min_altitude))  # Minimum altitudes and times

    # Define a comparator function for argrelextrema
    kmin = argrelextrema(altitude, np.less)
    perigee = np.column_stack((sol.t[kmin], altitude[kmin]))
    kmax = argrelextrema(altitude, np.greater)
    apogee = np.column_stack((sol.t[kmax], altitude[kmax]))

    # Plot perigee and apogee history on the same figure
    plt.figure()
    plt.plot(apogee[:, 0] / days, apogee[:, 1], 'b', linewidth=2)
    plt.plot(perigee[:, 0] / days, perigee[:, 1], 'r', linewidth=2)
    plt.grid(True, which='both')
    plt.xlabel('Time (days)')
    plt.ylabel('Altitude (km)')
    plt.ylim([0, 1000])
    plt.show()
    
    return perigee, apogee, maxima, minima, imax, imin, sol
    
# example usage
if __name__ == '__main__':
    # Constants
    mu = 398600  # Gravitational parameter (km^3/s^2)
    RE = 6378  # Earth's radius (km)
    wE = np.array([0, 0, 7.2921159e-5])  # Earth's angular velocity (rad/s)

    # Conversion factors
    hours = 3600  # Hours to seconds
    days = 24 * hours  # Days to seconds
    deg = np.pi / 180  # Degrees to radians

    # Satellite data
    CD = 2.2  # Drag coefficient
    m = 100  # Mass (kg)
    A = np.pi / 4 * (1 ** 2)  # Frontal area (m^2)

    # Initial orbital parameters (given)
    rp = RE + 215  # perigee radius (km)
    ra = RE + 939  # apogee radius (km)
    RA = 339.94 * deg  # Right ascension of the node (radians)
    i = 65.1 * deg  # Inclination (radians)
    w = 58 * deg  # Argument of perigee (radians)
    TA = 332 * deg  # True anomaly (radians)

    # Initial orbital parameters (inferred)
    e = (ra - rp) / (ra + rp)  # eccentricity
    a = (rp + ra) / 2  # Semimajor axis (km)
    h = np.sqrt(mu * a * (1 - e ** 2))  # angular momentum (km^2/s)
    T = 2 * np.pi / np.sqrt(mu) * a ** 1.5  # Period (s)

    # Run the simulation
    perigee, apogee, maxima, minima, imax, imin, sol = orbit_decay_cowell(rp, ra, RA, i, w, TA, e, a, h, T, mu, RE, wE, CD, m, A)

    # Print the results
    print(f"Apogee Maxima: {maxima[0][0]} km")
    print(f"Apogee Minima: {minima[0][0]} km")
    print(f"Perigee Maxima: {maxima[0][1]} km")
    print(f"Perigee Minima: {minima[0][1]} km")
    print(f"imax: {imax}")
    print(f"imin: {imin}")