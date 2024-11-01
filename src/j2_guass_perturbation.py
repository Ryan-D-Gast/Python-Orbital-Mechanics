"""
Author: Ryan Gast
Date: 1/17/2024
Calculates the J2 perturbations in orbital elements using Gauss' method.
    Converted From Matlab to Python from: 
    "Orbital Mechanics for Engineering Students" (2013), 
    Appendix D, D.42.
Version: 1.0
Contact: ryan.d.gast@gmail.com
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def j2_gauss_pertubation(rp0, ra0, RA0, i0, w0, TA0, mu=398600, RE = 6378, J2=1082.63e-6):
    """
    Computes the perturbations in the orbital elements due to J2 gravity field perturbation.

    Parameters:
    rp0 (float): Perigee radius (km).
    ra0 (float): Apogee radius (km).
    RA0 (float): Right Ascension of the ascending node (degrees).
    i0 (float): Inclination (degrees).
    w0 (float): Argument of perigee (degrees).
    TA0 (float): True anomaly (degrees).
    mu (float, optional): Gravitational parameter of the central body (default is 398600 km^3/s^2).
    RE (float, optional): Equatorial radius of the central body (default is 6378 km).
    J2 (float, optional): J2 coefficient of the central body (default is 1082.63e-6).

    Returns:
    RA (numpy.ndarray): Right Ascension of the ascending node over time (degrees).
    w (numpy.ndarray): Argument of perigee over time (degrees).
    h (numpy.ndarray): Angular momentum over time (km^2/s).
    e (numpy.ndarray): Eccentricity over time.
    i (numpy.ndarray): Inclination over time (degrees).
    """
    
    # Conversion factors:
    hours = 3600  # Hours to seconds
    days = 24 * hours  # Days to seconds
    deg = np.pi / 180  # Degrees to radians

    # Initial orbital parameters (inferred):
    e0 = (ra0 - rp0) / (ra0 + rp0)  # eccentricity
    h0 = np.sqrt(rp0 * mu * (1 + e0))  # angular momentrum (km^2/s)
    a0 = (rp0 + ra0) / 2  # Semimajor axis (km)
    T0 = 2 * np.pi / np.sqrt(mu) * a0 ** 1.5  # Period (s)

    # Store initial orbital elements (from above) in the vector coe0:
    coe0 = np.array([h0, e0, RA0, i0, w0, TA0])

    def rates(t, f):
        """
        Calculate the rates of change of orbital elements at time t.

        Parameters:
        - t: Time
        - f: Array of orbital elements [h, e, RA, i, w, TA]

        Returns:
        - dfdt: Array of rates of change of orbital elements [hdot, edot, RAdot, idot, wdot, TAdot]
        """
        # The orbital elements at time t:
        h, e, RA, i, w, TA = f
        r = h**2 / mu / (1 + e * np.cos(TA))  # The radius
        u = w + TA  # Argument of latitude

        # Orbital element rates at time t (Equations 12.89):
        hdot = -3/2 * J2 * mu * RE**2 / r**3 * np.sin(i)**2 * np.sin(2*u)
        edot = 3/2 * J2 * mu * RE**2 / h / r**3 * (
            h**2 / mu / r * np.sin(u) * np.sin(i)**2 * (3 * np.sin(TA) * np.sin(u) - 2 * np.cos(TA) * np.cos(u)) - np.sin(TA)
            - np.sin(i)**2 * np.sin(2*u) * (e + np.cos(TA))
        )
        RAdot = -3 * J2 * mu * RE**2 / h / r**3 * np.sin(u)**2 * np.cos(i)
        idot = -3/4 * J2 * mu * RE**2 / h / r**3 * np.sin(2*u) * np.sin(2*i)
        wdot = 3/2 * J2 * mu * RE**2 / e / h / r**3 * (
            -h**2 / mu / r * np.cos(TA) * (3 * np.sin(i)**2 * np.sin(u)**2 - 1)
            - np.sin(2*u) * np.sin(i)**2 * np.sin(TA) * (2 + e * np.cos(TA))
            + 2 * e * np.cos(i)**2 * np.sin(u)**2
        )
        TAdot = h / r**2 + 3/2 * J2 * mu * RE**2 / e / h / r**3 * (
            h**2 / mu / r * np.cos(TA) * (3 * np.sin(i)**2 * np.sin(u)**2 - 1)
            + np.sin(2*u) * np.sin(i)**2 * np.sin(TA) * (h**2 / mu / r + 1)
        )

        # Pass these rates back to odeint in the array dfdt:
        dfdt = [hdot, edot, RAdot, idot, wdot, TAdot]
        return dfdt

    # Use odeint to integrate the Gauss variational equations from t0 to tf:
    t0 = 0
    tf = 2 * days
    nout = 5000  # Number of solution points to output for plotting purposes
    tspan = np.linspace(t0, tf, nout)
    options = {'atol': 1e-8, 'rtol': 1e-8, 'first_step': T0/1000}
    result = solve_ivp(rates, (t0, tf), coe0, method='RK45', t_eval=tspan, **options)

    # Assign the time histories mnemonic variable names:
    h, e, RA, i, w, TA = result.y

    # Plot the time histories of the osculating elements:
    plt.figure(1, figsize=(6, 8))   
    plt.subplot(5, 1, 1)
    plt.plot(tspan / hours, (RA - RA0) / deg)
    plt.title('Right Ascension (degrees)')
    plt.xlabel('hours')
    plt.grid(True)

    plt.subplot(5, 1, 2)
    plt.plot(tspan / hours, (w - w0) / deg)
    plt.title('Argument of Perigee (degrees)')
    plt.xlabel('hours')
    plt.grid(True)

    plt.subplot(5, 1, 3)
    plt.plot(tspan / hours, h - h0)
    plt.title('Angular Momentum (km^2/s)')
    plt.xlabel('hours')
    plt.grid(True)

    plt.subplot(5, 1, 4)
    plt.plot(tspan / hours, e - e0)
    plt.title('Eccentricity')
    plt.xlabel('hours')
    plt.grid(True)

    plt.subplot(5, 1, 5)
    plt.plot(tspan / hours, (i - i0) / deg)
    plt.title('Inclination (degrees)')
    plt.xlabel('hours')
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    
    return RA, w, h, e, i

#example usage
if __name__ == '__main__':
    # Conversion factors
    deg = np.pi / 180  # Degrees to radians

    # Constants:
    mu = 398600  # Gravitational parameter (km^3/s^2)
    RE = 6378  # Earth's radius (km)
    J2 = 1082.63e-6  # Earth's J2

    # Initial orbital parameters (given):
    rp0 = RE + 300  # perigee radius (km)
    ra0 = RE + 3062  # apogee radius (km)
    RA0 = 45 * deg  # Right ascencion of the node (radians)
    i0 = 28 * deg  # Inclination (radians)
    w0 = 30 * deg  # Argument of perigee (radians)
    TA0 = 40 * deg  # True anomaly (radians)

    ra, w, h, e, i = j2_gauss_pertubation(rp0, ra0, RA0, i0, w0, TA0)
    print(ra[-1])
    print(w[-1])
    print(h[-1])
    print(e[-1])
    print(i[-1])