"""
Author: Ryan Gast
Date: 1/16/2024
Calculates the J2 perturbations in orbital elements using Encke's method.
    Converted From Matlab to Python from: 
    "Orbital Mechanics for Engineering Students" (2013), 
    Appendix D, D.41.
Version: 1.0
Contact: ryan.d.gast@gmail.com
"""

import numpy as np
from scipy.constants import pi
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from sv_from_coe import sv_from_coe
from rv_from_r0v0 import rv_from_r0v0
from coe_from_sv import coe_from_sv

def j2_enckes_pertubation(zp0, za0, RA0, i0, w0, TA0, mu=398600, RE=6378):
    """
    Computes the perturbations in orbital elements due to J2 gravity field perturbation using Encke's method.

    Args:
        zp0 (float): Perigee altitude (km).
        za0 (float): Apogee altitude (km).
        RA0 (float): Right Ascension of the ascending node (degrees).
        i0 (float): Inclination (degrees).
        w0 (float): Argument of perigee (degrees).
        TA0 (float): True anomaly (degrees).
        mu (float, optional): Gravitational parameter (km^3/s^2). Defaults to 398600.
        RE (float, optional): Earth's radius (km). Defaults to 6378.

    Returns:
        tuple: A tuple containing the following arrays:
            - tsave: Array of solution times (s).
            - RA: Array of right ascension variations (degrees).
            - w: Array of argument of perigee variations (degrees).
            - h: Array of angular momentum variations (km^2/s).
            - e: Array of eccentricity variations.
            - i: Array of inclination variations (degrees).
            - TA: Array of true anomaly variations (degrees).
            - r: Array of position magnitudes (km).
            - v: Array of velocity magnitudes (km/s).
    """
    def rates(t, f):
        """
        Calculates the derivative of the state vector with respect to time.

        Parameters:
        t (float): Time.
        f (numpy.ndarray): State vector.

        Returns:
        numpy.ndarray: Derivative of the state vector.
        """
        del_r = f[:3]  # Position deviation
        del_v = f[3:]  # Velocity deviation

        # Compute the state vector on the osculating orbit at time t
        Rosc, Vosc = rv_from_r0v0(R0, V0, t - t0)

        # Calculate the components of the state vector on the perturbed orbit and their magnitudes
        Rpp = Rosc + del_r
        Vpp = Vosc + del_v
        rosc = np.linalg.norm(Rosc)
        rpp = np.linalg.norm(Rpp)

        # Compute the J2 perturbing acceleration
        xx, yy, zz = Rpp
        fac = 3 / 2 * J2 * (mu / rpp ** 2) * (RE / rpp) ** 2
        ap = -fac * np.array([(1 - 5 * (zz / rpp) ** 2) * (xx / rpp),
                            (1 - 5 * (zz / rpp) ** 2) * (yy / rpp),
                            (3 - 5 * (zz / rpp) ** 2) * (zz / rpp)])

        # Compute the total perturbing acceleration
        F = 1 - (rosc / rpp) ** 3
        del_a = -mu / rosc ** 3 * (del_r - F * Rpp) + ap

        dfdt = np.concatenate((del_v, del_a))  # Return the derivative velocity and acceleration

        return dfdt

    # Constants
    J2 = 1082.63e-6
    deg = pi / 180  # Degrees to radians

    # Initial orbital parameters (inferred)
    rp0 = RE + zp0  # Perigee radius (km)
    ra0 = RE + za0  # Apogee radius (km)
    e0 = (ra0 - rp0) / (ra0 + rp0)  # Eccentricity
    a0 = (ra0 + rp0) / 2  # Semimajor axis (km)
    h0 = np.sqrt(rp0 * mu * (1 + e0))  # Angular momentum (km^2/s)
    T0 = 2 * pi / np.sqrt(mu) * a0 ** 1.5  # Period (s)
    t0 = 0
    tf = 2 * 24 * 3600  # Initial and final time (s)

    # Store the initial orbital elements in the array coe0
    coe0 = [h0, e0, RA0, i0, w0, TA0]

    # Obtain the initial state vector from sv_from_coe
    R0, V0 = sv_from_coe(coe0, mu)  # R0 is the initial position vector
    r0 = np.linalg.norm(R0)
    v0 = np.linalg.norm(V0)  # Magnitudes of T0 and V0
    del_t = T0 / 100  # Time step for Encke procedure

    # Begin the Encke integration
    t = t0  # Initialize the time scalar
    tsave = [t0]  # Initialize the vector of solution times
    y = np.array([np.concatenate((R0, V0))])  # Initialize the state vector
    del_y0 = np.zeros(6)  # Initialize the state vector perturbation
    t += del_t  # First time step

    # Loop over the time interval [t0, tf] with equal increments del_t
    while t <= tf + del_t / 2:
        # Integrate Equation 12.7 over the time increment del_t
        sol = solve_ivp(rates, (t0, t), del_y0, method='RK45', t_eval=[t])

        z = sol.y[:, -1]

        # Compute the osculating state vector at time t
        Rosc, Vosc = rv_from_r0v0(R0, V0, t - t0)

        # Rectify
        R0 = Rosc + z[:3]
        V0 = Vosc + z[3:]
        t0 = t

        # Prepare for next time step
        tsave.append(t)
        y = np.vstack((y, np.concatenate((R0, V0))))
        t += del_t
        del_y0 = np.zeros(6)

    # At each solution time extract the orbital elements from the state vector using coe_from_sv
    n_times = len(tsave)  # n_times is the number of solution times
    r = np.zeros(n_times)
    v = np.zeros(n_times)
    h = np.zeros(n_times)
    e = np.zeros(n_times)
    RA = np.zeros(n_times)
    i = np.zeros(n_times)
    w = np.zeros(n_times)
    TA = np.zeros(n_times)

    for j in range(n_times):
        R = y[j, :3]
        V = y[j, 3:]
        r[j] = np.linalg.norm(R)
        v[j] = np.linalg.norm(V)
        coe = coe_from_sv(R, V, mu)
        h[j] = coe[0]
        e[j] = coe[1]
        RA[j] = coe[2]
        i[j] = coe[3]
        w[j] = coe[4]
        TA[j] = coe[5]

    # Figure 1
    plt.figure(1)
    plt.subplot(2, 1, 1)
    plt.plot(np.array(tsave) / 3600, (RA - RA0) / deg)
    plt.title('Variation of Right Ascension')
    plt.xlabel('hours')
    plt.ylabel('ΔΩ (deg)')
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(np.array(tsave) / 3600, (w - w0) / deg)
    plt.title('Variation of Argument of Perigee')
    plt.xlabel('hours')
    plt.ylabel('Δω (deg)')
    plt.grid(True)

    plt.subplots_adjust(hspace=0.5)  # Adjust the vertical spacing between subplots

    # Figure 2
    plt.figure(2)
    plt.subplot(3, 1, 1)
    plt.plot(np.array(tsave) / 3600, (h - h0))
    plt.title('Variation of Angular Momentum')
    plt.xlabel('hours')
    plt.ylabel('Δh (km^2/s)')
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(np.array(tsave) / 3600, (e - e0))
    plt.title('Variation of Eccentricity')
    plt.xlabel('hours')
    plt.ylabel('Δe')
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(np.array(tsave) / 3600, (i - i0) / deg)
    plt.title('Variation of Inclination')
    plt.xlabel('hours')
    plt.ylabel('Δi (deg)')
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    
    return tsave, RA, w, h, e, i, TA, r, v

# example usage
if __name__ == '__main__':
    # Initial orbital parameters (given)
    zp0 = 300  # Perigee altitude (km)
    za0 = 3062  # Apogee altitude (km)
    RA0 = 45 * (pi / 180)  # Right ascension of the node (radians)
    i0 = 28 * (pi / 180)  # Inclination (radians)
    w0 = 30 * (pi / 180)  # Argument of perigee (radians)
    TA0 = 40 * (pi / 180)  # True anomaly (radians)

    tsave, RA, w, h, e, i, TA, r, v = j2_enckes_pertubation(zp0, za0, RA0, i0, w0, TA0)

    print(tsave[-1])
    print(RA[-1])
    print(w[-1])
    print(h[-1])
    print(e[-1])
    print(i[-1])
    print(TA[-1])
    print(r[-1])
    print(v[-1])
