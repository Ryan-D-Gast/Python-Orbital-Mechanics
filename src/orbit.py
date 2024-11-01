"""
Author: Ryan Gast
Date: 12/26/2023
Numerical solution of the two-body relative motion
problem. Includes the data from Example 2.3.
    Converted From Matlab to Python from: 
    "Orbital Mechanics for Engineering Students" (2013), 
    Appendix D, D.6.
Version: 1.0
Contact: ryan.d.gast@gmail.com
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime

# Constants
hours = 3600
G = 6.6742e-20

# Input data
m1 = 5.974e24
R = 6378
m2 = 1000
r0 = np.array([8000, 0, 6000])
v0 = np.array([0, 7, 0])
t0 = 0
tf = 4 * hours
mu = G * (m1 + m2)
y0 = np.concatenate([r0, v0])

# Rates function
def rates(t, y):
    r = np.linalg.norm(y[0:3])
    a = -mu * y[0:3] / r**3
    dydt = np.concatenate([y[3:6], a])
    return dydt

# Solve ODE
sol = solve_ivp(rates, [t0, tf], y0, method='RK45')

# Output function
def output(t, y):
    """
    Prints the results of the orbit simulation and plots the trajectory.

    Parameters:
    t (array-like): Array of time values.
    y (array-like): Array of state vectors.

    Returns:
    None
    """
    r = np.linalg.norm(y[:, 0:3], axis=1)
    imax = np.argmax(r)
    imin = np.argmin(r)
    rmax = r[imax]
    rmin = r[imin]
    v_at_rmax = np.linalg.norm(y[imax, 3:6])
    v_at_rmin = np.linalg.norm(y[imin, 3:6])

    # Print results
    print(f"\n Earth Orbit\n {datetime.now()}")
    print(f"\n The initial position is {r0} (km).")
    print(f"\n Magnitude = {np.linalg.norm(r0)} km")
    print(f"\n The initial velocity is {v0} (km/s).")
    print(f"\n Magnitude = {np.linalg.norm(v0)} km/s")
    print(f"\n Initial time = {t0 / hours} h.\n Final time = {tf / hours} h.")
    print(f"\n The minimum altitude is {rmin - R} km at time = {t[imin] / hours} h.")
    print(f"\n The speed at that point is {v_at_rmin} km/s.")
    print(f"\n The maximum altitude is {rmax - R} km at time = {t[imax] / hours} h.")
    print(f"\n The speed at that point is {v_at_rmax} km/s")

    # Plot results
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(y[:, 0], y[:, 1], y[:, 2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

output(sol.t, sol.y.T)