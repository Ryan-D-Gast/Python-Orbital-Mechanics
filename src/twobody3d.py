"""
Author: Ryan Gast
Date: 12/26/2023
Numerical solution for the motion of two bodies relative to
an inertial frame. Includes the data for Example 2.2
    Converted From Matlab to Python from: 
    "Orbital Mechanics for Engineering Students" (2013), 
    Appendix D, D.5.
Version: 1.0
Contact: ryan.d.gast@gmail.com
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from runge_kutta_fehlberg_45 import rkf45

def twobody3d():
    G = 6.67259e-20
    m1 = 1.e26
    m2 = 1.e26
    t0 = 0
    tf = 480
    R1_0 = np.array([0, 0, 0])
    R2_0 = np.array([3000, 0, 0])
    V1_0 = np.array([10, 20, 30])
    V2_0 = np.array([0, 40, 0])
    y0 = np.concatenate([R1_0, R2_0, V1_0, V2_0])

    def rates(t, y):
        R1 = y[0:3]
        R2 = y[3:6]
        V1 = y[6:9]
        V2 = y[9:12]
        r = np.linalg.norm(R2 - R1)
        A1 = G * m2 * (R2 - R1) / r**3
        A2 = G * m1 * (R1 - R2) / r**3
        dydt = np.concatenate([V1, V2, A1, A2])
        return dydt

    t, y = rkf45(rates, [t0, tf], y0)

    X1 = y[:, 0]
    Y1 = y[:, 1]
    Z1 = y[:, 2]
    X2 = y[:, 3]
    Y2 = y[:, 4]
    Z2 = y[:, 5]

    XG = (m1 * X1 + m2 * X2) / (m1 + m2)
    YG = (m1 * Y1 + m2 * Y2) / (m1 + m2)
    ZG = (m1 * Z1 + m2 * Z2) / (m1 + m2)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(X1, Y1, Z1, '-r')
    ax.plot(X2, Y2, Z2, '-g')
    ax.plot(XG, YG, ZG, '-b')
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    plt.show()

if __name__ == "__main__":
    twobody3d()