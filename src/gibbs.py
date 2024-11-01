"""
Author: Ryan Gast
Date: 12/30/2023
This function uses the Gibbs method of orbit determination to
compute the velocity corresponding to the second of three supplied position vectors.
    Converted From Matlab to Python from: 
    "Orbital Mechanics for Engineering Students" (2013), 
    Appendix D, D.24.
Version: 1.0
Contact: ryan.d.gast@gmail.com
"""

import numpy as np
from coe_from_sv import coe_from_sv

def gibbs(R1, R2, R3, mu=398600):
    """
    Computes the velocity vector at R2 using the Gibbs method.

    Parameters:
    - R1 (numpy.ndarray): Position vector at time t1.
    - R2 (numpy.ndarray): Position vector at time t2.
    - R3 (numpy.ndarray): Position vector at time t3.
    - mu (float): Gravitational parameter of the central body (default = 398600).

    Returns:
    - V2 (numpy.ndarray): Velocity vector at R2.
    - ierr (int): Error flag. Set to 1 if R1, R2, and R3 are not coplanar, otherwise set to 0.
    """
    tol = 1e-4
    ierr = 0

    # Magnitudes of R1, R2 and R3:
    r1 = np.linalg.norm(R1)
    r2 = np.linalg.norm(R2)
    r3 = np.linalg.norm(R3)

    # Cross products among R1, R2 and R3:
    c12 = np.cross(R1, R2)
    c23 = np.cross(R2, R3)
    c31 = np.cross(R3, R1)

    # Check that R1, R2 and R3 are coplanar; if not set error flag:
    if abs(np.dot(R1, c23) / r1 / np.linalg.norm(c23)) > tol:
        ierr = 1
        return None, ierr

    # Equation 5.13:
    N = r1 * c23 + r2 * c31 + r3 * c12

    # Equation 5.14:
    D = c12 + c23 + c31

    # Equation 5.21:
    S = R1 * (r2 - r3) + R2 * (r3 - r1) + R3 * (r1 - r2)

    # Equation 5.22:
    V2 = np.sqrt(mu / np.linalg.norm(N) / np.linalg.norm(D)) * (np.cross(D, R2) / r2 + S)

    return V2, ierr

# example usage:
if __name__ == '__main__':
    # Data declaration for Example 5.1:
    mu = 398600
    r1 = np.array([-294.32, 4265.1, 5986.7])
    r2 = np.array([-1365.5, 3637.6, 6346.8])
    r3 = np.array([-2940.3, 2473.7, 6555.8])

    # Echo the input data to the command window:
    print('Example 5.1: Gibbs Method')
    print('\nInput data:')
    print('\nGravitational parameter (km^3/s^2) =', mu)
    print('\nr1 (km) =', r1)
    print('\nr2 (km) =', r2)
    print('\nr3 (km) =', r3)

    # Algorithm 5.1:
    v2, ierr = gibbs(r1, r2, r3)

    # If the vectors r1, r2, r3, are not coplanar, abort:
    if ierr == 1:
        print('\nThese vectors are not coplanar.\n')
        exit()

    # Algorithm 4.2:
    # coe_from_sv is a user-defined function that you should have defined in the same Python file or imported from another Python file
    coe = coe_from_sv(r2, v2, mu)
    h, e, RA, incl, w, TA, a = coe

    # Output the results to the command window:
    print('Solution:')
    print('\nv2 (km/s) =', v2)
    print('\nOrbital elements:')
    print('\nAngular momentum (km^2/s) =', h)
    print('\nEccentricity =', e)
    print('\nInclination (deg) =', np.degrees(incl))
    print('\nRA of ascending node (deg) =', np.degrees(RA))
    print('\nArgument of perigee (deg) =', np.degrees(w))
    print('\nTrue anomaly (deg) =', np.degrees(TA))
    print('\nSemimajor axis (km) =', a)

    # If the orbit is an ellipse, output the period:
    if e < 1:
        T = 2 * np.pi / np.sqrt(mu) * a ** 1.5
        print('\nPeriod (s) =', T)