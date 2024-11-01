"""
Author: Ryan Gast
Date: 12/30/2023
This function uses the Lambert's problem to compute the velocity vectors
    Converted From Matlab to Python from: 
    "Orbital Mechanics for Engineering Students" (2013), 
    Appendix D, D.25.
Version: 1.0
Contact: ryan.d.gast@gmail.com
"""

import numpy as np
from scipy.optimize import newton
from stumpSC import stumpS, stumpC

def lambert(R1, R2, t, string='pro', mu = 398600):
    """
    Computes the Lambert's problem solution for two position vectors R1 and R2 at a given time t.

    Parameters:
    R1 (numpy.ndarray): Initial position vector.
    R2 (numpy.ndarray): Final position vector.
    t (float): Time of flight.
    string (str, optional): Determines the type of transfer. Default is 'pro' (prograde) 'retro' is other option.

    Returns:
    tuple: A tuple containing the velocity vectors V1 and V2.

    """
    
    # Subfunctions
    def y(z):
        """
        Calculate the value of y based on the given parameter z.

        Parameters:
        z (float): The input parameter.

        Returns:
        float: The calculated value of y.
        """
        return r1 + r2 + A*(z*stumpS(z) - 1)/np.sqrt(stumpC(z))

    def F(z, t):
        """
        Calculate the value of F(z, t) for the Lambert's problem.

        Parameters:
        z (float): The value of z.
        t (float): The value of t.

        Returns:
        float: The calculated value of F(z, t).
        """
        return (y(z)/stumpC(z))**1.5*stumpS(z) + A*np.sqrt(y(z)) - np.sqrt(mu)*t

    def dFdz(z, t):
        """
        Calculate the derivative of F with respect to z.

        Parameters:
        z (float): The independent variable.
        t (float): The dependent variable.

        Returns:
        float: The derivative of F with respect to z.
        """
        if z == 0:
            return np.sqrt(2)/40*y(0)**1.5 + A/8*(np.sqrt(y(0)) + A*np.sqrt(1/2/y(0)))
        else:
            return (y(z)/stumpC(z))**1.5*(1/2/z*(stumpC(z) - 3*stumpS(z)/2/stumpC(z)) + 3*stumpS(z)**2/4/stumpC(z)) + A/8*(3*stumpS(z)/stumpC(z)*np.sqrt(y(z)) + A*np.sqrt(stumpC(z)/y(z)))
    
    r1 = np.linalg.norm(R1)
    r2 = np.linalg.norm(R2)
    c12 = np.cross(R1, R2)
    theta = np.arccos(np.dot(R1, R2)/r1/r2)

    if string == 'pro':
        if c12[2] <= 0:
            theta = 2*np.pi - theta
    elif string == 'retro':
        if c12[2] >= 0:
            theta = 2*np.pi - theta

    A = np.sin(theta)*np.sqrt(r1*r2/(1 - np.cos(theta)))

    z = newton(F, x0=0, fprime=dFdz, args=(t,), tol=1e-8, maxiter=5000)

    f = 1 - y(z)/r1
    g = A*np.sqrt(y(z)/mu)
    gdot = 1 - y(z)/r2

    V1 = 1/g*(R2 - f*R1)
    V2 = 1/g*(gdot*R2 - R1)

    return V1, V2

# exmaple usage:
if __name__ == '__main__':
    from coe_from_sv import coe_from_sv
    # Constants
    deg = np.pi/180
    mu = 398600

    # Data declaration for Example 5.2:
    r1 = np.array([5000, 10000, 2100])
    r2 = np.array([-14600, 2500, 7000])
    dt = 3600
    string = 'pro'

    # Algorithm 5.2:
    v1, v2 = lambert(r1, r2, dt, string)

    # Algorithm 4.1 (using r1 and v1):
    coe = coe_from_sv(r1, v1, mu)

    # Save the initial true anomaly:
    TA1 = coe[5]

    # Algorithm 4.1 (using r2 and v2):
    coe = coe_from_sv(r2, v2, mu)

    # Save the final true anomaly:
    TA2 = coe[5]

    # Echo the input data and output the results to the command window:
    print('Example 5.2: Lambert\'s Problem')
    print('\nInput data:')
    print('\nGravitational parameter (km^3/s^2) =', mu)
    print('\nr1 (km) =', r1)
    print('\nr2 (km) =', r2)
    print('\nElapsed time (s) =', dt)
    print('\nSolution:')
    print('\nv1 (km/s) =', v1)
    print('\nv2 (km/s) =', v2)
    print('\nOrbital elements:')
    print('\nAngular momentum (km^2/s) =', coe[0])
    print('\nEccentricity =', coe[1])
    print('\nInclination (deg) =', coe[3]/deg)
    print('\nRA of ascending node (deg) =', coe[2]/deg)
    print('\nArgument of perigee (deg) =', coe[4]/deg)
    print('\nTrue anomaly initial (deg) =', TA1/deg)
    print('\nTrue anomaly final (deg) =', TA2/deg)
    print('\nSemimajor axis (km) =', coe[6])
    print('\nPeriapse radius (km) =', coe[0]**2/mu/(1 + coe[1]))

    # If the orbit is an ellipse, output its period:
    if coe[1] < 1:
        T = 2*np.pi/np.sqrt(mu)*coe[6]**1.5
        print('\nPeriod:')
        print('\nSeconds =', T)
        print('\nMinutes =', T/60)
        print('\nHours =', T/3600)
        print('\nDays =', T/24/3600)