"""
Author: Ryan Gast
Date: 12/31/2023
Calculate the position vector (r) and velocity vector (v) 
of an observed object in a rotating Earth frame using the Gauss method.
    Converted From Matlab to Python from: 
    "Orbital Mechanics for Engineering Students" (2013), 
    Appendix D, D.29.
Version: 1.0
Contact: ryan.d.gast@gmail.com
"""

import numpy as np
from Kepler_U import kepler_U
from f_and_g import f_and_g

def gauss(Rho1, Rho2, Rho3, R1, R2, R3, t1, t2, t3, mu=398600):
    tau1 = t1 - t2
    tau3 = t3 - t2
    tau = tau3 - tau1

    p1 = np.cross(Rho2, Rho3)
    p2 = np.cross(Rho1, Rho3)
    p3 = np.cross(Rho1, Rho2)

    Do = np.dot(Rho1, p1)
    D = np.array([[np.dot(R1, p1), np.dot(R1, p2), np.dot(R1, p3)],
                  [np.dot(R2, p1), np.dot(R2, p2), np.dot(R2, p3)],
                  [np.dot(R3, p1), np.dot(R3, p2), np.dot(R3, p3)]])

    E = np.dot(R2, Rho2)

    A = 1/Do*(-D[0,1]*tau3/tau + D[1,1] + D[2,1]*tau1/tau)
    B = 1/6/Do*(D[0,1]*(tau3**2 - tau**2)*tau3/tau + D[2,1]*(tau**2 - tau1**2)*tau1/tau)

    a = -(A**2 + 2*A*E + np.linalg.norm(R2)**2)
    b = -2*mu*B*(A + E)
    c = -(mu*B)**2

    Roots = np.roots([1, 0, a, 0, 0, b, 0, 0, c])
    x = posroot(Roots)  # You need to define this function

    f1 = 1 - 1/2*mu*tau1**2/x**3
    f3 = 1 - 1/2*mu*tau3**2/x**3

    g1 = tau1 - 1/6*mu*(tau1/x)**3
    g3 = tau3 - 1/6*mu*(tau3/x)**3

    rho2 = A + mu*B/x**3
    rho1 = 1/Do*((6*(D[2,0]*tau1/tau3 + D[1,0]*tau/tau3)*x**3 + mu*D[2,0]*(tau**2 - tau1**2)*tau1/tau3)/(6*x**3 + mu*(tau**2 - tau3**2)) - D[0,0])
    rho3 = 1/Do*((6*(D[0,2]*tau3/tau1 - D[1,2]*tau/tau1)*x**3 + mu*D[0,2]*(tau**2 - tau3**2)*tau3/tau1)/(6*x**3 + mu*(tau**2 - tau1**2)) - D[2,2])

    r1 = R1 + rho1*Rho1
    r2 = R2 + rho2*Rho2
    r3 = R3 + rho3*Rho3

    v2 = (-f3*r1 + f1*r3)/(f1*g3 - f3*g1)

    r_old = r2
    v_old = v2

    # iterative improvement
    rho1_old = rho1
    rho2_old = rho2
    rho3_old = rho3
    diff1 = 1
    diff2 = 1
    diff3 = 1
    n = 0
    nmax = 1000
    tol = 1.e-8

    while ((diff1 > tol) and (diff2 > tol) and (diff3 > tol)) and (n < nmax):
        n += 1
        ro = np.linalg.norm(r2)
        vo = np.linalg.norm(v2)
        vro = np.dot(v2, r2) / ro
        a = 2 / ro - vo**2 / mu

        x1 = kepler_U(tau1, ro, vro, a)  # You need to define this function
        x3 = kepler_U(tau3, ro, vro, a)  # You need to define this function
        
        ff1, gg1 = f_and_g(x1, tau1, ro, a)  # You need to define this function
        ff3, gg3 = f_and_g(x3, tau3, ro, a)  # You need to define this function

        f1 = (f1 + ff1) / 2
        f3 = (f3 + ff3) / 2
        g1 = (g1 + gg1) / 2
        g3 = (g3 + gg3) / 2

        c1 = g3 / (f1 * g3 - f3 * g1)
        c3 = -g1 / (f1 * g3 - f3 * g1)

        rho1 = 1 / Do * (-D[0, 0] + 1 / c1 * D[1, 0] - c3 / c1 * D[2, 0])
        rho2 = 1 / Do * (-c1 * D[0, 1] + D[1, 1] - c3 * D[2, 1])
        rho3 = 1 / Do * (-c1 / c3 * D[0, 2] + 1 / c3 * D[1, 2] - D[2, 2])

        r1 = R1 + rho1 * Rho1
        r2 = R2 + rho2 * Rho2
        r3 = R3 + rho3 * Rho3

        v2 = (-f3 * r1 + f1 * r3) / (f1 * g3 - f3 * g1)

        diff1 = abs(rho1 - rho1_old)
        diff2 = abs(rho2 - rho2_old)
        diff3 = abs(rho3 - rho3_old)

        rho1_old = rho1
        rho2_old = rho2
        rho3_old = rho3

    print(f"\n( **Number of Gauss improvement iterations = {n})\n\n")

    if n >= nmax:
        print(f"\n\n **Number of iterations exceeds {nmax} \n\n")

    r = r2
    v = v2

    return r, v, r_old, v_old

def posroot(Roots):
    """
    Returns the positive real root from a list of complex roots.

    Parameters:
    Roots (list): A list of complex roots.

    Returns:
    float: The positive real root.

    Raises:
    None

    """
    # Construct the vector of positive real roots:
    posroots = [root.real for root in Roots if root > 0 and root.imag == 0]
    npositive = len(posroots)

    # Exit if no positive roots exist:
    if npositive == 0:
        print('\n\n ** There are no positive real roots. \n\n')
        return None

    # If there is more than one positive root, output the
    # roots to the command window and prompt the user to
    # select which one to use:
    if npositive == 1:
        x = posroots[0]
    else:
        print('\n\n ** There are two or more positive real roots.\n')
        for i, root in enumerate(posroots):
            print(f'\n root #{i+1} = {root}')

        print('\n\n Make a choice:\n')
        nchoice = 0
        while nchoice < 1 or nchoice > npositive:
            nchoice = int(input(' Use root #? '))
        x = posroots[nchoice-1]
        print(f'\n We will use {x} .\n')

    return x

# example usage
if __name__ == "__main__":
    from coe_from_sv import coe_from_sv

    # Constants
    deg = np.pi/180
    mu = 398600
    Re = 6378
    f = 1/298.26

    # Data declaration for Example 5.11
    H = 1
    phi = 40*deg
    t = np.array([0, 118.104, 237.577])
    ra = np.array([43.5365, 54.4196, 64.3178])*deg
    dec = np.array([-8.78334, -12.0739, -15.1054])*deg
    theta = np.array([44.5065, 45.000, 45.4992])*deg

    # Equations 5.64, 5.76 and 5.79
    fac1 = Re/np.sqrt(1-(2*f - f*f)*np.sin(phi)**2)
    fac2 = (Re*(1-f)**2/np.sqrt(1-(2*f - f*f)*np.sin(phi)**2) + H)*np.sin(phi)
    R = np.zeros((3, 3))
    rho = np.zeros((3, 3))

    for i in range(3):
        R[i, :] = [(fac1 + H)*np.cos(phi)*np.cos(theta[i]), (fac1 + H)*np.cos(phi)*np.sin(theta[i]), fac2]
        rho[i, :] = [np.cos(dec[i])*np.cos(ra[i]), np.cos(dec[i])*np.sin(ra[i]), np.sin(dec[i])]

    # Algorithms 5.5 and 5.6
    # Assuming gauss and coe_from_sv are defined functions
    r, v, r_old, v_old = gauss(rho[0, :], rho[1, :], rho[2, :], R[0, :], R[1, :], R[2, :], t[0], t[1], t[2], mu)
    coe_old = coe_from_sv(r_old, v_old, mu)
    coe = coe_from_sv(r, v, mu)

    # Print the input data and output the solution
    print('\n Example 5.11: Orbit determination by the Gauss method\n')
    print('\n Radius of earth (km) = {}'.format(Re))
    print('\n Flattening factor = {}'.format(f))
    print('\n Gravitational parameter (km^3/s^2) = {}'.format(mu))
    print('\n\n Input data:\n')
    print('\n Latitude (deg) = {}'.format(phi/deg))
    print('\n Altitude above sea level (km) = {}'.format(H))
    print('\n\n Observations:')
    print('\n Right')
    print(' Local')
    print('\n Time (s) Ascension (deg) Declination (deg)')
    print(' Sidereal time (deg)')
    for i in range(3):
        print('\n {:9.4f} {:11.4f} {:19.4f} {:20.4f}'.format(t[i], ra[i]/deg, dec[i]/deg, theta[i]/deg))
    print('\n\n Solution:\n')
    print('\n Without iterative improvement...\n')
    print('\n')
    print('\n r (km) = [{}, {}, {}]'.format(r_old[0], r_old[1], r_old[2]))
    print('\n v (km/s) = [{}, {}, {}]'.format(v_old[0], v_old[1], v_old[2]))
    print('\n')
    print('\n Angular momentum (km^2/s) = {}'.format(coe_old[0]))
    print('\n Eccentricity = {}'.format(coe_old[1]))
    print('\n RA of ascending node (deg) = {}'.format(coe_old[2]/deg))
    print('\n Inclination (deg) = {}'.format(coe_old[3]/deg))
    print('\n Argument of perigee (deg) = {}'.format(coe_old[4]/deg))
    print('\n True anomaly (deg) = {}'.format(coe_old[5]/deg))
    print('\n Semimajor axis (km) = {}'.format(coe_old[6]))
    print('\n Periapse radius (km) = {}'.format(coe_old[0]**2 /mu/(1 + coe_old[1])))
    if coe_old[1]<1:
        T = 2*np.pi/np.sqrt(mu)*coe_old[6]**1.5
        print('\n Period:')
        print('\n Seconds = {}'.format(T))
        print('\n Minutes = {}'.format(T/60))
        print('\n Hours = {}'.format(T/3600))
        print('\n Days = {}'.format(T/24/3600))
    print('\n\n With iterative improvement...\n')
    print('\n')
    print('\n r (km) = [{}, {}, {}]'.format(r[0], r[1], r[2]))
    print('\n v (km/s) = [{}, {}, {}]'.format(v[0], v[1], v[2]))
    print('\n')
    print('\n Angular momentum (km^2/s) = {}'.format(coe[0]))
    print('\n Eccentricity = {}'.format(coe[1]))
    print('\n RA of ascending node (deg) = {}'.format(coe[2]/deg))
    print('\n Inclination (deg) = {}'.format(coe[3]/deg))
    print('\n Argument of perigee (deg) = {}'.format(coe[4]/deg))
    print('\n True anomaly (deg) = {}'.format(coe[5]/deg))
    print('\n Semimajor axis (km) = {}'.format(coe[6]))
    print('\n Periapse radius (km) = {}'.format(coe[0]**2 /mu/(1 + coe[1])))
    if coe[1]<1:
        T = 2*np.pi/np.sqrt(mu)*coe[6]**1.5
        print('\n Period:')
        print('\n Seconds = {}'.format(T))
        print('\n Minutes = {}'.format(T/60))
        print('\n Hours = {}'.format(T/3600))
        print('\n Days = {}'.format(T/24/3600))