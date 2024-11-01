"""
Author: Ryan Gast
Date: 1/4/2024
Calculate the state vector at the end of a finite time,
constant thrust delta-v maneuver.
    Converted From Matlab to Python from: 
    "Orbital Mechanics for Engineering Students" (2013), 
    Appendix D, D.30.
Version: 1.0
Contact: ryan.d.gast@gmail.com
"""

import numpy as np
from scipy.integrate import solve_ivp
from coe_from_sv import coe_from_sv
from rv_from_r0v0_ta import rv_from_r0v0_ta

def integrate_thrust(r0, v0, t0, t_burn, m0, T, Isp, mu, deg, g0):
    """
    Integrate the thrust acceleration to compute the state vector and mass after the burn,
    as well as the orbital elements of the post-burn trajectory.

    Parameters:
    - r0 (numpy array): Initial position vector [x, y, z]
    - v0 (numpy array): Initial velocity vector [vx, vy, vz]
    - t0 (float): Initial time
    - t_burn (float): Duration of the thrust burn
    - m0 (float): Initial mass
    - T (float): Thrust magnitude
    - Isp (float): Specific impulse
    - mu (float): Gravitational parameter of the central body
    - deg (float): Conversion factor from radians to degrees
    - g0 (float): Standard gravity acceleration at sea level

    Returns:
    - r1 (numpy array): Final position vector after the burn
    - v1 (numpy array): Final velocity vector after the burn
    - m1 (float): Final mass after the burn
    - e (float): Eccentricity of the post-burn trajectory
    - a (float): Semimajor axis of the post-burn trajectory
    - ra (numpy array): Position vector at apogee of the post-burn trajectory
    - va (numpy array): Velocity vector at apogee of the post-burn trajectory
    - rmax (float): Distance from the central body at apogee of the post-burn trajectory
    - vmax (float): Velocity magnitude at apogee of the post-burn trajectory
    """  
    # Initial conditions
    y0 = np.concatenate((r0, v0, [m0]))

    # Define the system of differential equations
    def rates(t, f):
        """
        Calculate the rates of change for the state vector.

        Parameters:
        t (float): Time.
        f (list): State vector [x, y, z, vx, vy, vz, m].

        Returns:
        list: Rates of change [vx, vy, vz, ax, ay, az, mdot].
        """
        x, y, z, vx, vy, vz, m = f
        r = np.linalg.norm([x, y, z])
        v = np.linalg.norm([vx, vy, vz])
        ax = -mu*x/r**3 + T/m*vx/v
        ay = -mu*y/r**3 + T/m*vy/v
        az = -mu*z/r**3 + T/m*vz/v
        mdot = -T*1000/g0/Isp
        return [vx, vy, vz, ax, ay, az, mdot]

    # Solve the system of differential equations
    sol = solve_ivp(rates, [t0, t_burn], y0, rtol=1.e-8)

    # Compute the state vector and mass after the burn
    r1 = sol.y[:3, -1]
    v1 = sol.y[3:6, -1]
    m1 = sol.y[6, -1]

    # Compute the orbital elements of the post-burn trajectory
    coe = coe_from_sv(r1, v1, mu)  # You need to define this function
    e = coe[1]  # eccentricity
    TA = coe[5]  # true anomaly (radians)
    a = coe[6]  # semimajor axis

    # Find the state vector at apogee of the post-burn trajectory
    if TA <= np.pi:
        dtheta = np.pi - TA
    else:
        dtheta = 3*np.pi - TA

    ra, va = rv_from_r0v0_ta(r1, v1, dtheta/deg, mu)  # You need to define this function
    rmax = np.linalg.norm(ra)
    vmax = np.linalg.norm(va)
    
    return r1, v1, m1, e, a, ra, va, rmax, vmax

# example usage
if __name__ == '__main__':
    # Constants
    deg = np.pi/180
    mu = 398600
    RE = 6378
    g0 = 9.807

    # Input data
    r0 = np.array([RE+480, 0, 0])
    v0 = np.array([0, 7.7102, 0])
    t0 = 0
    t_burn = 261.1127
    m0 = 2000
    T = 10
    Isp = 300

    # Call the integrate_thrust function
    r1, v1, m1, e, a, ra, va, rmax, vmax = integrate_thrust(r0, v0, t0, t_burn, m0, T, Isp, mu, deg, g0)

    print('\n\n' + '-'*70)
    print('\nBefore ignition:')
    print(f'\n Mass = {m0} kg')
    print('\n State vector:')
    print(f'\n r = [{r0[0]:10g}, {r0[1]:10g}, {r0[2]:10g}] (km)')
    print(f'\n Radius = {np.linalg.norm(r0)}')
    print(f'\n v = [{v0[0]:10g}, {v0[1]:10g}, {v0[2]:10g}] (km/s)')
    print(f'\n Speed = {np.linalg.norm(v0)}\n')
    print(f'\nThrust = {T:12g} kN')
    print(f'\nBurn time = {t_burn:12.6f} s')
    print(f'\nMass after burn = {m1:12.6E} kg\n')
    print('\nEnd-of-burn-state vector:')
    print(f'\n r = [{r1[0]:10g}, {r1[1]:10g}, {r1[2]:10g}] (km)')
    print(f'\n Radius = {np.linalg.norm(r1)}')
    print(f'\n v = [{v1[0]:10g}, {v1[1]:10g}, {v1[2]:10g}] (km/s)')
    print(f'\n Speed = {np.linalg.norm(v1)}\n')
    print('\nPost-burn trajectory:')
    print(f'\n Eccentricity = {e}')
    print(f'\n Semimajor axis = {a} km')
    print('\n Apogee state vector:')
    print(f'\n r = [{ra[0]:17.10E}, {ra[1]:17.10E}, {ra[2]:17.10E}] (km)')
    print(f'\n Radius = {np.linalg.norm(ra)}')
    print(f'\n v = [{va[0]:17.10E}, {va[1]:17.10E}, {va[2]:17.10E}] (km/s)')
    print(f'\n Speed = {np.linalg.norm(va)}')
    print('\n\n' + '-'*70 + '\n\n')