"""
Author: Ryan Gast
Date: 1/14/2024
Script to compute the trajectory of a spacecraft from Earth to the Moon
    Converted From Matlab to Python from: 
    "Orbital Mechanics for Engineering Students" (2013), 
    Appendix D, D.38.
Version: 1.0
Contact: ryan.d.gast@gmail.com
"""

import numpy as np
from scipy.integrate import solve_ivp
from datetime import datetime
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ra_and_dec_from_r import ra_and_dec_from_r
from simpsons_lunar_ephermis import simpsons_lunar_ephemeris
from astropy.time import Time

def rates(t, y):
    jd = jd0 - (ttt - t) / days
    X, Y, Z, vX, vY, vZ = y
    r_ = np.array([X, Y, Z])
    r = np.linalg.norm(r_)
    rm_, _ = simpsons_lunar_ephemeris(jd)
    rm = np.linalg.norm(rm_)
    rms_ = rm_ - r_
    rms = np.linalg.norm(rms_)
    aearth_ = -mu_e * r_ / r**3
    amoon_ = mu_m * (rms_ / rms**3 - rm_ / rm**3)
    a_ = aearth_ + amoon_
    aX, aY, aZ = a_
    dydt = [vX, vY, vZ, aX, aY, aZ]
    return dydt

def plotit_XYZ(X, Y, Z, Xm, Ym, Zm, imin):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Geocentric inertial coordinate axes:
    L = 20*Re
    ax.quiver(0, 0, 0, L, 0, 0, color='k')
    ax.quiver(0, 0, 0, 0, L, 0, color='k')
    ax.quiver(0, 0, 0, 0, 0, L, color='k')

    # Earth:
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = Re * np.outer(np.cos(u), np.sin(v))
    y = Re * np.outer(np.sin(u), np.sin(v))
    z = Re * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='b', alpha=0.5)

    # Spacecraft at TLI, closest approach, and tf
    ax.scatter([X[0], X[imin], X[-1]], [Y[0], Y[imin], Y[-1]], [Z[0], Z[imin], Z[-1]], c='k')

    # Moon at TLI, closest approach, and end of simulation
    for i in [0, imin, -1]:
        x = Rm * np.outer(np.cos(u), np.sin(v)) + Xm[i]
        y = Rm * np.outer(np.sin(u), np.sin(v)) + Ym[i]
        z = Rm * np.outer(np.ones(np.size(u)), np.cos(v)) + Zm[i]
        ax.plot_surface(x, y, z, color='g', alpha=0.99)

    # Spacecraft trajectory
    ax.plot(X, Y, Z, 'r', linewidth=1.5)

    # Moon trajectory
    ax.plot(Xm, Ym, Zm, 'g', linewidth=0.5)

    ax.set_box_aspect([1,1,1])  # equal scaling
    plt.axis('off')
    ax.view_init(elev=45, azim=45)

    plt.show()
    
#...general data
deg = np.pi/180
days = 24*3600
Re = 6378
Rm = 1737
m_e = 5974.e21
m_m = 73.48e21
mu_e = 398600.4
mu_m = 4902.8
D = 384400
RS = D*(m_m/m_e)**(2/5)

#...Data declaration for Example 9.03
Title = 'Example 9.3 4e'
# Date and time of lunar arrival:
year = 2020
month = 5
day = 4
hour = 12
minute = 0
second = 0
t0 = 0
z0 = 320
alpha0 = 90
dec0 = 15
gamma0 = 40
fac = .9924  # Fraction of Vesc
ttt = 3*days
tf = ttt + 2.667*days

#...State vector of moon at target date:
# Initialize jd0 as a datetime object
jd0 = Time(datetime(year, month, day, hour, minute, second), format='datetime').jd

rm0_, vm0_ = simpsons_lunar_ephemeris(jd0)  # You need to define this function
RA, Dec = ra_and_dec_from_r(rm0_)  # You need to define this function

distance = np.linalg.norm(rm0_)
hmoon_ = np.cross(rm0_, vm0_)
hmoon = np.linalg.norm(hmoon_)
inclmoon = np.arccos(hmoon_[2]/hmoon) * 180 / np.pi

#...Initial position vector of probe:
I_ = np.array([1, 0, 0])
J_ = np.array([0, 1, 0])
K_ = np.cross(I_, J_)
r0 = Re + z0
r0_ = r0*(np.cos(np.deg2rad(alpha0))*np.cos(np.deg2rad(dec0))*I_ +
           np.sin(np.deg2rad(alpha0))*np.cos(np.deg2rad(dec0))*J_ +
           np.sin(np.deg2rad(dec0))*K_)
vesc = np.sqrt(2*mu_e/r0)
v0 = fac*vesc
w0_ = np.cross(r0_, rm0_)/np.linalg.norm(np.cross(r0_, rm0_))

#...Initial velocity vector of probe:
ur_ = r0_/np.linalg.norm(r0_)
uperp_ = np.cross(w0_, ur_)/np.linalg.norm(np.cross(w0_, ur_))
vr = v0*np.sin(np.deg2rad(gamma0))
vperp = v0*np.cos(np.deg2rad(gamma0))
v0_ = vr*ur_ + vperp*uperp_
uv0_ = v0_/v0

#...Initial state vector of the probe:
y0 = np.array([r0_[0], r0_[1], r0_[2], v0_[0], v0_[1], v0_[2]])

#...Pass the initial conditions and time interval to solve_ivp, which
# calculates the position and velocity of the spacecraft at discrete
# times t, returning the solution in the column vector y. solve_ivp uses
# the subfunction 'rates' below to evaluate the spacecraft acceleration
# at each integration time step.
sol = solve_ivp(rates, [t0, tf], y0, rtol=1.e-10, atol=1.e-10)  # You need to define the rates function

#...Spacecraft trajectory
# in ECI frame:
X = sol.y[0, :]
Y = sol.y[1, :]
Z = sol.y[2, :]
vX = sol.y[3, :]
vY = sol.y[4, :]
vZ = sol.y[5, :]

# in Moon-fixed frame:
x = []
y = []
z = []

#...Moon trajectory
# in ECI frame:
Xm = []
Ym = []
Zm = []
vXm = []
vYm = []
vZm = []
xm = []
ym = []
zm = []
dist_min = 1.e30

for i in range(len(sol.t)):
    ti = sol.t[i]
    r_ = np.array([X[i], Y[i], Z[i]]).T
    jd = jd0 - (ttt - ti)/days
    rm_, vm_ = simpsons_lunar_ephemeris(jd)
    Xm.append(rm_[0]); Ym.append(rm_[1]); Zm.append(rm_[2])
    vXm.append(vm_[0]); vYm.append(vm_[1]); vZm.append(vm_[2])
    x_ = rm_
    z_ = np.cross(x_, vm_)
    y_ = np.cross(z_, x_)
    i_ = x_/np.linalg.norm(x_)
    j_ = y_/np.linalg.norm(y_)
    k_ = z_/np.linalg.norm(z_)
    Q = np.array([i_, j_, k_])
    rx_ = Q @ r_
    x.append(rx_[0]); y.append(rx_[1]); z.append(rx_[2])
    rmx_ = Q @ rm_
    xm.append(rmx_[0]); ym.append(rmx_[1]); zm.append(rmx_[2])
    dist_ = r_ - rm_
    dist = np.linalg.norm(dist_)
    if dist < dist_min:
        imin = i
        dist_min = dist

rmTLI_ = np.array([Xm[0], Ym[0], Zm[0]])
RATLI, DecTLI = ra_and_dec_from_r(rmTLI_)
v_atdmin_ = np.array([vX[imin], vY[imin], vZ[imin]])
rm_perilune_ = np.array([Xm[imin], Ym[imin], Zm[imin]]).T
vm_perilune_ = np.array([vXm[imin], vYm[imin], vZm[imin]]).T
RA_at_perilune, Dec_at_perilune = ra_and_dec_from_r(rm_perilune_)
target_error = np.linalg.norm(rm_perilune_ - rm0_)
rel_speed = np.linalg.norm(v_atdmin_ - vm_perilune_)
rend_ = np.array([X[-1], Y[-1], Z[-1]])
alt_end = np.linalg.norm(rend_) - Re
ra_end, dec_end = ra_and_dec_from_r(rend_)

time = np.zeros(imin)  # Define the "time" variable as an array of zeros
rms = np.zeros(imin)  # Define the "rms" variable as an array of zeros
incl = np.zeros(imin)  # Define the "incl" variable as an array of zeros
for i in range(imin):
    time[i] = sol.t[i]
    r_ = np.array([X[i], Y[i], Z[i]]).T
    r = np.linalg.norm(r_)
    v_ = np.array([vX[i], vY[i], vZ[i]]).T
    rm_ = np.array([Xm[i], Ym[i], Zm[i]]).T
    rm = np.linalg.norm(rm_)
    rms_ = rm_ - r_
    rms[i] = np.linalg.norm(rms_)
    aearth_ = -mu_e*r_/r**3
    amoon_ = mu_m*(rms_/rms[i]**3 - rm_/rm**3)
    atot_ = aearth_ + amoon_
    binormal_ = np.cross(v_, atot_)/np.linalg.norm(np.cross(v_, atot_))
    binormalz = binormal_[2]
    incl[i] = np.arccos(binormalz) * 180 / np.pi
    
print(f'\n\n{Title}\n\n')
print(f'Date and time of arrival at moon: {month}/{day}/{year} {hour}:{minute}:{second}')
print(f'\nMoon\'s position: ')
print(f'\n Distance = {distance} km')
print(f'\n Right Ascension = {RA} deg')
print(f'\n Declination = {Dec} deg')
print(f'\nMoon\'s orbital inclination = {inclmoon} deg\n')
print(f'\nThe probe at earth departure (t = {t0} sec):')
print(f'\n Altitude = {z0} km')
print(f'\n Right ascension = {alpha0} deg')
print(f'\n Declination = {dec0} deg')
print(f'\n Flight path angle = {gamma0} deg')
print(f'\n Speed = {v0} km/s')
print(f'\n Escape speed = {vesc} km/s')
print(f'\n v/vesc = {v0/vesc}')
print(f'\n Inclination of translunar orbit = {np.arccos(w0_[2]) * 180 / np.pi} deg\n')
print(f'\nThe moon when the probe is at TLI:')
print(f'\n Distance = {np.linalg.norm(rmTLI_)} km')
print(f'\n Right ascension = {RATLI} deg')
print(f'\n Declination = {DecTLI} deg')
print(f'\nThe moon when the probe is at perilune: ')
print(f'\n Distance = {np.linalg.norm(rm_perilune_)} km')
print(f'\n Speed = {np.linalg.norm(vm_perilune_)} km/s')
print(f'\n Right ascension = {RA_at_perilune} deg')
print(f'\n Declination = {Dec_at_perilune} deg')
print(f'\n Target error = {target_error} km')
print(f'\n\nThe probe at perilune:')
print(f'\n Altitude = {dist_min - Rm} km')
print(f'\n Speed = {np.linalg.norm(v_atdmin_)} km/s')
print(f'\n Relative speed = {rel_speed} km/s')
print(f'\n Inclination of osculating plane = {incl[imin-1]} deg')
print(f'\n Time from TLI to perilune = {abs(sol.t[imin])/3600} hours ({abs(sol.t[imin])/3600/24} days)')
print(f'\n\nTotal time of flight = {sol.t[-1]/days} days')
print(f'\nTime to target point = {ttt/days} days')
print(f'\nFinal earth altitude = {alt_end} km')
print(f'\nFinal right ascension = {ra_end} deg')
print(f'\nFinal declination = {dec_end} deg\n')

# Plot the trajectory relative to the inertial frame:
plotit_XYZ(X,Y,Z,Xm,Ym,Zm,imin)

# Plot inclination of the osculating plane vs distance from the Moon
plt.figure()
plt.plot(rms/RS, incl)
plt.axhline(y=90, color='r', linestyle='--')
plt.title('Osculating Plane Inclination vs Distance from Moon')
plt.xlabel('r_{ms}/R_s')
plt.ylabel('Inclination (deg)')
plt.grid(True)
plt.minorticks_on()

# Plot the trajectory relative to the rotating Moon-fixed frame:
plotit_XYZ(x,y,z,xm,ym,zm,imin)


    
