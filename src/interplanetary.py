"""
Author: Ryan Gast
Date: 1/9/2023
Determines the spacecraft trajectory from the sphere
of influence of planet 1 to that of planet 2 
    Converted From Matlab to Python from: 
    "Orbital Mechanics for Engineering Students" (2013), 
    Appendix D, D.36.
Version: 1.0
Contact: ryan.d.gast@gmail.com
"""

import numpy as np
from lambert import lambert
from planet_elements_and_sv import planet_elements_and_sv

def interplanetary(depart, arrive):
    """
    Determines the spacecraft trajectory from the sphere of influence of planet 1 to that of planet 2.

    Parameters:
    depart (tuple): A tuple containing the planet ID, year, month, day, hour, minute, and second of departure.
    arrive (tuple): A tuple containing the planet ID, year, month, day, hour, minute, and second of arrival.

    Returns:
    tuple: A tuple containing the planet 1 state vector, planet 2 state vector, and spacecraft trajectory.
    """
    
    planet_id, year, month, day, hour, minute, second = depart
    _, Rp1, Vp1, jd1 = planet_elements_and_sv(planet_id, year, month, day, hour, minute, second)

    planet_id, year, month, day, hour, minute, second = arrive
    _, Rp2, Vp2, jd2 = planet_elements_and_sv(planet_id, year, month, day, hour, minute, second)

    tof = (jd2 - jd1) * 24 * 3600
    R1 = Rp1
    R2 = Rp2

    V1, V2 = lambert(R1, R2, tof, 'pro', 1.327124e11)
 
    planet1 = np.concatenate((Rp1, Vp1, [jd1]))
    planet2 = np.concatenate((Rp2, Vp2, [jd2]))
    trajectory = np.concatenate((V1, V2))

    return planet1, planet2, trajectory

# example usage
if __name__ == "__main__":
    from coe_from_sv import coe_from_sv
    # Constants
    mu = 1.327124e11
    deg = np.pi / 180

    # Departure
    depart = [3, 1996, 11, 7, 0, 0, 0]

    # Arrival
    arrive = [4, 1997, 9, 12, 0, 0, 0]

    # Calculate trajectory
    planet1, planet2, trajectory = interplanetary(depart, arrive)
    R1 = planet1[:3]
    Vp1 = planet1[3:6]
    jd1 = planet1[6]
    R2 = planet2[:3]
    Vp2 = planet2[3:6]
    jd2 = planet2[6]
    V1 = trajectory[:3]
    V2 = trajectory[3:]
    tof = jd2 - jd1

    # Calculate orbital elements
    coe = coe_from_sv(R1, V1, mu)
    coe2 = coe_from_sv(R2, V2, mu)

    # Calculate v-infinity
    vinf1 = V1 - Vp1
    vinf2 = V2 - Vp2

    # Print results
    print(f"Departure:\nPlanet: {depart[0]}\nYear: {depart[1]}\nMonth: {depart[2]}\nDay: {depart[3]}\nHour: {depart[4]}\nMinute: {depart[5]}\nSecond: {depart[6]}\nJulian day: {jd1}")
    print(f"Planet position vector (km) = {R1}\nMagnitude = {np.linalg.norm(R1)}")
    print(f"Planet velocity (km/s) = {Vp1}\nMagnitude = {np.linalg.norm(Vp1)}")
    print(f"Spacecraft velocity (km/s) = {V1}\nMagnitude = {np.linalg.norm(V1)}")
    print(f"v-infinity at departure (km/s) = {vinf1}\nMagnitude = {np.linalg.norm(vinf1)}")
    print(f"Time of flight = {tof} days")
    print(f"Arrival:\nPlanet: {arrive[0]}\nYear: {arrive[1]}\nMonth: {arrive[2]}\nDay: {arrive[3]}\nHour: {arrive[4]}\nMinute: {arrive[5]}\nSecond: {arrive[6]}\nJulian day: {jd2}")
    print(f"Planet position vector (km) = {R2}\nMagnitude = {np.linalg.norm(R2)}")
    print(f"Planet velocity (km/s) = {Vp2}\nMagnitude = {np.linalg.norm(Vp2)}")
    print(f"Spacecraft Velocity (km/s) = {V2}\nMagnitude = {np.linalg.norm(V2)}")
    print(f"v-infinity at arrival (km/s) = {vinf2}\nMagnitude = {np.linalg.norm(vinf2)}")
    print(f"Orbital elements of flight trajectory:\nAngular momentum (km^2/s) = {coe[0]}\nEccentricity = {coe[1]}\nRight ascension of the ascending node (deg) = {coe[2]/deg}\nInclination to the ecliptic (deg) = {coe[3]/deg}\nArgument of perihelion (deg) = {coe[4]/deg}\nTrue anomaly at departure (deg) = {coe[5]/deg}\nTrue anomaly at arrival (deg) = {coe2[5]/deg}\nSemimajor axis (km) = {coe[6]}")
    if coe[1] < 1:
        print(f"Period (days) = {2*np.pi/np.sqrt(mu)*coe[6]**1.5/24/3600}")
