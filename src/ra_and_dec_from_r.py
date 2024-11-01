"""
Author: Ryan Gast
Date: 12/27/2023
Obtains right ascension and declination from the position
vector.
    Converted From Matlab to Python from: 
    "Orbital Mechanics for Engineering Students" (2013), 
    Appendix D, D.17.
Version: 1.0
Contact: ryan.d.gast@gmail.com
"""

import numpy as np

def ra_and_dec_from_r(r):
    """
    Calculates the right ascension (ra) and declination (dec) from the direction cosines of a vector r.

    Parameters:
    r (list): A list containing the direction cosines of the vector r.

    Returns:
    tuple: A tuple containing the calculated right ascension (ra) and declination (dec) in degrees.
    """
    # direction cosines of r
    l = r[0]/np.linalg.norm(r)
    m = r[1]/np.linalg.norm(r)
    n = r[2]/np.linalg.norm(r)
    dec = np.degrees(np.arcsin(n))
    if m > 0:
        ra = np.degrees(np.arccos(l/np.cos(np.radians(dec))))
    else:
        ra = 360 - np.degrees(np.arccos(l/np.cos(np.radians(dec))))
    return ra, dec

# example usage
if __name__ == "__main__":
    r = np.array([-5368, -1784, 3691])
    ra, dec = ra_and_dec_from_r(r)
    print(ra, dec)