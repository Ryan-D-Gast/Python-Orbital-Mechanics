"""
Author: Ryan Gast
Date: 12/29/2023
Plots the ground track of a satellite orbit.
    Converted From Matlab to Python from: 
    "Orbital Mechanics for Engineering Students" (2013), 
    Appendix D, D.23.
Version: 1.0
Contact: ryan.d.gast@gmail.com
"""

import matplotlib.pyplot as plt

def plot_ground_track(RA, Dec, ra, dec, n_curves):
    """
    Plots the ground track of a satellite.

    Parameters:
    RA (list): List of right ascension values for each curve.
    Dec (list): List of declination values for each curve.
    ra (list): List of right ascension values for start and finish points.
    dec (list): List of declination values for start and finish points.
    n_curves (int): Number of curves to plot.

    Returns:
    None
    """
    plt.figure()
    plt.xlabel('East longitude (degrees)')
    plt.ylabel('Latitude (degrees)')
    plt.axis('equal')
    plt.grid(True)
    for i in range(1, n_curves+1):
        plt.plot(RA[i], Dec[i])
    plt.axis([0, 360, -90, 90])
    plt.text(ra[0], dec[0], 'o Start')
    plt.text(ra[-1], dec[-1], 'o Finish')
    plt.plot([min(ra), max(ra)], [0, 0], color='k')  # the equator
    plt.show()