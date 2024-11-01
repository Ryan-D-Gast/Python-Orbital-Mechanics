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

def form_sep_curves(ra, dec):
    """
    Forms separate curves based on the given RA and Dec values.

    Parameters:
    ra (list): List of RA values.
    dec (list): List of Dec values.

    Returns:
    RA (dict): Dictionary containing separate curves of RA values.
    Dec (dict): Dictionary containing separate curves of Dec values.
    n_curves (int): Total number of curves formed.
    """
    tol = 100
    curve_no = 1
    n_curves = 1
    k = 0
    ra_prev = ra[0]
    RA = {}
    Dec = {}
    for i in range(len(ra)):
        if abs(ra[i] - ra_prev) > tol:
            curve_no += 1
            n_curves += 1
            k = 0
        k += 1
        if curve_no not in RA:
            RA[curve_no] = []
        if curve_no not in Dec:
            Dec[curve_no] = []
        RA[curve_no].append(ra[i])
        Dec[curve_no].append(dec[i])
        ra_prev = ra[i]
    return RA, Dec, n_curves