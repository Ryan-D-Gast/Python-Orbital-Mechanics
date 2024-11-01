"""
Author: Ryan Gast
Date: 12/31/2023
This function computes the Julian day number at 0 UT 
for any year between 1900 and 2100 using Equation 5.48.
    Converted From Matlab to Python from: 
    "Orbital Mechanics for Engineering Students" (2013), 
    Appendix D, D.26.
Version: 1.0
Contact: ryan.d.gast@gmail.com
"""

import math

def J0(year, month, day):
    """
    Calculate the Julian Day Number (J0) for a given date.

    Parameters:
    year (int): The year of the date. Must be between 1901 and 2099.
    month (int): The month of the date. Must be between 1 and 12.
    day (int): The day of the date. Must be between 1 and 31.

    Returns:
    float: The Julian Day Number (J0) for the given date.
    """
    j0 = 367*year - math.floor(7*(year + math.floor((month + 9)/12))/4) \
    + math.floor(275*month/9) + day + 1721013.5
    return j0

# example usage
if __name__ == "__main__":
    year = 2004
    month = 5
    day = 12
    hour = 14
    minute = 45
    second = 30

    # Calculate UT
    ut = hour + minute/60 + second/3600

    # Calculate j0 and jd
    j0 = J0(year, month, day)
    jd = j0 + ut/24

    # Print the results
    print('Example 5.4: Julian day calculation')
    print('Input data:')
    print('Year =', year)
    print('Month =', month)
    print('Day =', day)
    print('Hour =', hour)
    print('Minute =', minute)
    print('Second =', second)
    print(jd)
    print('Julian day number = {:.3f}'.format(jd))