"""
Author: Ryan Gast
Date: 12/31/2023
TThis function calculates the local sidereal time.
    Converted From Matlab to Python from: 
    "Orbital Mechanics for Engineering Students" (2013), 
    Appendix D, D.27.
Version: 1.0
Contact: ryan.d.gast@gmail.com
"""

import math
from J0 import J0

def LST(y, m, d, ut, EL):
    """
    Calculates the Local Sidereal Time (LST) given the date, universal time, and observer's east longitude.

    Args:
        y (int): The year.
        m (int): The month.
        d (int): The day.
        ut (float): The universal time in hours.
        EL (float): The observer's east longitude in degrees.

    Returns:
        float: The Local Sidereal Time (LST) in degrees.
    """

    def zeroTo360(x):
        """
        Returns the value of x within the range [0, 360).

        If x is greater than or equal to 360, it subtracts the largest multiple of 360 from x to bring it within the range.
        If x is less than 0, it subtracts the largest multiple of 360 (minus 1) from x to bring it within the range.

        Args:
            x (float): The input value.

        Returns:
            float: The value of x within the range [0, 360).
        """
        if x >= 360:
            x = x - math.floor(x/360)*360
        elif x < 0:
            x = x - (math.floor(x/360) - 1)*360
        return x

    # Equation 5.48
    j0 = J0(y, m, d)
    # Equation 5.49
    j = (j0 - 2451545)/36525
    # Equation 5.50
    g0 = 100.4606184 + 36000.77004*j + 0.000387933*j**2 - 2.583e-8*j**3
    # Reduce g0 so it lies in the range 0 - 360 degrees
    g0 = zeroTo360(g0)
    # Equation 5.51
    gst = g0 + 360.98564724*ut/24
    # Equation 5.52
    lst = gst + EL
    # Reduce lst to the range 0 - 360 degrees
    lst = lst - 360*math.floor(lst/360)
    return lst

# example usage
if __name__ == "__main__":
    # Data declaration for Example 5.6:
    # East longitude:
    degrees = 139
    minutes = 47
    seconds = 0
    # Date:
    year = 2004
    month = 3
    day = 3
    # Universal time:
    hour = 4
    minute = 30
    second = 0

    # Convert negative (west) longitude to east longitude:
    if degrees < 0:
        degrees = degrees + 360

    # Express the longitudes as decimal numbers:
    EL = degrees + minutes/60 + seconds/3600
    WL = 360 - EL

    # Express universal time as a decimal number:
    ut = hour + minute/60 + second/3600

    # Algorithm 5.3:
    lst = LST(year, month, day, ut, EL)

    # Echo the input data and output the results to the command window:
    print('Example 5.6: Local sidereal time calculation')
    print('Input data:')
    print('Year =', year)
    print('Month =', month)
    print('Day =', day)
    print('UT (hr) =', ut)
    print('West Longitude (deg) =', WL)
    print('East Longitude (deg) =', EL)
    print('\nSolution:')
    print('Local Sidereal Time (deg) =', lst)
    print('Local Sidereal Time (hr) =', lst/15)