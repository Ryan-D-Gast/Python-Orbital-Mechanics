"""
Author: Ryan Gast
Date: 12/29/2023
Finds the root of a function using the bisection method.
    Converted From Matlab to Python from: 
    "Orbital Mechanics for Engineering Students" (2013), 
    Appendix D, D.9.
    Note: was called "bisect" in Matlab. but due to the name conflict 
    with the scipy library it was renamed to "my_bisect".
Version: 1.0
Contact: ryan.d.gast@gmail.com
"""

import numpy as np

def bisect(fun, xl, xu):
    """
    This function evaluates a root of a function using
    the bisection method.
    tol - error to within which the root is computed
    n - number of iterations
    xl - low end of the interval containing the root
    xu - upper end of the interval containing the root
    i - loop index
    xm - mid-point of the interval from xl to xu
    fun - name of the function whose root is being found
    fxl - value of fun at xl
    fxm - value of fun at xm
    root - the computed root
    User functions required: none
    """
    tol = 1.e-6
    n = int(np.ceil(np.log(abs(xu - xl)/tol)/np.log(2)))
    for i in range(n):
        xm = (xl + xu)/2
        fxl = fun(xl)
        fxm = fun(xm)
        if fxl*fxm > 0:
            xl = xm
        else:
            xu = xm
    root = xm
    return root


