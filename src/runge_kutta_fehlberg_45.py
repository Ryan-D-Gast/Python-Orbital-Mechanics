"""
Author: Ryan Gast
Date: 12/23/2023
Numerical integration of a system of first-order differential
equations by the Runge-Kutta-Fehlberg 4(5) method with adaptive step size control.
    Converted From Matlab to Python from: 
    "Orbital Mechanics for Engineering Students" (2013), 
    Appendix D, D.4.
Version: 1.0
Contact: ryan.d.gast@gmail.com
"""

import numpy as np

def rkf45(ode_function, tspan, y0, tolerance=1.e-8):
    """
    Runge-Kutta-Fehlberg 4(5) method for solving ordinary differential equations (ODEs).

    Parameters:
    - ode_function: The function that defines the ODE system. It should take two arguments: t (time) and y (state vector).
      can return an np.array of differential equations. 
      See the example below for example usage or the test of the function in test_runge_kutta.py.
    - tspan: A list or tuple containing the initial and final time values.
    - y0: The initial state vector.
    - tolerance: The desired tolerance for the solution, does not require a input will default to 1.e-8.

    Returns:
    - tout: Array of time values at which the solution is computed.
    - yout: Array of state vectors corresponding to the time values in `tout`.
    """
    import numpy as np

    a = np.array([0, 1/4, 3/8, 12/13, 1, 1/2])
    b = np.array([
        [0, 0, 0, 0, 0],
        [1/4, 0, 0, 0, 0],
        [3/32, 9/32, 0, 0, 0],
        [1932/2197, -7200/2197, 7296/2197, 0, 0],
        [439/216, -8, 3680/513, -845/4104, 0],
        [-8/27, 2, -3544/2565, 1859/4104, -11/40]
    ])
    c4 = np.array([25/216, 0, 1408/2565, 2197/4104, -1/5, 0])
    c5 = np.array([16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55])

    t0 = tspan[0]
    tf = tspan[1]
    t = t0
    y = y0
    tout = np.array([t])
    yout = np.array([y])

    h = (tf - t0) / 100  # Assumed initial time step.

    while t < tf:
        hmin = 16 * np.finfo(float).eps * t
        ti = t
        yi = y
        f = np.zeros((len(y), 6))

        for i in range(6):
            t_inner = ti + a[i] * h
            y_inner = yi
            for j in range(i):
                y_inner = y_inner + h * b[i, j] * f[:, j]
            f[:, i] = ode_function(t_inner, y_inner)

        te = h * np.dot(f, (c4 - c5))  # Difference between 4th and 5th order solutions
        te_max = np.max(np.abs(te))
        ymax = np.max(np.abs(y))
        te_allowed = tolerance * np.max([ymax, 1.0])
        delta = (te_allowed / (te_max + np.finfo(float).eps)) ** (1/5)

        if te_max <= te_allowed:
            h = min(h, tf - t)
            t = t + h
            y = yi + h * np.dot(f, c5)
            tout = np.append(tout, t)
            yout = np.vstack([yout, y])

        h = min(delta * h, 4 * h)
        if h < hmin:
            print(f"\n\n Warning: Step size fell below its minimum allowable value ({hmin}) at time {t}.\n\n")
            return tout, yout
        
    return tout, yout

# example usage for the ODE y' = 3y + t, y(0) = 1.
if __name__ == '__main__':
    # Define the ODE function
    def ode_function(t, y):
        return 3*y[0] + t  # y' = 3y + t

    # Define the initial conditions and parameters
    tspan = [0, 1]
    y0 = [1]
    
    # Call the rk1_4 function
    tout, yout = rkf45(ode_function, tspan, y0)

    # Calculate the exact solution
    y_exact = np.exp(3*tout) - tout/3 - 1/9

    # Print the maximum error between the numerical solution and the exact solution
    print('Maximum error:', np.max(np.abs(yout[:,0] - y_exact[:,0])))
    number_of_steps = len(tout)
    print('Number of steps:', number_of_steps)
    print(yout)

    # Plotting
    import matplotlib.pyplot as plt

    # Plot the numerical solution
    plt.plot(tout, yout[:,0], label='Numerical solution')

    # Plot the exact solution
    plt.plot(tout, y_exact, label='Exact solution')

    # Add a legend
    plt.legend()

    # Add labels and a title
    plt.xlabel('t')
    plt.ylabel('y')
    plt.title('Comparison of numerical and exact solutions')

    # Show the plot
    plt.show()