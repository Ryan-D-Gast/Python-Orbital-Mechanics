"""
Author: Ryan Gast
Date: 12/23/2023
Numerical integration of a system of first-order differential
equations by choice of Runge-Kutta methods RK1, RK2, RK3. or RK4.
    Converted From Matlab to Python from: 
    "Orbital Mechanics for Engineering Students" (2013), 
    Appendix D, D.2.
Version: 1.0
Contact: ryan.d.gast@gmail.com
"""


import numpy as np

def rk1_4(ode_function, tspan, y0, h, rk):
    """
    Runge-Kutta method for solving ordinary differential equations (ODEs).

    Parameters:
    - ode_function: The function that defines the ODE system. 
      It should take two arguments: t (time) and y (state vector) and will return the function.
      can return an np.array of differential equations. 
      See the example below for example usage or the test of the function in test_runge_kutta.py.
    - tspan: A list or tuple containing the start and end time of the integration interval.
    - y0: The initial state vector.
    - h: The step size for the integration.
    - rk: The order of the Runge-Kutta method to use. Must be 1, 2, 3, or 4.

    Returns:
    - tout: Numpy array containing the time values at which the solution is computed.
    - yout: Numpy array containing the solution of the ODE system at each time point.

    Raises:
    - ValueError: If the parameter rk is not 1, 2, 3, or 4.
    """
    
    if rk == 1:
        n_stages = 1
        a = [0]
        b = [0]
        c = [1]
    elif rk == 2:
        n_stages = 2
        a = [0, 1]
        b = np.array([[0], [1]])
        c = [1/2, 1/2]
    elif rk == 3:
        n_stages = 3
        a = [0, 1/2, 1]
        b = np.array([[0, 0], [1/2, 0], [-1, 2]])
        c = [1/6, 2/3, 1/6]
    elif rk == 4:
        n_stages = 4
        a = [0, 1/2, 1/2, 1]
        b = np.array([[0, 0, 0], [1/2, 0, 0], [0, 1/2, 0], [0, 0, 1]])
        c = [1/6, 1/3, 1/3, 1/6]
    else:
        raise ValueError('The parameter rk must have the value 1, 2, 3 or 4.')

    t0 = tspan[0]
    tf = tspan[1]
    t = t0
    y = y0
    tout = [t0]  # Initialize tout as a list
    yout = [y0]  # Initialize yout as a list

    f = np.zeros((len(y0), n_stages))

    while t + h <= tf:
        ti = t
        yi = y
        for i in range(n_stages):
            t_inner = ti + a[i]*h
            y_inner = yi
            for j in range(i):
                y_inner = y_inner + h*b[i,j]*f[:,j]
            f[:,i] = ode_function(t_inner, y_inner)
        h = min(h, tf-t)
        t = t + h
        y = yi + h*np.dot(f, c)
        tout.append(t)  # Append t to the list
        yout.append(y)  # Append y to the list

    # Convert tout and yout to numpy arrays after the loop
    tout = np.vstack(tout)
    yout = np.vstack(yout)

    return tout, yout


# Example usage for the ODE y' = 3y + t, y(0) = 1.
if __name__ == '__main__':
    # Define the ODE function
    def ode_function(t, y):
        return 3*y[0] + t  # y' = 3y + t

    # Define the initial conditions and parameters
    tspan = [0, 1]
    y0 = [1]
    h = 0.1
    rk = 2

    # Call the rk1_4 function
    tout, yout = rk1_4(ode_function, tspan, y0, h, rk)

    # Calculate the exact solution
    y_exact = np.exp(3*tout) - tout/3 - 1/9

    # Print the maximum error between the numerical solution and the exact solution
    print('Maximum error:', np.max(np.abs(yout[:,0] - y_exact[:,0])))
    number_of_steps = len(tout)
    print('Number of steps:', number_of_steps)

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