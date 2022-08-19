# ENGSCI263: Benchmarking


#imports
import numpy as np
import warnings
from matplotlib import pyplot as plt
from modelG15 import pressure_ode

def step_rk4(f, tk, yk, h, args=None):
    """
    Perform one step of the Classic RK4 method

    Parameters
    ----------
    f : callable
        derivative function
    tk : float
        initial value of independent variable
    yk : float
        initial value of solution
    h : float
        step size
    args : iterable
        Optional parameters to pass into derivative function.

    Notes
    -------

    *Enter notes*

    Returns
    -------
    integer, float
            y value over the step h
    """
    if args is None:
        args = []

    f0 = f(tk, yk, *args)
    f1 = f(tk + h/2, yk + (h*f0)/2, *args)
    f2 = f(tk + h/2, yk + (h*f1)/2, *args)
    f3 = f(tk + h, yk + h*f2, *args)

    return yk + h*((f0+2*f1+2*f2+f3)/6)

def solve_ode(f, t0, t1, dt, p0, pars):
    ''' Solve an ODE numerically.

        Parameters:
        -----------
        f : callable
            Function that returns dxdt given variable and parameter inputs.
        q : float
            forcing term
        t0 : float
            Initial time of solution.
        t1 : float
            Final time of solution.
        dt : float
            Time step length.
        x0 : float
            Initial value of solution.
        pars : array-like
            List of parameters passed to ODE function f.

        Returns:
        --------
        t : array-like
            Independent variable solution vector.
        x : array-like
            Dependent variable solution vector.

        Notes:
        ------
        ODE will be solved using RK4 method.

        Assume that the pressure_ode takes the following inputs, in order:
            1. independent variable
            2. dependent variable
            3. forcing term, q
            4. all other parameters
    '''
    #Return arrays
    pressure_values = [p0]
    t_range = np.arange(t0, t1+dt, dt)

    for index, tk in enumerate(t_range[:-1]):
        pressure_values.append(
            step_rk4(f, tk, pressure_values[index], dt, pars))

    return t_range, np.array(pressure_values)

def plot_benchmark():
    ''' Compare analytical and numerical solutions.

        Parameters:
        -----------
        none

        Returns:
        --------
        none

        Notes:
        ------
        This function called within if __name__ == "__main__":

        It should contain commands to obtain analytical and numerical solutions,
        plot these, and either display the plot to the screen or save it to the disk.
        
    '''
    #Parameters set
    q, a, b, p0, c, dqdt = 1, 1, 1, 0, 0, 0
    pars = [q, a, b, p0, c, dqdt]

    #Analytical Solution
    # X = e^(-t) + 1

    ###################
    #PLOT 1 - Benchmark
    ###################

    f1, ax1 = plt.subplots(nrows=1, ncols=1)

    #Analytical Solution
    t_range = np.arange(0, 10.1, 0.1)
    analytic_x = np.e**(-1*t_range) - 1
    plt.plot(t_range, analytic_x, "-", color="red",
             label="analytical solution")

    #Numerical Solution
    t_range, numeric_x = solve_ode(pressure_ode, 0, 10, 0.1, 0, pars)
    plt.plot(t_range, numeric_x, "x", color="blue", label="numerical solution")

    # Naming axes and title and legends
    plt.xlabel("t")
    plt.ylabel("X")
    plt.title(
        "Benchmark comparison between analytical solution and numerical solution")
    plt.legend(loc=1, prop={'size': 10})

    save_figure = 1
    if not save_figure:
        plt.show()
    else:
        plt.savefig('Benchmark', dpi=300)

    ########################
    #PLOT 2 - Error Analysis
    ########################

    f2, ax2 = plt.subplots(nrows=1, ncols=1)

    warnings.simplefilter('ignore')

    relative_error = []
    for i in range(len(t_range)):
        relative_error.append(
            (abs(analytic_x[i] - numeric_x[i])/numeric_x[i]))

    plt.plot(t_range, relative_error)

    # Naming axes and title and legends
    plt.xlabel("t")
    plt.ylabel("relative error against benchmark")
    plt.title(
        "Error Analysis")

    save_figure = 1
    if not save_figure:
        plt.show()
    else:
        plt.savefig('Error Analysis', dpi=300)

    #############################
    #PLOT 3 - Convergence Testing
    #############################

    f3, ax3 = plt.subplots(nrows=1, ncols=1)

    x_at_ten = []

    step_size = np.arange(0.1, 2.1, 0.1)
    inverse_step_size = 1/step_size

    for step in step_size:
        t_range, numeric_x = solve_ode(
            pressure_ode, 0, 10, step, 0, pars)
        x_at_ten.append(numeric_x[-1])  # Append last value (t=10)

    plt.plot(inverse_step_size, x_at_ten, "x")

    # Naming axes and title and legends
    plt.xlabel("1/step")
    plt.ylabel("x(t=10)")
    plt.title(
        "timestep convergence")

    save_figure = 1
    if not save_figure:
        plt.show()
    else:
        plt.savefig('timestep convergence', dpi=300)


if __name__ == "__main__":
    plot_benchmark()
