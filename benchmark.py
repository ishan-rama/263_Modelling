#########################################################################################
# benchmark.py - Function that produces benchmark plots for Model Verification.
#
# 	Functions:
#       solve_pressure_ode_BENCHMARK: Solves pressure_ode numerically with preset parameters for benchmarking.
#       plot_benchmark_pressure: Generates 3 plots 
#           1. Benchmark - Comparing Analytical and Numerical solutions
#           2. Error Analysis - Relative Error between Analytical and Numberical solutions
#           3. Convergence Analysis - Convergence test wtih varying time steps
#       solve_subsidence_model_BENCHMARK: Solves subsidence_model numerically with preset parameters for benchmarking.  
#       plot_benchmark_subsidence: Benchmark of subsidence_model.      
#########################################################################################

#library imports
import warnings
import numpy as np
from matplotlib import pyplot as plt

#file imports
from model_solver import *

#To display or not to display
display = True


def solve_pressure_ode_BENCHMARK(f, t0, t1, dt, p0, pars):
    ''' Solves pressure_ode numerically with preset parameters for benchmarking.

        Parameters:
        -----------
        f : callable
            Function that returns dp/dt (pressure_ode)
        t0 : float
            Initial time of solution
        t1 : float
            Final time of solution
        dt : float
            Time step
        p0 : float
            Ambient Pressure outside the reservoir
        pars : array-like
            Strength parameters - [a, b, c]

        Returns:
        --------
        t_range : array-like
            Independent time variable vector.
        pressure_values : array-like
            Dependent variable solution vector.

        Notes:
        ------
        ODE will be solved using RK4 method.
        Assume that the pressure_ode takes the following inputs, in order:
            1. dependent variable
            2. parameter list [q, p0, pars:(a, b, c), dqdt= 0, dqdt= 0]
    '''  
    #Create solution arrays
    pressure_values = [p0]
    t_range = np.arange(t0, t1 + dt, dt)

    #Load in mass extraction values interpolated at t_range times
    #q = interpolate_mass_extraction(t_range) #Not needed for benchmarking
    q = 1

    #Find the derivative at each point numerically
    #dqdt = np.gradient(q, dt) #Not needed as q is a constant 
    dqdt = 0

    for index, tk in enumerate(t_range[:-1]):
        # Joining parameters into one list
        args = [q, p0] + pars + [dqdt, dqdt]
        pressure_values.append(step_rk4(f, pressure_values[index], dt, args))

    return t_range, np.array(pressure_values)


def plot_benchmark_pressure(display = False):
    ''' Compare analytical and numerical solutions.

        Parameters:
        -----------
        display : boolean, default False
            switch to either display(TRUE) or save plots(FALSE)

        Returns:
        --------
        Generates 3 plots
            1. Benchmark - Comparing Analytical and Numerical solutions
            2. Error Analysis - Relative Error between Analytical and Numberical solutions
            3. Convergence Analysis - Convergence test wtih varying time steps 

        Note:
        ------
        For Benchmarking we need to choose a constant q, therefore we cannot call 
        solve_pressure_ode but instead call a simpler refactored version of it with
        simpler parameters and steps.
    '''
    #Preset parameters
    a, b, c, p0 = 1, 1, 0, 0
    pars = [a, b, c]
    t0, t1, dt = 0, 10, 0.1

    #Analytical Solution based of parameters
    # X = e^(-t) - 1

    ###################
    #PLOT 1 - Benchmark
    ###################

    f1, ax1 = plt.subplots(nrows=1, ncols=1)

    #Analytical Solution
    t_range = np.arange(t0, t1 + dt, dt)
    analytic_x = np.e**(-1*t_range) - 1
    plt.plot(t_range, analytic_x, "-", color="red",
             label="analytical solution")

    #Numerical Solution
    t_range, numeric_x = solve_pressure_ode_BENCHMARK(pressure_ode, t0, t1, dt, p0, pars)
    plt.plot(t_range, numeric_x, "x", color="blue", label="numerical solution")

    # Naming axes and title and legends
    plt.xlabel("t")
    plt.ylabel("X")
    plt.title("Benchmark comparison between analytical solution and numerical solution")
    plt.legend(loc=1, prop={'size': 10})

    if display:
        plt.show()
    else:
        plt.savefig('Benchmark_pressure_ode', dpi=300)


    ########################
    #PLOT 2 - Error Analysis
    ########################

    f2, ax2 = plt.subplots(nrows=1, ncols=1)

    warnings.simplefilter('ignore') #Ingore division close to 0 warning

    relative_error = []
    for i in range(len(t_range)):
        relative_error.append(
            (abs(analytic_x[i] - numeric_x[i]/numeric_x[i])))

    plt.plot(t_range, relative_error)

    # Naming axes and title and legends
    plt.xlabel("t")
    plt.ylabel("relative error between analytical and numerical")
    plt.title(
        "Error Analysis")

    if display:
        plt.show()
    else:
        plt.savefig('Error_Analysis', dpi=300)


    #############################
    #PLOT 3 - Convergence Testing
    #############################

    f3, ax3 = plt.subplots(nrows=1, ncols=1)

    x_at_ten = []

    step_size = np.arange(0.1, 1.8, 0.1)
    inverse_step_size = 1/step_size

    for step in step_size:
        t_range, numeric_x = solve_pressure_ode_BENCHMARK(
            pressure_ode, t0, t1, step, p0, pars)
        x_at_ten.append(numeric_x[-1])  # Append last value (t=10)

    plt.plot(inverse_step_size, x_at_ten, "x")

    # Naming axes and title and legends
    plt.xlabel("1/step")
    plt.ylabel("x(t=10)")
    plt.title(
        "timestep convergence")
    plt.axhline(y = -0.9999546, color= "red", label = "Analytical Value")
    plt.legend()

    if display:
        plt.show()
    else:
        plt.savefig('timestep_convergence', dpi=300)


def solve_subsidence_model_BENCHMARK():
    """Solves subsidence_model numerically with preset parameters for benchmarking.

    Parameters
    ----------
    f : callable
        calls subsidence_model()
    t0 : float
        start time of time domain
    t1 : float
        end time of time domain
    dt : float
        time step
    p : array-like
        pressure values over time range t0-t1
    p0 : float
        Ambient Pressure outside the reservoir
    pars : array-like
        parameters - [d, Tm, Td]

    Returns
    -------
    t_range : array-like
        Independent time variable vector.
    s_values : array-like
        Subsidence solution vector.
    """
    s_values = []
    t_range = np.arange(0, 51, 1) 

    d= 1
    p0 = 50
    p = np.arange(0, 51, 1)
    pars= [0, 0, 0]

    for index in range(len(t_range)):
        p_change = p0 - p[index]
        s = subsidence_model(t_range[index], p_change, *pars)
        s_values.append(s)

    return t_range, s_values


def plot_benchmark_subsidence(display= False):
    ''' Benchmark of subsidence_model

        Parameters:
        -----------
        display : boolean, default False
            switch to either display(TRUE) or save plots(FALSE)

        Returns:
        --------
        Plot:
            1. Benchmark 
    '''

    #Analytical Solution based of parameters
    # X = -t + 50

    #################
    #PLOT - Benchmark
    #################

    f1, ax1 = plt.subplots(nrows=1, ncols=1)

    #Analytical Solution
    t_range = np.arange(0, 51, 1)
    analytic_x = np.arange(50, -1, -1)
    plt.plot(t_range, analytic_x, "-", color="red",
             label="analytical solution")

    #Numerical Solution
    t_range, numeric_x = solve_subsidence_model_BENCHMARK()
    plt.plot(t_range, analytic_x, "x", color="blue",
             label="numerical solution")

    # Naming axes and title and legends
    plt.xlabel("t")
    plt.ylabel("X")
    plt.title(
        "Benchmark of subsidence_model")
    plt.legend(loc=1, prop={'size': 10})

    if display:
        plt.show()
    else:
        plt.savefig('Benchmark_subsidence_model', dpi=300)


if __name__ == "__main__":
    plot_benchmark_pressure(display)
    plot_benchmark_subsidence(display)
