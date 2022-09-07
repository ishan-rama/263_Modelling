#########################################################################################
# Function that produces benchmark plots for Model Verification.
#
# 	Functions:
#       plot_benchmark: Generates 3 plots 
#           1. Benchmark - Comparing Analytical and Numerical solutions
#           2. Error Analysis - Relative Error between Analytical and Numberical solutions
#           3. Convergence Analysis - Convergence test wtih varying time steps        
#########################################################################################

#imports
import warnings
import numpy as np
from matplotlib import pyplot as plt
from ode_solver import *


def plot_benchmark(display = False):
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
        plt.savefig('Benchmark', dpi=300)


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
    plt.ylabel("relative error between analytical and numerical solution")
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
