#########################################################################################
# Function library for models, ode solvers and loading in data.
#
# 	Functions:
#       interpolate_mass_extraction: Loads in and interpolates mass extraction data at t_interp times using cubic splines.
#		pressure_ode: Returns the derivative dp/dt for given parameters.
#		tep_rk4: Performs one step of the Classic RK4 method
#		solve_pressure_ode: Solves pressure_ode numerically using above functions.
#########################################################################################

#imports
from scipy import interpolate
import numpy as np


def interpolate_mass_extraction(t_interp):
    ''' Loads in and interpolates mass extraction data at t_interp times using cubic splines.

        Parameters:
        -----------
        t_interp : array-like
            Vector of times at which to interpolate the mass extraction rate: q

        Returns:
        --------
        q_values : array-like
            Mass extraction rate (kg/s) interpolated at t_interp values.
    '''
    time, mass_rate = np.genfromtxt(
        'sb_mass.txt', delimiter=',', skip_header=1).T

    #Computes coefficients of cublic splines and stores into list of tuples
    splines = interpolate.splrep(time, mass_rate)

    #Computes q values at t_interp values
    q_values = interpolate.splev(t_interp, splines)

    return q_values


def pressure_ode(p, q, p0, a, b, c, dqdt):
    ''' Return the pressure derivative dp/dt for given parameters.

        Parameters:
        -----------
        p : float
            Dependent variable - Pressure
        q : float
            Source/sink rate - Mass extraction in kg/s
        p0 : float
            Ambient value of dependent variable - Initial Pressure
        a : float
            Source/sink strength parameter
        b : float
            Recharge strength parameter
        c :
            Slow-drainage strength parameter
        dqdt :
            Rate of change of source/sink rate

        Returns:
        --------
        dp/dt : float
            Derivative of dependent variable with respect to independent variable.

        Examples:
        ---------
        >>> ode_model(1, 2, 3, 4, 5, 6, 0)
        10
    '''

    return -a * q - b * (p - p0) - c * dqdt


def step_rk4(f, yk, h, args):
    """
    Performs one step of the Classic RK4 method

    Parameters
    ----------
    f : callable
        derivative function
    yk : float
        value of solution at tk
    h : float
        step size
    args : iterable
        Optional parameters to pass into derivative function.

    Returns
    -------
    y(tk+h) : float
        y value over the step h
    """
    #Different args due to different dqdt at evaluated points
    args1= args[:-1] #Excludes dqdt2
    args2= args[:-2] + [(args[-1]+ args[-2])/2] #Takes average of dqdt1 and dqdt2 for middle point
    args3 = args[:-2] + args[-1] #Excludes dqdt1
    
    f0 = f(yk, *args1)
    f1 = f(yk + (h * f0) / 2, *args2)
    f2 = f(yk + (h * f1) / 2, *args2)
    f3 = f(yk + h * f2, *args3)

    return yk + h * ((f0 + 2 * f1 + 2 * f2 + f3) / 6)


def solve_pressure_ode(f, t0, t1, dt, p0, pars):
    ''' Solves pressure_ode numerically.

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
            Initial value of solution - Initial Pressure
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
            2. parameter list [q, p0, pars:(a, b, c), dqdt1, dqdt2]

        Important: Since we are using RK4 method, we need dqdt values
        at 3 different points and so we parse in the first and third dqdt
        value, where the middle dqdt is computed by taking the average of 
        the other 2 points in the rk4 method.
    '''
    #Create solution arrays
    pressure_values = [p0]
    t_range = np.arange(t0, t1 + dt, dt)

    #Load in mass extraction values interpolated at t_range times
    q = interpolate_mass_extraction(t_range)

    #Find the derivative at each point numerically
    dqdt = np.gradient(q, dt)

    for index, tk in enumerate(t_range[:-1]):
        args= [q[index], p0] + pars + [dqdt[index], dqdt[index+1]] #Joining parameters into one list
        pressure_values.append(step_rk4(f, pressure_values[index], dt, args))

    return t_range, np.array(pressure_values)




