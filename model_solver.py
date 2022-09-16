#########################################################################################
# model_solver.py - Function library for models and solvers.
#
# 	Functions:
#		pressure_ode: Returns the derivative dp/dt for given parameters.
#		step_rk4: Performs one step of the Classic RK4 method.
#		solve_pressure_ode: Solves pressure_ode numerically using above functions.
#       subsidence_model: Computes subsidence at a given time point.
#       solve_subsidence_model: Solves subsidence_model numerically.
#########################################################################################

#imports
from decimal import DivisionByZero
from scipy import interpolate
import numpy as np

#file imports
from process_data import *


def pressure_ode(p, q, p0, a, b, c, dqdt):
    ''' Return the pressure derivative dp/dt for given parameters.

        Parameters:
        -----------
        p : float
            Dependent variable - Pressure
        q : float
            Source/sink rate - Mass extraction in kg/s
        p0 : float
            Ambient value of dependent variable 
        a : float
            Source/sink strength parameter
        b : float
            Recharge strength parameter
        c : float
            Slow-drainage strength parameter
        dqdt : float
            Rate of change of source/sink rate

        Returns:
        --------
        dp/dt : float
            Derivative of dependent variable with respect to independent variable.

        Examples:
        ---------
        >>> pressure_ode(1, 2, 3, 4, 5, 6, 0)
        2
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
    args : array-like
        Parameter list -[q, p0, a, b, c, dqdt]
        q : array-like
            mass extraction rate over time range t0-t1
        p0 : float
            Ambient pressure outside the reservoir
        a : float
            Source/sink strength parameter
        b : float
            Recharge strength parameter
        c : float
            Slow-drainage strength parameter
        dqdt : float
            Rate of change of source/sink rate

    Returns
    -------
    y(tk+h) : float
        y value over the step h
    """
    #Different args due to different dqdt at evaluated points
    args1= args[:-1] #Excludes dqdt2
    args2= args[:-2] + [(args[-1]+ args[-2])/2] #Takes average of dqdt1 and dqdt2 for middle point
    args3 = args[:-2] + [args[-1]] #Excludes dqdt1
    
    f0 = f(yk, *args1)
    f1 = f(yk + (h * f0) / 2, *args2)
    f2 = f(yk + (h * f1) / 2, *args2)
    f3 = f(yk + h * f2, *args3)

    return yk + h * ((f0 + 2 * f1 + 2 * f2 + f3) / 6)


def solve_pressure_ode(f, t0, t1, dt, q, dqdt, p_init, p0, pars):
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
        q : array-like
            mass extraction rate over time range t0-t1
        dqdt : array-like
            change in mass extraction over time range t0-t1
        p_init: float
            initial condition - starting pressure at t0
        p0 : float
            Ambient pressure outside the reservoir
        pars : array-like
            Strength parameters - [a, b, c]

        Returns:
        --------
        t_range : array-like
            Independent time variable vector.
        pressure_values : array-like
            Pressure solution vector.

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
    pressure_values = [p_init]
    t_range = np.arange(t0, t1 + dt, dt)

    for index in (range(len(t_range)-1)):
        args= [q[index], p0] + pars + [dqdt[index], dqdt[index+1]] #Joining parameters into one list
        pressure_values.append(step_rk4(f, pressure_values[index], dt, args))

    return t_range, np.array(pressure_values)


def subsidence_model(t, p_change, d, Tm, Td):
    """Computes subsidence at a given time point.

    Parameters
    ----------
    t : float
        year of when subsidence is being computed
    p_change : float
        change in pressure with respect to ambient pressure
    d : float
        Lumped parameter
    Tm : float
        year when subsidence event reaches its peak
    Td : float
        diffusion time (years)

    Returns
    -------
    s : float
        subsidence (meters)
    """
    try:
        s = d * p_change * (1 - (1/(1+np.exp((t-Tm)/Td))))
    except ZeroDivisionError:
        s = d * p_change  #Ingores exp component
        pass

    return s


def solve_subsidence_model(f, t0, t1, dt, p, p0, pars):
    """Solves subsidence_model numerically.

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
    #Create solution arrays
    s_values = []
    t_range = np.arange(t0, t1 + dt, dt)

    for index in range(len(t_range)):
        p_change = p0 - p[index] #Calculating pressure change with fixed parameter p0
        s = subsidence_model(t_range[index], p_change, *pars)
        s_values.append(s)
   
    return t_range, s_values
