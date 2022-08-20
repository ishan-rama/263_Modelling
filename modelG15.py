import numpy as np

def pressure_ode(t, p, q, a, b, p0, c, dqdt):
    ''' Return the derivative dp/dt at time, t, for given parameters.

        Parameters:
        -----------
        t : float
            Independent variable.
        p : float
            Dependent variable.
        q : float
            Source/sink rate.
        a : float
            Source/sink strength parameter.
        b : float
            Recharge strength parameter.
        p0 : float
            Ambient value of dependent variable.
        c :
            Slow-drainage parameter.
        dqdt :
            Rate of change of source/sink rate.

        Returns:
        --------
        dxdt : float
            Derivative of dependent variable with respect to independent variable.

        Notes:
        ------
        None

        Examples:
        ---------
        >>> ode_model(0, 1, 2, 3, 4, 5, 6, 0)
        10

    '''

    return -a * q - b * (p - p0) - c * dqdt

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
        t0 : float
            Initial time of solution.
        t1 : float
            Final time of solution.
        dt : float
            Time step length.
        p0 : float
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