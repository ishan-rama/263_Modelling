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

def solve_pressure_ode(f, t0, t1, dt, p0, pars):
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

def load_pressure_data():
    ''' Returns time and temperature measurements from kettle experiment.
        Parameters:
        -----------
        none
        Returns:
        --------
        t : array-like
            Vector of times (seconds) at which measurements were taken.
        T : array-like
            Vector of mass extraction measurements.
    '''
    # File I/O commands to read in the data
    Time, Pressure = np.genfromtxt('sb_pres.txt', delimiter=',', skip_header=1).T

    return Time, Pressure

def interpolate_mass_extraction(t):
    ''' Return heat source parameter q for kettle experiment.
        Parameters:
        -----------
        t : array-like
            Vector of times at which to interpolate the heat source.
        Returns:
        --------
        q : array-like
            Heat source (Watts) interpolated at t.
    '''
    # suggested approach
    # hard code vectors tv and qv which define a piecewise heat source for your kettle
    # experiment
    # use a built in Python interpolation function

    Time, Mass = np.genfromtxt('sb_mass.txt', delimiter=',', skip_header=1).T

    return np.interp(t, Time, Mass)

if __name__ == "__main__":
    q = interpolate_mass_extraction(t)