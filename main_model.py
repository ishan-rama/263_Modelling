import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit


def pressure_ode(t, p, p0, q, a, b, c, dqdt):
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
    f1 = f(tk + h / 2, yk + (h * f0) / 2, *args)
    f2 = f(tk + h / 2, yk + (h * f1) / 2, *args)
    f3 = f(tk + h, yk + h * f2, *args)

    return yk + h * ((f0 + 2 * f1 + 2 * f2 + f3) / 6)


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
    # Return arrays
    pressure_values = [p0]
    t_range = np.arange(t0, t1 + dt, dt)

    q = interpolate_mass_extraction(t_range)
    dqdt = q.copy()

    for i in range(len(q)-1):
        dqdt[i] = (q[i+1] - q[i])/dt

    for index, tk in enumerate(t_range[:-1]):
        pressure_values.append(
            step_rk4(f, tk, pressure_values[index], dt, [p0, q[index]] + pars + [dqdt[index]]))

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
    Time, Pressure = np.genfromtxt(
        'sb_pres.txt', delimiter=',', skip_header=1).T

    return Time, Pressure


def interpolate_mass_extraction(t):
    ''' Return heat source parameter q for warakei model.
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


def plot_pressure_model():
    t0 = 1953
    t1 = 2012
    p0 = 56.26
    dt = 0.001  # STEPSIZE
    a = 5.1e-4
    b = 0.65e-3
    c = 0.015
    pars = [a, b, c]

    #RANDOM EXPERIMATION STUFF BY ISHAN TO ALLOW FOR DIFFERENT STEPSIZE
    Time, Pressure = np.genfromtxt('sb_pres.txt', delimiter=',', skip_header=1).T
    t_range = np.arange(t0, t1 + dt, dt)


    t_data, p1_data = load_pressure_data()
    p_data = np.interp(t_range, Time, p1_data)
    plt.plot(t_data, p1_data, "x", color="red", label="observations")
    #####################################################################


    t, p = solve_pressure_ode(pressure_ode, t0, t1, dt, p0, pars)
    plt.plot(t, p, "-", color="blue", label="initial model")

    def Pmodel(t, *pars):
        t0 = 1953
        t1 = 2012
        p0 = 56.26
        dt = 0.001 #STEPSIZE 

        tm, Tm = solve_pressure_ode(pressure_ode, t0, t1, dt, p0, list(pars))
        return Tm

    theta0 = [a, b, c]
    constants = curve_fit(Pmodel, t_data, p_data, theta0)
    a_const = constants[0][0]
    b_const = constants[0][1]
    c_const = constants[0][2]

    print(f'a = {a_const}, b = {b_const}, c = {c_const}')

    pars = [a_const, b_const, c_const]

    dt = 0.001  # STEPSIZE

    t, p = solve_pressure_ode(pressure_ode, t0, t1, dt, p0, pars)
    plt.plot(t, p, "-", color="green", label="improved model")

    plt.xlabel('Time [yr]')
    plt.ylabel('Pressure [bar]')
    plt.title('Model Plot')
    plt.legend()

    save_figure = 1
    if not save_figure:
        plt.show()
    else:
        plt.savefig('model_plot', dpi=300)
        plt.show()

    return a_const, b_const, c_const


def subsidence_model(t, p, d):
    return d * p * (1 - (1 / (1 + np.exp((t - 1979.1) / 8.2))))


def plot_subsidence_model(a, b, c):
    pars = [a, b, c]

    t0 = 1953
    t1 = 2012
    p0 = 56.26
    dt = 1
    d = 0.5

    t, p = solve_pressure_ode(pressure_ode, t0, t1, dt, p0, pars)

    s = p.copy()
    for i in range(len(t)):
        s[i] = subsidence_model(t[i], p[i], d)
    plt.plot(t, s, "-", color="blue", label="improved solution")

    Time, Disp = np.genfromtxt('sb_disp.txt', delimiter=',', skip_header=1).T
    plt.plot(Time, Disp, "x", color="red", label="observations")

    plt.xlabel('Time [yr]')
    plt.ylabel('Subsidence [m]')
    plt.title('Model Plot')
    plt.legend()

    plt.show()


if __name__ == "__main__":
    a, b, c = plot_pressure_model()
    #plot_subsidence_model(a, b, c)
