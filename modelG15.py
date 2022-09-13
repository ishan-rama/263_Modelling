from cProfile import label
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


def solve_pressure_ode(f, t0, t1, dt, p0,q, dqdt, pars):
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
    Time, Pressure = np.genfromtxt('sb_pres.txt', delimiter=',', skip_header=1).T

    return Time, Pressure

def load_subsidence_data():
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
    Time, Subsidence = np.genfromtxt('sb_disp.txt', delimiter=',', skip_header=1).T

    return Time, Subsidence


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


def plot_pressure_model():
    t0 = 1953
    t1 = 2012
    p0 = 56.26
    dt = 0.01
    a = 5.1e-4
    b = 0.65e-3
    c = 0.015
    pars = [a, b, c]
    t_range = np.arange(t0, t1 + dt, dt)
    q = interpolate_mass_extraction(t_range)
    dqdt = q.copy()

    for i in range(len(q)-1):
        dqdt[i] = (q[i+1] - q[i])/dt

    t_data, p_data = load_pressure_data()
    p1_data = np.interp(t_range, t_data, p_data)
    plt.plot(t_data, p_data, "x", color="red", label="observations")

    t, p = solve_pressure_ode(pressure_ode, t0, t1, dt, p0, q, dqdt, pars)
    plt.plot(t, p, "-", color="blue", label="initial model")


    def Pmodel(t, *pars):
        t0 = 1953
        t1 = 2012
        p0 = 56.26
        dt = 0.01

        tm, Tm = solve_pressure_ode(pressure_ode, t0, t1, dt, p0, q, dqdt, list(pars))
        return Tm

    theta0 = [a, b, c]
    constants = curve_fit(Pmodel, t_data, p1_data, theta0)
    a_const = constants[0][0]
    b_const = constants[0][1]
    c_const = constants[0][2]

    print(f'a = {a_const}, b = {b_const}, c = {c_const}')

    pars = [a_const, b_const, c_const]

    t, p = solve_pressure_ode(pressure_ode, t0, t1, dt, p0, q, dqdt, pars)
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


def forward_prediction(qf):
    '''Return heat source parameter q for kettle experiment.
        Parameters:
        -----------
        qf : array
            an array including the future mass extraction rates.
        Returns:
        --------
        Nothing
    '''
    a = 0.0014699848529127948
    b = 0.06322104915197868
    c = 0.008719645302839404
    pars = [a, b, c]
    p0 = 56.26
    dt = 1
    d = 0.741947934080013
    d_t = 12.161738208806344
    tm = 1983.9891553334935

    tp, p = load_pressure_data()
    ts, s = load_subsidence_data()
    t0 = tp[0]
    
    t_current = np.arange(t0,tp[-1]+2,dt)
    t_future = np.arange(tp[-1]+2,2030.5,dt)
    t = np.concatenate((t_current,t_future))

    q0 = interpolate_mass_extraction(t_current)

    pressure = []
    subsidence = []

    for i in range(len(qf)):
        q1 = qf[i]*np.ones(len(t_future))
        q = np.concatenate((q0,q1))
        dqdt = q.copy()
        for j in range(len(q)-1):
            dqdt[j] = (q[j+1] - q[j])/dt
        
        t, pres = solve_pressure_ode(pressure_ode, t0, 2029.5, dt, p0, q, dqdt, pars)
        subs = subsidence_model(t, pres, d, d_t, tm)

        pressure.append(pres)
        subsidence.append(subs)
    
    f, (ax1,ax2) = plt.subplots(1,2)

    # plot forward prediction
    for i in range(len(qf)):
        ax1.plot(t,pressure[i],label=f"Mass extraction = {qf[i]}kg/s")
        ax2.plot(t,subsidence[i],label=f"Mass extraction = {qf[i]}kg/s")

    # plot data point
    ax1.plot(tp, p, 'x', label="Data points")
    ax2.plot(ts, s, 'x', label="Data points")

    # plot model
    ax1.plot(t_current,pressure[0][:len(t_current)], label="Model")
    ax2.plot(t_current,subsidence[0][:len(t_current)], label="Model")

    ax1.set_xlabel('Time [yr]')
    ax1.set_ylabel('Pressure [bar]')
    ax1.set_title("Forecast for pressure")
    ax1.legend()

    ax2.set_xlabel('Time [yr]')
    ax2.set_ylabel('subsidence [m]')
    ax2.set_title("Forecast for subsidence")
    ax2.legend()

    save_figure = 1
    f.set_size_inches(20.,9.)
    if not save_figure:
        plt.show()
    else:
        plt.savefig('forward prediction', dpi=300)

def subsidence_model(t, p, d, diffuse_t, t_max):
    #Default diffuse_t = 8.2 (diffusion time)
    p0 = 56.26
    s = []
    for i in range(len(t)):
        s.append(
            d * (p0 - p[i]) * (1 - (1 / (1 + np.exp((t[i] - t_max) / diffuse_t)))))

    return np.array(s)

def plot_subsidence_model(a,b,c):
    pars = [a, b, c]

    t0 = 1953
    t1 = 2013
    p0 = 56.26
    dt = 1
    d = 1
    diffuse_t = 8.2
    t_max = 1979.1

    t_range = np.arange(t0, t1 + dt, dt)

    q = interpolate_mass_extraction(t_range)
    dqdt = q.copy()

    for i in range(len(q)-1):
        dqdt[i] = (q[i+1] - q[i])/dt

    t, p = solve_pressure_ode(pressure_ode, t0, t1, dt, p0, q, dqdt, pars)

    Time, Disp = load_subsidence_data()
    plt.plot(Time, Disp, "x", color="red", label="observations")

    full_disp_values = np.interp(t, Time, Disp)

    s = subsidence_model(t, p, d, diffuse_t, t_max)

    def Smodel(t, *pars):

        s = subsidence_model(t, p, *pars)

        return s

    theta0 = [d, diffuse_t, t_max]
    constants = curve_fit(Smodel, t, full_disp_values, theta0)
    d_const = constants[0][0]
    diffuse_t_const = constants[0][1]
    t_max_const  = constants[0][2]

    print(f'd = {d_const}, diffuse_t_const = {diffuse_t_const}, t_max = {t_max_const}')

    s3 = subsidence_model(t, p, d_const, diffuse_t_const, t_max_const)

    plt.plot(t, s3, "-", color="blue", label="improved solution")

    plt.xlabel('Time [yr]')
    plt.ylabel('Subsidence [m]')
    plt.title('Model Plot')
    plt.legend()

    save_figure = 1
    if not save_figure:
        plt.show()
    else:
        plt.savefig('subsidence plot', dpi=300)

if __name__ == "__main__":
    a, b, c = plot_pressure_model()
    plot_subsidence_model(a, b, c)
    forward_prediction([1250,900,450,0])