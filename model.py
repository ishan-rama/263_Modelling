# imports
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit


def ode_model(t, x, q, x0, a, b):
    ''' Return the derivative dx/dt at time, t, for given parameters.

        Parameters:
        -----------
        t : float
            Independent variable.
        x : float
            Dependent variable.
        q : float
            Source/sink rate.
        a : float
            Source/sink strength parameter.
        b : float
            Recharge strength parameter.
        x0 : float
            Ambient value of dependent variable.

        Returns:
        --------
        dxdt : float
            Derivative of dependent variable with respect to independent variable.

        Notes:
        ------
        None

        Examples:
        ---------
        //>>> ode_model(0, 1, 2, 3, 4, 5)
        //22

    '''

    dxdt = -a * q - b * (x - x0)

    return dxdt


def solve_ode(f, t0, t1, dt, x0, pars=None):
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
        ODE should be solved using the Improved Euler Method.

        Function q(t) should be hard coded within this method. Create duplicates of
        solve_ode for models with different q(t).

        Assume that ODE function f takes the following inputs, in order:
            1. independent variable
            2. dependent variable
            3. forcing term, q
            4. all other parameters
    '''
    if pars is None:
        pars = []

    t = [t0]
    result = [x0]

    for i in np.arange(t0 + dt, t1 + 1, dt):
        t.append(i)
        # Evaluate the predictor value
        f0 = f(i, result[-1], *pars)
        # Evaluate the corrector value
        f1 = f(i + dt, result[-1] + dt * f0, *pars)

        # calculate the solution
        x = result[-1] + (dt * (f0 / 2 + f1 / 2))
        result.append(x)

    return t, result

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


def solve_ode_pressure(f, t, x0, pars=None):
    ''' Solve an ODE numerically.

        Parameters:
        -----------
        f : callable
            Function that returns dxdt given variable and parameter inputs.
        t : array-like
            List of time of solution.
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
        ODE should be solved using the Improved Euler Method.

        Function q(t) should be hard coded within this method. Create duplicates of
        solve_ode for models with different q(t).

        Assume that ODE function f takes the following inputs, in order:
            1. independent variable
            2. dependent variable
            3. forcing term, q
            4. all other parameters

    '''
    if pars is None:
        pars = []

    dt = t[1] - t[0]
    x = 0. * t
    x[-1] = x0
    q = interpolate_mass_extraction(t)

    for i in range(0, len(t)):
        dxdtp = f(t[i - 1], x[i - 1], q[i - 1], x0, *pars)
        xp = x[i - 1] + dt * dxdtp
        dxdtc = f(t[i], xp, q[i], x0, *pars)
        x[i] = x[i - 1] + dt * (dxdtp + dxdtc) / 2.

    return t, x


def plot_pressure_model():
    ''' Plot the kettle LPM over top of the data.

        Parameters:
        -----------
        none

        Returns:
        --------
        none

        Notes:
        ------
        This function called within if __name__ == "__main__":

        It should contain commands to read and plot the experimental data, run and
        plot the kettle LPM for hard coded parameters, and then either display the
        plot to the screen or save it to the disk.

    '''
    Time, Pressure = load_pressure_data()


    def Tmodel(t, *pars):
        x0=56.26
        tm,Tm = solve_ode_pressure(ode_model, t, x0, pars)
        return Tm

    p0=[5.1e-4,0.65e-3]
    constants=curve_fit(Tmodel,Time, Pressure, p0)

    a_const=constants[0][0]
    b_const=constants[0][1]
    print(a_const)
    print(b_const)
    pars=[a_const, b_const]

    t, pressure = solve_ode_pressure(ode_model, Time, 56.26, pars)
    plt.plot(Time, Pressure, 'ro')
    plt.plot(Time, pressure)
    plt.xlabel('Time [yr]')
    plt.ylabel('Pressure [bar]')
    plt.show()

if __name__ == "__main__":
    plot_pressure_model()