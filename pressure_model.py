#########################################################################################
# pressure_model.py
#   Calibrating parameters to fit the pressure data and plots best fitted model.
#########################################################################################

#imports
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from ode_solver import *


def plot_pressure_model(display= False):
    """Calibrating parameters to fit the pressure data and plots best fitted model.

    Returns
    -------
    Plot of best fitted model against pressure data.
    """
    t0 = 1953
    t1 = 2012
    p0 = 56.26
    dt = 0.01

    a = 1.3e-3
    b = 0.07
    c = 0.006
    pars = [p0, a, b, c]
    
    t_range = np.arange(t0, t1 + dt, dt)
    
    t_data, p_data = load_pressure_data()

    #Cubic spline interpolating pressure values at t_range values
    splines = interpolate.splrep(t_data, p_data)
    p1_data = interpolate.splev(t_range, splines)

    plt.plot(t_data, p_data, "x", color="red", label="observations")

    t, p = solve_pressure_ode(pressure_ode, t0, t1, dt, pars)
    plt.plot(t, p, "-", color="blue", label="initial model")

    def Pmodel(t, *pars):
        t0 = 1953
        t1 = 2012
        dt = 0.01
        tm, Tm = solve_pressure_ode(
            pressure_ode, t0, t1, dt, list(pars))
        return Tm

    theta0 = [p0, a, b, c]
    constants = curve_fit(Pmodel, t_data, p1_data, theta0)
    p0_const = constants[0][0]
    a_const = constants[0][1]
    b_const = constants[0][2]
    c_const = constants[0][3]

    print(f'p0= {p0_const}, a = {a_const}, b = {b_const}, c = {c_const}')

    pars = [p0_const, a_const, b_const, c_const]
    t1= 2012

    t, p = solve_pressure_ode(pressure_ode, t0, t1, dt, pars)
    plt.plot(t, p, "-", color="green", label="improved model")

    plt.xlabel('Time [yr]')
    plt.ylabel('Pressure [bar]')
    plt.title('Initial Pressure (p0) = %.2f, a = %.5f, b = %.5f, c = %.5f' %
              (p0_const, a_const, b_const, c_const))
    plt.legend()

    save_figure = 1
    if not save_figure:
        plt.show()
    else:
        plt.savefig('model_plot', dpi=300)
        plt.show()

    return a_const, b_const, c_const
