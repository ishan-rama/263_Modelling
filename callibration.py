###################################################################################################
# callibration.py - Function library that callibrates parameters for pressure and subsidence models
#
#   Functions:  
#       plot_pressure_model: Calibrates parameters (p0, a, b, c) to fit the pressure data and 
#               plots best fitted model using curve_fit from scipy library.
#       plot_subsidence_model: Calibrates parameters (d, Tm, Td) to fit the subsidence data and plots 
#               best fitted model using curve_fit from scipy library.
#       plot_pressure_misfit: Plots a misfit scatter plot of the best fitted pressure model against the data.
#       plot_subsidence_misfit: Plots a misfit scatter plot of the best fitted subsidence model against the interpolated data.
###################################################################################################

#library imports
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

#file imports
from model_solver import *

#To display or not to display that is the question
display = True


def plot_pressure_model(display= False):
    """Calibrating parameters to fit the pressure data and plots best fitted model.

    Parameters:
    -----------
    display : boolean, optional
        boolean to either display(TRUE) or save plots(FALSE), by default False

    Returns:
    -------
    Callibrated Parameters:
        a : float
            Source/sink strength parameter
        b : float
            Recharge strength parameter
        c : float
            Slow-drainage strength parameter
    Plot: 
        Best fitted model against pressure data as well as interpolated pressure
        values and an ad-hoc calibration of the model prior to curve_fit

    Notes
    ------
    Prints callibrated parameters to the standard output
    """
    t0 = 1953 #Start time
    t1 = 2013 #End time
    dt = 0.01 #Time step

    p0 = 56.26 #Ambient Pressure
    p_init= 56.26 #Initial Pressure
    #Parameters - ad-hoc estimates
    a = 2.2e-03
    b = 1.1e-01
    c = 6.8e-03
    pars = [a, b, c]
    
    t_range = np.arange(t0, t1 + dt, dt) #Time range

    #Load in mass extraction values interpolated at t_range times
    t_data, q_data = load_mass_exraction_data()
    q = interpolate_mass_extraction(t_data, q_data, t_range)

    #Find the derivative at each point numerically
    dqdt = np.gradient(q, dt)

    #Interpolate pressure values at t_range values for computing misfit in curve_fit()
    t_data, p_data = load_pressure_data()
    p_interp = interpolate_pressure_data(t_data, p_data, t_range)

    f1, ax1 = plt.subplots(nrows=1, ncols=1)
    plt.plot(t_data, p_data, "x", color="black", label="observations")
    plt.plot(t_range, p_interp, color="blue", label="interpolation")

    t, p = solve_pressure_ode(pressure_ode, t0, t1, dt, q, dqdt, p_init, p0, pars)
    plt.plot(t, p, "-", color="green", label="ad-hoc callibrated model")

    def Pmodel(t, *pars):
        t0 = 1953
        t1 = 2013
        dt = 0.01
        p0 = 56.26
        p_init = 56.26
        t_data, q_data = load_mass_exraction_data()
        q = interpolate_mass_extraction(t_data, q_data, t_range)
        dqdt = np.gradient(q, dt)
        tm, Tm = solve_pressure_ode(
            pressure_ode, t0, t1, dt, q, dqdt, p_init, p0, list(pars))
        return Tm

    theta0 = [a, b, c]
    constants = curve_fit(Pmodel, t_range, p_interp, theta0)
    a_const = constants[0][0]
    b_const = constants[0][1]
    c_const = constants[0][2]

    print(f'a = {a_const}, b = {b_const}, c = {c_const}')

    pars = [a_const, b_const, c_const]

    t, p = solve_pressure_ode(pressure_ode, t0, t1, dt, q, dqdt, p_init, p0, pars)
    plt.plot(t, p, "-", color="red", label="improved model")

    plt.xlabel('Time [yr]')
    plt.ylabel('Pressure [bar]')
    plt.title('a = %.7f, b = %.7f, c = %.7f' %
              (a_const, b_const, c_const))
    plt.legend()

    if display:
        plt.show()
    else:
        plt.savefig('pressure_model_plot', dpi=300)
        plt.close()

    return a_const, b_const, c_const


def plot_subsidence_model(display= False):
    """Calibrating parameters to fit the subsidence data and plots best fitted model.

    Parameters
    ----------
    display : boolean, optional
        boolean to either display(TRUE) or save plots(FALSE), by default False

    Returns
    -------
    Callibrated Parameters:
        d : float
        Lumped parameter
        Tm : float
            year when subsidence event reaches its peak
        Td : float
            diffusion time (years)
    Plot: 
        Best fitted model against subsidence data as well as interpolated subsidence values

    Notes
    ------
    Prints callibrated parameters to the standard output
    """
    
    t0 = 1952 #Start time
    t1 = 2013 #End time
    dt = 0.5 #Time step
    p0 = 56.26 #Ambient Pressure outside the reservoir

    #Parameters - estimated values
    d = 0.904
    Tm = 1979.1
    Td = 8.2
    pars = [d, Tm, Td]

    t_range = np.arange(t0, t1 + dt, dt) #Time range

    #Interpolate pressure and subsidence values at t_range values for computing misfit in curve_fit()
    t_data, p_data = load_pressure_data()
    p = interpolate_pressure_data(t_data, p_data, t_range)
    t_s_data, s_data = load_subsidence_data()
    s_interp = interpolate_subsidence_data(t_s_data, s_data, t_range)

    f1, ax1 = plt.subplots(nrows=1, ncols=1)
    plt.plot(t_s_data, s_data, "x", color="black", label="observations")
    plt.plot(t_range, s_interp, color="blue", label="interpolation")

    t, s = solve_subsidence_model(subsidence_model, t0, t1, dt, p, p0, pars)

    def Smodel(t, *pars):
        t0 = 1952
        t1 = 2013
        dt = 0.5
        p0 = 56.18127 #from pressure model fitted parameter
        t_data, p_data = load_pressure_data()
        p = interpolate_pressure_data(t_data, p_data, t_range)
        t, s = solve_subsidence_model(subsidence_model, t0, t1, dt, p, p0, list(pars))
        return s

    theta0 = [d, Tm, Td]
    constants = curve_fit(Smodel, t_range, s_interp, theta0)
    d_const = constants[0][0]
    Tm_const= constants[0][1]
    Td_const = constants[0][2]

    print(f'd = {d_const}, Tm = {Tm_const}, Td = {Td_const}')

    fitted_pars = [d_const, Tm_const, Td_const]
    t_range, s_fitted = solve_subsidence_model(t_range, t0, t1, dt, p, p0, fitted_pars)

    plt.plot(t_range, s_fitted, "-", color="red", label="improved model")

    plt.xlabel('Time [yr]')
    plt.ylabel('Subsidence [m]')
    plt.title('d = %.7f, Tm = %.7f, Td = %.7f' %(d_const, Tm_const, Td_const))
    plt.legend()
    
    if display:
        plt.show()
    else:
        plt.savefig('subsidence_model_plot', dpi=300)
        plt.close()

    return d_const, Tm_const, Td_const


def plot_pressure_misfit(a, b, c, display= False):
    """Plots a misfit scatter plot of the best fitted pressure model against the data

    Parameters:
    ----------
    a : float
        Source/sink strength parameter
    b : float
        Recharge strength parameter
    c : float
        Slow-drainage strength parameter
    display : boolean, optional
        boolean to either display(TRUE) or save plots(FALSE), by default False
    
    Returns:
    ------
    Plot :
        Plot of the misfit from the best fitted model and the data
    """
    t0 = 1953  # Start time
    t1 = 2013  # End time
    dt = 0.125  # Time step
    p0 = 56.26 #Ambient Pressure
    p_init= 56.26 #Initial Pressure 
    pars = [a, b, c]

    t_range = np.arange(t0, t1 + dt, dt) #Time range
    #Load in mass extraction values interpolated at t_range times
    t_data, q_data = load_mass_exraction_data()
    q = interpolate_mass_extraction(t_data, q_data, t_range)
    #Find the derivative at each point numerically
    dqdt = np.gradient(q, dt)

    #Pressure data
    t_data, p_data = load_pressure_data()

    #Best Model fitted
    t_model, p_model_vals = solve_pressure_ode(pressure_ode, t0, t1, dt, q, dqdt, p_init, p0, pars)

    misfit= [] 
    t_model = list(t_model)
    #Compute misfit array
    for t_dat, p_dat in zip(t_data, p_data):
        index = t_model.index(int(t_dat))
        misfit.append(p_model_vals[index]- p_dat)

    f1, ax1 = plt.subplots(nrows=1, ncols=1)    

    plt.plot(t_data, misfit, "x", color="red", label="misfit")
    plt.axhline(0.0, linestyle= "dotted" )

    plt.xlabel('Time [years]')
    plt.ylabel('Pressure misfit [bars]')
    plt.title('Misfit of best fitted model against data.')
    plt.legend()


    if display:
        plt.show()
    else:
        plt.savefig('pressure_model_misfit', dpi=300)
        plt.close()


def plot_subsidence_misfit(d, Tm, Td, display= False):
    """Plots a misfit scatter plot of the best fitted subsidence model against the interpolated data

    Parameters:
    ----------
    d : float
        Lumped parameter
    Tm : float
        year when subsidence event reaches its peak
    Td : float
        diffusion time (years)
    
    Returns:
    ------
    Plot :
        Plot of the misfit from the best fitted model and the interpolated data
    """
    t0 = 1952  # Start time
    t1 = 2013  # End time
    dt = 0.125  # Time step
    p0 = 56.26  # Ambient Pressure outside the reservoir
    pars = [d, Tm, Td]

    t_range = np.arange(t0, t1 + dt, dt) #Time range
    t_data, p_data = load_pressure_data()
    p = interpolate_pressure_data(t_data, p_data, t_range)

    #Interpolation of Subsidence data
    t_s_data, s_data = load_subsidence_data()
    year_range = np.arange(t0, t1 + dt, 1) #Time range
    s_interp = interpolate_subsidence_data(t_s_data, s_data, year_range)

    #Best Model fitted
    t_model, s_fitted = solve_subsidence_model(t_range, t0, t1, dt, p, p0, pars)

    misfit= [] 
    t_model = list(t_model)
    #Compute misfit array
    for t_dat, s_dat in zip(t_s_data, s_data):
        index = t_model.index(int(t_dat))
        misfit.append(s_fitted[index]- s_dat)

    f1, ax1 = plt.subplots(nrows=1, ncols=1)    

    plt.plot(t_s_data, misfit, "x", color="red", label="misfit")
    plt.axhline(0.0, linestyle= "dotted" )

    plt.xlabel('Time [years]')
    plt.ylabel('Subsidence misfit [bars]')
    plt.title('Misfit of best fitted model against interpolated data.')
    plt.legend()


    if display:
        plt.show()
    else:
        plt.savefig('subsidence_model_misfit', dpi=300)
        plt.close()


if __name__ == "__main__":
    a, b, c = plot_pressure_model(display)
    d, Tm, Td = plot_subsidence_model(display)
    plot_pressure_misfit(a, b, c, display)
    plot_subsidence_misfit(d, Tm, Td, display)
