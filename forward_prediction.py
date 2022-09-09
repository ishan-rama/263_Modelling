###################################################################################################
# forward_prediction.py - Function library that performs forward predictions on pressure and
#                         subsidence models callibrated in calibration.py
#
#   Functions:
#       forward_pressure: Performs future forecast for different future mass extraction rate
#               outcomes on pressure model.
#       forward_subsidence: Performs future forecast for different future mass extraction rate
#               outcomes on subsidence model.
###################################################################################################

#imports 
import numpy as np
from matplotlib import pyplot as plt

#file imports
from callibration import *
from process_data import *

def forward_pressure(mass_rate_outcomes, display= False):
    """Performs future forecast for different future mass extraction rate outcomes on pressure model.

    Parameters
    ----------
    mass_rate_outcomes : array-like
        a list of future mass extraction rates (integer values only!!!)
    display : boolean, optional
        boolean to either display(TRUE) or save plots(FALSE), by default False
    
    Returns
    -------
    future_pressures : array-like
        a list of lists of future pressure values for each mass extraction rate outcome
    Plot:
        A plot of future pressure forecasts based on different future mass extraction rates given as input
    """

    t0 = 1952 #Start time domain
    t1 = 2013 #End time domain
    tf = 2050 #End future prediction time domain
    dt = 0.01  #Time step
    p0 = 56.26 #Ambient Pressure
    p_init = 56.26 #Initial pressure

    #Getting callibrated parameters from fitted pressure_model
    a, b, c = plot_pressure_model(False)
    pars = [a, b, c]

    t_current = np.arange(t0, t1+dt, dt) #Current time domain
    t_future = np.arange(t1, tf+dt, dt) #Future time domain

    #Load in mass and pressure data
    t_m_data, q_data = load_mass_exraction_data()
    q_current = interpolate_mass_extraction(t_m_data, q_data, t_current)
    dqdt_current = np.gradient(q_current, dt)
    t_p_data, p_data = load_pressure_data()

    #Get interpolation of p_data
    p_interp = interpolate_pressure_data(t_p_data, p_data, t_current)
    
    #List storing list of future pressure values 
    future_pressures = []

    #Forecasting depending on different future mass extraction rates
    for q_rate in mass_rate_outcomes:
        q_future = q_rate * np.ones(len(t_future))
        dqdt = np.gradient(q_future, dt)
        t, future_pressure = solve_pressure_ode(pressure_ode, t1, tf, dt, q_future, dqdt, p_interp[-1], p0, pars)

        future_pressures.append(future_pressure)

    f1, ax1 = plt.subplots(nrows=1, ncols=1)

    #plot future forecast over only the future time domain t1-tf
    for i in range(len(mass_rate_outcomes)):
        ax1.plot(t_future, future_pressures[i],
                label=f"Mass extraction = {mass_rate_outcomes[i]}kg/s")
    
    # plot data point
    ax1.plot(t_p_data, p_data, 'x', color='black', label="Observations")
    # plot model across current time domain
    ax1.plot(t_current, p_interp, label="Interpolation")

    ax1.set_xlabel('Time [year]')
    ax1.set_ylabel('Pressure [bar]')
    ax1.set_title("Forecast for pressure")
    ax1.legend()

    if display:
        plt.show()
    else:
        plt.savefig('forward_pressure_prediction', dpi=300)
        plt.close()

    return future_pressures


def forward_subsidence(mass_rate_outcomes, future_pressures, display=False):
    """Performs future forecast for different future mass extraction rate outcomes on subsidence model.

    Parameters
    ----------
    mass_rate_outcomes : array-like
        a list of future mass extraction rates (integer values only!!!)
    future_pressures : array-like
        future_pressures : array-like
        a list of lists of future pressure values for each mass extraction rate outcome
    display : boolean, optional
        boolean to either display(TRUE) or save plots(FALSE), by default False
    
    Returns
    -------
    future_subsidences : array-like
        a list of lists of future subsidence values for each mass extraction rate outcome
    Plot:
        A plot of future subsidence forecasts based on different future mass extraction rates given as input
    """
    t0 = 1952  # Start time domain
    t1 = 2013  # End time domain
    tf = 2050  # End future prediction time domain
    dt = 0.01  # Time step
    p0 = 56.26 # Ambient Pressure

    #Getting callibrated parameters from fitted subsidence_model
    d, Tm, Td = plot_subsidence_model(False)
    pars = [d, Tm, Td]

    t_current = np.arange(t0, t1+dt, dt) #Current time domain
    t_future = np.arange(t1, tf+dt, dt) #Future time domain

    #Load in pressure and subsidence data and interpolate
    t_p_data, p_data = load_pressure_data()
    p_interp = interpolate_pressure_data(t_p_data, p_data, t_current)
    t_s_data, s_data = load_subsidence_data()
    s_interp = interpolate_subsidence_data(t_s_data, s_data, t_current)
    
    #List storing list of future subsidence values
    future_subsidences = []

    #Forecasting depending on different future pressure values
    for future_pressure in future_pressures:
        t, future_subsidence = solve_subsidence_model(subsidence_model, t1, tf, dt, future_pressure, p0, pars)

        #Offset to match interpolated data
        future_subsidence = [val-0.49131 for val in future_subsidence]
        future_subsidences.append(future_subsidence)

    f1, ax1 = plt.subplots(nrows=1, ncols=1)
    
    #plot future forecast over only the future time domain t1-tf
    for i in range(len(mass_rate_outcomes)):
        ax1.plot(t_future, future_subsidences[i], label=f"Mass extraction = {mass_rate_outcomes[i]}kg/s")
    
    # plot data point
    ax1.plot(t_s_data, s_data, 'x', color='black', label="Observations")
    # plot model across current time domain
    ax1.plot(t_current, s_interp, label="Interpolation")

    ax1.set_xlabel('Time [year]')
    ax1.set_ylabel('Subsidence [metres]')
    ax1.set_title("Forecast for subsidence")
    ax1.legend()

    if display:
        plt.show()
    else:
        plt.savefig('forward_subsidence_prediction', dpi=300)
        plt.close()

    return future_subsidences

if __name__ == '__main__':
    mass_rate_outcomes = [0, 450, 900, 1250]
    future_pressures = forward_pressure(mass_rate_outcomes)
    future_subsidences = forward_subsidence(mass_rate_outcomes, future_pressures)
