#########################################################################
# preliminary_plots.py - Plot given raw data for exploratory analysis
#########################################################################

#library imports
import numpy as np
from matplotlib import pyplot as plt

#file imports
from process_data import *

#To display or not to display that is the question
display= True


def prelim_plots(display= False):
    """Plot given raw data for exploratory analysis
    """
    t0 = 1953 #Start time
    t1 = 2013 #End time
    dt = 0.01 #Time step

    t_range = np.arange(t0, t1 + dt, dt)  # Time range

    #Load in mass extraction values interpolated at t_range times
    t_q_data, q_data = load_mass_exraction_data()
    q_interp = interpolate_mass_extraction(t_q_data, q_data, t_range)

    #Interpolate pressure values at t_range values interpolated at t_range times
    t_p_data, p_data = load_pressure_data()
    p_interp = interpolate_pressure_data(t_p_data, p_data, t_range)

    #Interpolate subsidence values at t_range values interpolated at t_range times
    t_s_data, s_data = load_subsidence_data()
    s_interp = interpolate_subsidence_data(t_s_data, s_data, t_range)

    fig, ax1 = plt.subplots()

    ax1.plot(t_range, p_interp, color= "red", label= "PRESSURE")
    ax1.set_ylabel('PRESSURE (bar)', color= "red")

    ax2 = ax1.twinx()
    ax2.plot(t_range, s_interp, color= "blue", label= "SUBSIDENCE")
    ax2.set_ylabel('SUBSIDENCE (m)', color="blue")

    ax3 = ax1.twinx()
    ax3.plot(t_range, q_interp, color = "black", label= "MASS EXTRACTION RATE")
    ax3.spines['right'].set_position(('outward', 40))
    ax3.set_ylabel('MASS EXTRACTION RATE (kg/s)', color="black")

    ax1.tick_params(axis= 'y', colors= "red")
    ax2.tick_params(axis= 'y', colors= "blue")
    ax3.tick_params(axis= 'y', colors= "black")

    ax2.spines['right'].set_color("blue")
    ax3.spines['right'].set_color("black")
    ax3.spines['left'].set_color("red")

    plt.title("MASS EXTRACTION, PRESSURE and SUBSIDENCE at WAIRAKEI OVER TIME")
    fig.legend(loc = 'upper center', frameon = True)

    if display:
        plt.show()
    else:
        plt.savefig('prelim_plot', dpi=300)
        plt.close()


if __name__ == '__main__':
    prelim_plots(display)
