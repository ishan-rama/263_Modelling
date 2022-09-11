#########################################################################################
# process_data.py - Function library for loading and interpolating data.
#
# 	Functions:
#       load_mass_exraction_data: Returns time and mass extraction data from sb_mass.txt.
#       interpolate_mass_extraction: Interpolates mass extraction data at given time points using cubic splines.
#       load_pressure_data: Returns time and pressure measurements from sb_pres.txt.
#       interpolate_pressure_data: Interpolate pressure data at given time points using cubic splines.
#       load_subsidence_data: Returns time and subsidence measurements from sb_disp.txt.
#       interpolate_subsidence_data: Interpolate subsidence data at given time points using cubic splines.
#########################################################################################

#imports
from scipy import interpolate
import numpy as np
#sb_disp.txt, sb_mass.txt, sb_pres.txt files are required!!!


def load_mass_exraction_data():
    ''' Returns time and mass extraction data from sb_mass.txt.

    Parameters:
    -----------
    none

    Returns:
    --------
    time : array-like
        Vector of times (seconds) at which measurements were taken.
    mass_rate : array-like
        Vector of mass extraction rate values (kg/s).
    '''
    # File I/O commands to read in the data
    time, mass_rate = np.genfromtxt(
        'sb_mass.txt', delimiter=',', skip_header=1).T

    return time, mass_rate


def interpolate_mass_extraction(t_data, q_data, t_interp):
    '''Interpolate mass extraction data at t_interp times using cubic splines.

        Parameters:
        -----------
        t_data : array-like
            Vector of times of when mass extraction rate data was measured
        p_data : array-like
            Vector of mass extraction rate values measured at t_data values
        t_interp : array-like
            Vector of times at which to interpolate the mass extraction rate: q

        Returns:
        --------
        q_values : array-like
            Mass extraction rate (kg/s) interpolated at t_interp values.
    '''
    #Computes coefficients of cublic splines and stores into list of tuples
    splines = interpolate.splrep(t_data, q_data)
    #Computes q values at t_interp values
    q_values = interpolate.splev(t_interp, splines)

    return q_values


def load_pressure_data():
    ''' Returns time and pressure measurements from sb_pres.txt.

        Parameters:
        -----------
        none

        Returns:
        --------
        time : array-like
            Vector of times (year)
        pressure : array-like
            Vector of pressure measurements (bars)
    '''
    # File I/O commands to read in the data
    time, pressure = np.genfromtxt(
        'sb_pres.txt', delimiter=',', skip_header=1).T

    return time, pressure


def interpolate_pressure_data(t_data, p_data, t_interp):
    """Interpolate pressure data at given time points

    Parameters
    ----------
    t_data : array-like
        Vector of times of when pressure data was measured
    p_data : array-like
        Vector of pressure values measured at t_data values
    t_interp : array-like
        Vector of times at which to interpolate the pressure
    
    Returns:
    --------
    p_values : array-like
        Pressure (bar) interpolated at t_interp values.
    """
    #Computes coefficients of cublic splines and stores into list of tuples
    splines = interpolate.splrep(t_data, p_data)
    #Computes q values at t_interp values
    p_values = interpolate.splev(t_interp, splines)

    return p_values


def load_subsidence_data():
    ''' Returns time and subsidence measurements from sb_disp.txt.

        Parameters:
        -----------
        none

        Returns:
        --------
        time : array-like
            Vector of times (seconds) at which measurements were taken.
        subsidence : array-like
            Vector of subsidence values (metres).
    '''
    # File I/O commands to read in the data
    time, subsidence = np.genfromtxt(
        'sb_disp.txt', delimiter=',', skip_header=1).T

    return time, subsidence


def interpolate_subsidence_data(t_data, s_data, t_interp):
    """Interpolate subsidence data at given time points

    Parameters
    ----------
    t_data : array-like
        Vector of times of when subsidence data was measured
    s_data : array-like
        Vector of subsidence values measured at t_data values
    t_interp : array-like
        Vector of times at which to interpolate the subsidence
    
    Returns:
    --------
    s_values : array-like
        Subsidence (metres) interpolated at t_interp values.
    """
    #Computes coefficients of cublic splines and stores into list of tuples
    splines = interpolate.splrep(t_data, s_data)
    #Computes q values at t_interp values
    s_values = interpolate.splev(t_interp, splines)

    return s_values
