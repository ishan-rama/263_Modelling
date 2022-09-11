###################################################################################################
# inverse_model_use.py - Function library that performs inverse modelling.
#
#   Functions:
#       inverse_modelling: 
###################################################################################################


#imports 
import numpy as np
from matplotlib import pyplot as plt
import statistics as stat

#file imports
from forward_prediction import *

#To display or not to display that is the question
display = False


def inverse_modelling(pressure_samples, display):
    """Plots histogram of parameter 'a' samples used for inverse_modelling

    Parameters
    ----------
    pressure_samples : array-like
		parameter samples from the multivariate normal
    display : boolean, optional
        boolean to either display(TRUE) or save plots(FALSE), by default False
    """
    f, ax1 = plt.subplots(1, 1)

    #Get a samples from pressure parameter samples
    a_samples = [sample[0] for sample in pressure_samples]

    bins = np.linspace(np.min(a_samples)*0.999, np.max(a_samples)
                       * 1.001, int(np.sqrt(len(a_samples)))+1)
    ax1.hist(a_samples, bins, density= True, stacked=True)
    #Calculating standard deviation and mean
    mu = stat.mean(a_samples)
    sd = stat.stdev(a_samples)

    confint_90 = 0.9*(sd/(np.sqrt(len(a_samples))))

    print(mu)
    print(confint_90)

    ax1.set_xlabel('a_value')
    ax1.set_ylabel('Probability density of parameter a value')
    ax1.set_title("PDF of parameter 'a' value with mean_a = {mu}")
    ax1.axvline(x= 0.0014389, color="r", linestyle='--', label="best fitted value a= 0.0014389")
    ax1.axvline(x=mu+confint_90, color="grey", linestyle='--',
                label="90%/ confidence interval")
    ax1.axvline(x=mu-confint_90, color="grey", linestyle='--')
    ax1.legend()

    if display:
        plt.show()
    else:
        f.savefig('inverse_modelling', dpi=300)


if __name__ == '__main__':
    #GETTING INPUTS MAY TAKE A MINUTE OR TWO
    #########################################################################
    mass_rate_outcomes = [0, 450, 900, 1250]
    future_pressures_best = forward_pressure(mass_rate_outcomes, display)
    #future_subsidences_best = forward_subsidence(mass_rate_outcomes, future_pressures_best, display)

    #Sampled Parameters
    N_samples = 100

    #Prediction of pressure model
    a, b, c, P = pressure_grid_search()
    pressure_samples = construct_pressure_samples(a, b, c, P, N_samples)
    future_pressures = foward_pressure_uncertainty(mass_rate_outcomes, pressure_samples, display)
    ###########################################################################
    
    #Plot for inverse_modelling of parameter a
    a = inverse_modelling(pressure_samples, False)
    
