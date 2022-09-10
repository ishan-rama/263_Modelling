#########################################################################################################
# main.py - Main File that generates all plots, Set display to False if you would like to save the plots.
#########################################################################################################

#file imports
from benchmark import *
from callibration import *
from uncertainty import *
from forward_prediction import *

#To display or not to display that is the question
display = True

#Switches to run certain functions from certain files
benchmark = True
callibration = True
uncertainty = True
prediction = True


if __name__ == "__main__":
    if benchmark:
        #Benchmark
        plot_benchmark_pressure(display) 
        plot_benchmark_subsidence(display)
    if callibration:
        #Best fitted models
        a, b, c = plot_pressure_model(display)
        d, Tm, Td = plot_subsidence_model(display)
        plot_pressure_misfit(a, b, c, display)
        plot_subsidence_misfit(d, Tm, Td, display)
    if uncertainty:
        #Part 1 - Observation error plots
        observation_err_pressure(display)
        observation_err_subsidence(display)

        #Part 2 Posterior Plots
        a, b, c, P = pressure_grid_search()
        plot_pressure_posterior3D(a, b, c, P, display)
        d, Tm, Td, P = subsidence_grid_search()
        plot_subsidence_posterior3D(d, Tm, Td, P, display)

        #Sampled Parameters : Sample size = 100
        N = 100
        pressure_samples = construct_pressure_samples(a, b, c, P, N)
        plot_pressure_samples3D(a, b, c, P, pressure_samples, display)
        subsidence_samples = construct_subsidence_samples(d, Tm, Td, P, N)
        plot_subsidence_samples3D(d, Tm, Td, P, subsidence_samples, display)

        #Part 3 Model Ensemble
        pressure_model_ensemble(pressure_samples, display)
        subsidence_model_ensemble(subsidence_samples, display)
    if prediction:
        mass_rate_outcomes = [0, 450, 900, 1250]
        future_pressures_best = forward_pressure(mass_rate_outcomes, display)
        future_subsidences_best = forward_subsidence(
            mass_rate_outcomes, future_pressures_best, display)

        #Sampled Parameters 
        N_samples = 50

        #Prediction of pressure model
        a, b, c, P = pressure_grid_search()
        pressure_samples = construct_pressure_samples(a, b, c, P, N_samples)
        future_pressures = foward_pressure_uncertainty(
            mass_rate_outcomes, pressure_samples, display)

        #Prediction of subsidence model
        d, Tm, Td, P = subsidence_grid_search()
        subsidence_samples = construct_subsidence_samples(d, Tm, Td, P, N_samples)
        future_subsidences = foward_subsidence_uncertainty(mass_rate_outcomes, future_pressures, subsidence_samples, display)

