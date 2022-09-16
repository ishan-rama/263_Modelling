#########################################################################################
# uncertainty.py - Perfoming uncertainty analysis on both pressure and subsidence models
#
# 	Part 1: Observation error
#		observation_err_pressure: Plotting pressure data with observation error
#      	observation_err_subsidence: Plotting subsidence data with observation error
#	Part 2: Parameter uncertainty - Posterior 
#   	pressure_grid_search: This function implements a grid search to compute the posterior over a and b and c
#       subsidence_grid_search: This function implements a grid search to compute the posterior over d and Tm and Td
#       construct_pressure_samples: This function constructs samples from a multivariate normal distribution
#			fitted to the pressure data.
#       construct_subsidence_samples: This function constructs samples from a multivariate normal distribution
#           fitted to the subsidence data.
#	Part 3: Model Ensemble
#       pressure_model_ensemble: Runs the pressure model for given parameter samples and plots the results.
#       subsidence_model_ensemble: Runs the subsidence model for given parameter samples and plots the results.
#########################################################################################

#library imports
import numpy as np
from matplotlib import pyplot as plt

#file imports
from model_solver import *
from uncertainty_plotting import *

#To display or not to display that is the question
display = True


def	observation_err_pressure(display= False):
	"""Plotting pressure data with observation error

	Parameters
	----------
	display : boolean, optional
        boolean to either display(TRUE) or save plots(FALSE), by default False
	"""
	#load pressure data
	t_p_data, p_data = load_pressure_data()

	# plot some data
	f,ax1 = plt.subplots(1,1,figsize=(12,6))

	plt.plot([],[],'ro',label='pressure observation error')
	ax1.set_xlabel('time [yr]')

	plt.plot(t_p_data, p_data, 'ro')
	#variation by 2 bars
	v = 2.
	for tpi,pi in zip(t_p_data, p_data):
		plt.plot([tpi,tpi],[pi-v,pi+v], 'r-', lw=0.5)
	ax1.set_ylabel('pressure [bar]');
	ax1.set_xlim([None, 2013])
	plt.title("Observation error of pressure data by 2 bar variance")
	plt.legend()

	if display:
		plt.show()
	else:
		plt.savefig('observation_err_pressure', dpi=300)
		plt.close()


def	observation_err_subsidence(display= False):
	"""Plotting subsidence data with observation error

	Parameters
	----------
	display : boolean, optional
        boolean to either display(TRUE) or save plots(FALSE), by default False
	"""
    #load subsidence data
	t_s_data, s_data = load_subsidence_data()

	# plot some data
	f,ax1 = plt.subplots(1,1,figsize=(12,6))

	plt.plot([],[],'ro',label='subsidence observation error')
	ax1.set_xlabel('time [yr]')

	plt.plot(t_s_data, s_data, 'ro')
	#variation by 0.5 metres
	v = 0.5
	for tpi, pi in zip(t_s_data, s_data):
		plt.plot([tpi,tpi],[pi-v,pi+v], 'r-', lw=0.5)
	ax1.set_ylabel('subsidence [metres]');
	ax1.set_xlim([None,2012])
	plt.title("Observation error of subsidence data by 0.5m variance")
	plt.legend()

	if display:
		plt.show()
	else:
		plt.savefig('observation_err_subsidence', dpi=300)
		plt.close()


def pressure_grid_search():
	''' This function implements a grid search to compute the posterior over a and b and c

		Returns:
		--------
		a : array-like
			Vector of 'a' parameter values.
		b : array-like
			Vector of 'b' parameter values.
		c : array-like
			Vector of 'c' parameter values.
		P : array-like
			Posterior probability distribution.
	'''
	# 1. define parameter ranges for the grid search
	a_best = 0.0014389
	b_best = 0.0615643
	c_best = 0.0080875

	# number of values considered for each parameter within a given interval
	N = 51

	# vectors of parameter values
	a = np.linspace(a_best/2, a_best*1.5, N)
	b = np.linspace(b_best/2, b_best*1.5, N)
	c = np.linspace(c_best/2, c_best*1.5, N)

	# grid of parameter values: returns every possible combination of parameters in a and b and c
	A, B, C = np.meshgrid(a, b, c, indexing='ij')

	# empty 3D matrix for objective function
	S = np.zeros(A.shape)

	#Fixed parameters 
	p_init, p0 = 56.26, 56.26
	# data for calibration from 1953 to 2000
	t0, t1, dt = 1953, 2000, 1
	t_range = np.arange(t0, t1+ dt, dt)
	t_p_data, p_data = load_pressure_data()
	p_interp = interpolate_pressure_data(t_p_data, p_data, t_range)

	#Load in mass extraction values interpolated at t_range times
	t_data, q_data = load_mass_exraction_data()
	q = interpolate_mass_extraction(t_data, q_data, t_range)
	#Find the derivative at each point numerically
	dqdt = np.gradient(q, dt)

	#error variance - 2bar
	v= 2.

	# grid search algorithm
	for i in range(len(a)):
			for j in range(len(b)):
				for k in range(len(c)):
					pars = [a[i], b[j], c[k]]
					t, p_model = solve_pressure_ode(pressure_ode, t0 ,t1, dt, q, dqdt, p_init, p0, pars)
					S[i,j,k] = np.sum((p_interp-p_model)**2)/v

	#compute the posterior
	P = np.exp(-S/2.)

	# normalize to a probability density function
	Pint = np.sum(P)*(a[1]-a[0])*(b[1]-b[0])*(c[1]-c[0])
	P = P/Pint

	return a, b, c, P


def subsidence_grid_search():
	''' This function implements a grid search to compute the posterior over d and Tm and Td

	Returns:
	--------
	d : array-like
		Vector of 'd' parameter values.
	Tm : array-like
		Vector of 'Tm' parameter values.
	Td : array-like
		Vector of 'Td' parameter values.
	P : array-like
		Posterior probability distribution.
	'''
	# 1. define parameter ranges for the grid search
	d_best = 0.700891
	Tm_best = 1982.30404
	Td_best = 10.8336

	# number of values considered for each parameter within a given interval
	N = 51

	# vectors of parameter values
	d = np.linspace(d_best/2, d_best*1.5, N)
	Tm = np.linspace(Tm_best/2, Tm_best*1.5, N)
	Td = np.linspace(Td_best/2, Td_best*1.5, N)

	# grid of parameter values: returns every possible combination of parameters in a and b and c
	A, B, C = np.meshgrid(d, Tm, Td, indexing='ij')

	# empty 3D matrix for objective function
	S = np.zeros(A.shape)

	# data for calibration from 1953 to 2000
	t0, t1, dt = 1953, 2000, 1
	p0 = 56.26
	t_range = np.arange(t0, t1 + dt, dt)
	t_p_data, p_data = load_pressure_data()
	p_interp = interpolate_pressure_data(t_p_data, p_data, t_range)
	t_s_data, s_data = load_subsidence_data()
	s_interp = interpolate_subsidence_data(t_s_data, s_data, t_range)

	#error variance - 0.5m
	v= 0.5

	# grid search algorithm
	for i in range(len(d)):
		for j in range(len(Tm)):
			for k in range(len(Td)):
				pars = [d[i], Tm[j], Td[k]]
				t, s_model = solve_subsidence_model(pressure_ode, t0 ,t1, dt, p_interp, p0, list(pars))
				S[i,j,k] = np.sum((s_interp-s_model)**2)/v

	#compute the posterior
	P = np.exp(-S/2.)

	# normalize to a probability density function
	Pint = np.sum(P)*(d[1]-d[0])*(Tm[1]-Tm[0])*(Td[1]-Td[0])
	P = P/Pint

	return d, Tm, Td, P


#Helper function for constructing parameter samples
def fit_mvn(parspace, dist):
	"""Finds the parameters of a multivariate normal distribution that best fits the data

	Parameters:
	-----------
	parspace : array-like
		list of meshgrid arrays spanning parameter space
	dist : array-like 
		PDF over parameter space
	Returns:
	--------
	mean : array-like
		distribution mean
	cov : array-like
		covariance matrix		
	"""

	# dimensionality of parameter space
	N = len(parspace)

	# flatten arrays
	parspace = [p.flatten() for p in parspace]
	dist = dist.flatten()

	# compute means
	mean = [np.sum(dist*par)/np.sum(dist) for par in parspace]

	# compute covariance matrix
	# empty matrix
	cov = np.zeros((N, N))
	# loop over rows
	for i in range(0, N):
		# loop over upper triangle
		for j in range(i, N):
			# compute covariance
			cov[i, j] = np.sum(dist*(parspace[i] - mean[i])
								* (parspace[j] - mean[j]))/np.sum(dist)
			# assign to lower triangle
			if i != j:
				cov[j, i] = cov[i, j]

	return np.array(mean), np.array(cov)


def construct_pressure_samples(a, b, c, P, N_samples):
	''' This function constructs samples from a multivariate normal distribution
		fitted to the pressure data.

	Parameters:
	-----------
	a : array-like
		Vector of 'a' parameter values.
	b : array-like
		Vector of 'b' parameter values.
	c : array-like
		Vector of 'c' parameter values.
	P : array-like
		Posterior probability distribution.
	N_samples : int
		Number of samples to take.

	Returns:
	--------
	samples : array-like
		parameter samples from the multivariate normal
	'''
	# compute properties (fitting) of multivariate normal distribution
	# mean = a vector of parameter means
	# covariance = a matrix of parameter variances and correlations
	A, B, C = np.meshgrid(a, b, c, indexing='ij')
	mean, covariance = fit_mvn([A,B,C], P)

	#create samples using numpy function multivariate_normal
	samples = np.random.multivariate_normal(mean, covariance, N_samples)

	return samples


def construct_subsidence_samples(d, Tm, Td, P, N_samples):
	''' This function constructs samples from a multivariate normal distribution
		fitted to the subsidence data.

		Parameters:
		-----------
		d : array-like
		Vector of 'd' parameter values.
		Tm : array-like
			Vector of 'Tm' parameter values.
		Td : array-like
			Vector of 'Td' parameter values.
		P : array-like
			Posterior probability distribution.
		N_samples : int
			Number of samples to take.

		Returns:
		--------
		samples : array-like
			parameter samples from the multivariate normal
	'''
	# compute properties (fitting) of multivariate normal distribution
	# mean = a vector of parameter means
	# covariance = a matrix of parameter variances and correlations
	A, B, C = np.meshgrid(d, Tm, Td, indexing='ij')
	mean, covariance = fit_mvn([A, B, C], P)

	#create samples using numpy function multivariate_normal
	samples = np.random.multivariate_normal(mean, covariance, N_samples)

	return samples


def pressure_model_ensemble(samples, display= False):
	'''Runs the pressure model for given parameter samples and plots the results.

		Parameters:
		-----------
		samples : array-like
			parameter samples from the multivariate normal
	'''
	t0 = 1953  # Start time
	t1 = 2013 #End time
	dt = 0.01 #Time step
	p0 = 56.26 #Ambient Pressure
	p_init= 56.26 #Initial Pressure

	t_range = np.arange(t0, t1 + dt, dt) #Time range

	#Load in mass extraction values interpolated at t_range times
	t_data, q_data = load_mass_exraction_data()
	q = interpolate_mass_extraction(t_data, q_data, t_range)

	#Find the derivative at each point numerically
	dqdt = np.gradient(q, dt)

	f,ax = plt.subplots(1,1)

	#for each sample, solve and plot the model 
	for a, b, c in samples:
		t, p_model = solve_pressure_ode(pressure_ode, t0, t1, dt, q, dqdt, p_init, p0, [a, b, c])
		ax.plot(t, p_model,'k-', lw=0.25,alpha=0.2)
	ax.plot([],[],'k-', lw=0.5,alpha=0.4, label='model ensemble')

	# get the data
	t_data, p_data = load_pressure_data()
	ax.axvline(2000, color='b', linestyle=':', label='calibration/forecast')

	v = 2.
	ax.errorbar(t_data, p_data ,yerr=v,fmt='ro', label='data')
	ax.set_xlabel('time')
	ax.set_ylabel('pressure [bars]')
	ax.legend()

	if display:
		plt.show()
	else:
		plt.savefig('pressure_model_ensemble', dpi=300)
		plt.close()


def subsidence_model_ensemble(samples, display= False):
	'''Runs the subsidence model for given parameter samples and plots the results.

		Parameters:
		-----------
		samples : array-like
			parameter samples from the multivariate normal
	'''
	t0 = 1953  # Start time
	t1 = 2013 #End time
	dt = 0.01 #Time step
	p0 = 56.26 #Ambient Pressure

	t_range = np.arange(t0, t1 + dt, dt) #Time range

	t_range = np.arange(t0, t1 + dt, dt)
	t_p_data, p_data = load_pressure_data()
	p_interp = interpolate_pressure_data(t_p_data, p_data, t_range)
	t_s_data, s_data = load_subsidence_data()
	s_interp = interpolate_subsidence_data(t_s_data, s_data, t_range)

	f,ax = plt.subplots(1,1)

	#for each sample, solve and plot the model
	for d, Tm, Td in samples:
		t, s_model = solve_subsidence_model(pressure_ode, t0, t1, dt, p_interp, p0, [d, Tm, Td])
		ax.plot(t, s_model, 'k-', lw=0.25, alpha=0.2)
	ax.plot([], [], 'k-', lw=0.5, alpha=0.4, label='model ensemble')

	# get the data
	t_data, s_data = load_subsidence_data()
	ax.axvline(2000, color='b', linestyle=':', label='calibration/forecast')

	#error variance - 0.5m
	v = 0.5
	ax.errorbar(t_data, s_data ,yerr=v,fmt='ro', label='data')
	ax.set_xlabel('time')
	ax.set_ylabel('subsidence [metres]')
	ax.legend()

	if display:
			plt.show()
	else:
		plt.savefig('subsidence_model_ensemble', dpi=300)
		plt.close()


if __name__ == "__main__":
	#Part 1 - Observation error plots
	observation_err_pressure(display)
	observation_err_subsidence(display)

	#Part 2 Posterior Plots
	a, b, c, P =pressure_grid_search()
	plot_pressure_posterior3D(a, b, c, P, display)
	d, Tm, Td, P = subsidence_grid_search()
	plot_subsidence_posterior3D(d, Tm, Td, P, display)

	#Sampled Parameters : Sample size = 100
	N= 100
	pressure_samples = construct_pressure_samples(a, b, c, P, N)
	plot_pressure_samples3D(a, b, c, P, pressure_samples, display)
	subsidence_samples = construct_subsidence_samples(d, Tm, Td, P, N)
	plot_subsidence_samples3D(d, Tm, Td, P, subsidence_samples, display)

	#Part 3 Model Ensemble
	pressure_model_ensemble(pressure_samples, display)
	subsidence_model_ensemble(subsidence_samples, display)



