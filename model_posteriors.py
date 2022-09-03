import numpy as np
from modelG15 import *
from plotting import *

def get_familiar_with_model():
    t0 = 1953
    t1 = 2012
    p0 = 56.26
    dt = 1
    a = 0.0014699848529127948
    b = 0.06322104915197868
    c = 0.008719645302839404
    pars = [a, b, c]

    t_range = np.arange(t0, t1 + dt, dt)

    q = interpolate_mass_extraction(t_range)
    dqdt = q.copy()
    for i in range(len(q)-1):
        dqdt[i] = (q[i+1] - q[i])/dt

    t_data, p_data = load_pressure_data()
    t, p = solve_pressure_ode(pressure_ode, t0, t1, dt, p0, q, dqdt, pars)

    v = 2.

    S = np.sum((p_data - p)**2) / v

    f,ax = plt.subplots(1,1)
    ax.plot(t_range,p,'b-', label='model')
    ax.errorbar(t_range,p_data,yerr=v,fmt='ro', label='data')
    ax.set_xlabel('time')
    ax.set_ylabel('pressure')
    ax.set_title('objective function: S={:3.2f}'.format(S))
    ax.legend()
    plt.show()

def grid_search():
    t0 = 1953
    t1 = 2012
    p0 = 56.26
    dt = 1
    a_best = 0.0014699848529127948
    b_best = 0.06322104915197868
    c = 0.008719645302839404
    d = 0.7441129057361028
    diffuse_t = 12.20957927092153
    t_max = 1984.073774683781

    t_range = np.arange(t0, t1 + dt, dt)

    q = interpolate_mass_extraction(t_range)
    dqdt = q.copy()
    for i in range(len(q)-1):
        dqdt[i] = (q[i+1] - q[i])/dt

    N = 51

    a = np.linspace(a_best/2,a_best*1.5, N)
    b = np.linspace(b_best/2,b_best*1.5, N)

    A, B = np.meshgrid(a, b, indexing='ij')

    S = np.zeros(A.shape)

    t_data, p_data = load_pressure_data()
    Time, Disp = load_subsidence_data()

    Disp = np.interp(t_range, Time, Disp)

    v = 2.

    for i in range(len(a)):
        for j in range(len(b)):

            t, p = solve_pressure_ode(pressure_ode, t0, t1, dt, p0, q, dqdt, [a[i], b[i], c])
            s = subsidence_model(t, p, d, diffuse_t, t_max)
            S[i,j] = np.sum((Disp - s)**2)/v

    P = np.exp(-S/2.)

    Pint = np.sum(P)*(a[1]-a[0])*(b[1]-b[0])
    P = P/Pint

    plot_posterior(a, b, P=P)

    return a,b,P

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
    cov = np.zeros((N,N))
        # loop over rows
    for i in range(0,N):
            # loop over upper triangle
        for j in range(i,N):
                # compute covariance
            cov[i,j] = np.sum(dist*(parspace[i] - mean[i])*(parspace[j] - mean[j]))/np.sum(dist)
                # assign to lower triangle
            if i != j: cov[j,i] = cov[i,j]
            
    return np.array(mean), np.array(cov)
        

def construct_samples(a,b,P,N_samples):
	''' This function constructs samples from a multivariate normal distribution
	    fitted to the data.

		Parameters:
		-----------
		a : array-like
			Vector of 'a' parameter values.
		b : array-like
			Vector of 'b' parameter values.
		P : array-like
			Posterior probability distribution.
		N_samples : int
			Number of samples to take.

		Returns:
		--------
		samples : array-like
			parameter samples from the multivariate normal
	'''
	# **to do**
	# 1. FIGURE OUT how to use the multivariate normal functionality in numpy
	#    to generate parameter samples
	# 2. ANSWER the questions in the lab document

	# compute properties (fitting) of multivariate normal distribution
	# mean = a vector of parameter means
	# covariance = a matrix of parameter variances and correlations
	A, B = np.meshgrid(a,b,indexing='ij')
	mean, covariance = fit_mvn([A,B], P)

	# 1. create samples using numpy function multivariate_normal (Google it)
	#samples=
	samples = np.random.multivariate_normal(mean, covariance, N_samples)

	# plot samples and predictions
	plot_samples(a, b, P=P, samples=samples)

	return samples

if __name__=="__main__":
    get_familiar_with_model()
    a,b,posterior = grid_search()
    N = 100
    samples = construct_samples(a, b, posterior, N)
