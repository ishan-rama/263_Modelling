import numpy as np
from modelG15 import *
from plotting import *
import scipy.stats as st

def get_familiar_with_model():
    t_data, p_data = np.genfromtxt('sb_pres.txt', delimiter=',', skip_header=1)[:28,:].T
    t_s_data, s_data = np.genfromtxt('sb_disp.txt', delimiter=',', skip_header=1)[:28,:].T
    t0 = t_data[0]
    t1 = t_data[-1]
    p0 = 56.26
    dt = 1
    a = 5.1e-4
    b = 0.65e-3
    c = 0.008719645302839404
    pars = [a, b, c]

    d = 0.7441129057361028
    diffuse_t = 12.20957927092153
    t_max = 1984.073774683781

    t_range = np.arange(t0, t1 + dt, dt)

    q = interpolate_mass_extraction(t_range)
    dqdt = q.copy()
    for i in range(len(q)-1):
        dqdt[i] = (q[i+1] - q[i])/dt

    t, p = solve_pressure_ode(pressure_ode, t0, t1, dt, p0, q, dqdt, pars)
    s = subsidence_model(t, p, d, diffuse_t, t_max)

    v = 2.

    S = np.sum((p_data - p)**2) / v

    f,(ax,ax1) = plt.subplots(1,2)
    ax.plot(t,p,'b-', label='model')
    ax1.plot(t,s,'b-', label='model')
    ax.errorbar(t,p_data,yerr=v,fmt='ro', label='data')
    ax1.errorbar(t,s_data,yerr=v,fmt='ro', label='data')
    ax.set_xlabel('time')
    ax.set_ylabel('pressure')
    ax.set_title('objective function: S={:3.2f}'.format(S))
    ax.legend()
    plt.show()

def grid_search():
    ''' This function implements a grid search to compute the posterior over a and b.

        Returns:
        --------
        a : array-like
            Vector of 'a' parameter values.
        b : array-like
            Vector of 'b' parameter values.
        P : array-like
            Posterior probability distribution.
    '''
    # **to do**
    # 1. DEFINE parameter ranges for the grid search
    # 2. COMPUTE the sum-of-squares objective function for each parameter combination
    # 3. COMPUTE the posterior probability distribution
    # 4. ANSWER the questions in the lab document

    tp, po = np.genfromtxt('sb_pres.txt', delimiter=',', skip_header=1)[:28,:].T
    t0 = tp[0]
    t1 = tp[-1]
    p0 = 56.26
    dt = 1

    t_range = np.arange(t0, t1 + dt, dt)

    q = interpolate_mass_extraction(t_range)
    dqdt = q.copy()
    for i in range(len(q)-1):
        dqdt[i] = (q[i+1] - q[i])/dt

    # 1. define parameter ranges for the grid search
    a_best, b_best, c = plot_pressure_model()

    # number of values considered for each parameter within a given interval
    N = 51	

    # vectors of parameter values
    a = np.linspace(a_best/2,a_best*1.5, N)
    b = np.linspace(b_best/2,b_best*1.5, N)

    # grid of parameter values: returns every possible combination of parameters in a and b
    A, B = np.meshgrid(a, b, indexing='ij')

    # empty 2D matrix for objective function
    S = np.zeros(A.shape)

    # data for calibration

    # error variance - 2 bar
    v = 2.

    # grid search algorithm
    for i in range(len(a)):
        for j in range(len(b)):
            # 3. compute the sum of squares objective function at each value 
            #pm =
            _, pm = solve_pressure_ode(pressure_ode, t0, t1, dt, p0, q, dqdt, [a[i],b[j],c])
            S[i,j] = np.sum((po-pm)**2)/v

    # 4. compute the posterior
    #P=
    P = np.exp(-S/2.)

    # normalize to a probability density function
    Pint = np.sum(P)*(a[1]-a[0])*(b[1]-b[0])
    P = P/Pint

    # plot posterior parameter distribution
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

def model_ensemble(samples):
    ''' Runs the model for given parameter samples and plots the results.

        Parameters:
        -----------
        samples : array-like
            parameter samples from the multivariate normal
    '''
    p0 = 56.26
    c = 0.008719645302839404
    
    # **to do**
    # Run your parameter samples through the model and plot the predictions.
    d = 0.7441129057361028
    diffuse_t = 12.20957927092153
    t_max = 1984.073774683781

    # 1. choose a time vector to evaluate your model between 1953 and 2012 
    # t =
    t = np.linspace(1953, 2012, 101)
    t0 = t[0]
    t1 = t[-1]
    dt = t[1]-t[0]
    q = interpolate_mass_extraction(t)
    dqdt = q.copy()
    for i in range(len(q)-1):
        dqdt[i] = (q[i+1] - q[i])/dt

    # 2. create a figure and axes (see TASK 1)
    #f,ax =
    f,(ax,ax1) = plt.subplots(1,2)

    # 3. for each sample, solve and plot the model  (see TASK 1)
    for a,b in samples:
        #pm=
        #ax.plot(
        #*hint* use lw= and alpha= to set linewidth and transparency
        _, pm = solve_pressure_ode(pressure_ode, t0, t1-dt, dt, p0, q, dqdt, [a,b,c])
        s = subsidence_model(t, pm, d, diffuse_t, t_max)
        ax.plot(t,pm,'k-', lw=0.25,alpha=0.2)
        ax1.plot(t,s,'k-', lw=0.25,alpha=0.2)
    ax.plot([],[],'k-', lw=0.5,alpha=0.4, label='model ensemble')
    ax1.plot([],[],'k-', lw=0.5,alpha=0.4, label='model ensemble')

    # get the data
    tp,po = load_pressure_data()
    ts,so = load_subsidence_data()
    ax.axvline(1980, color='b', linestyle=':', label='calibration/forecast')
    ax1.axvline(1980, color='b', linestyle=':', label='calibration/forecast')

    # 4. plot Wairakei data as error bars
    # *hint* see TASK 1 for appropriate plotting commands
    v = 2.
    ax.errorbar(tp,po,yerr=v,fmt='ro', label='data')
    ax1.errorbar(ts,so,yerr=v,fmt='ro', label='data')
    ax.set_xlabel('time')
    ax1.set_xlabel('time')
    ax.set_ylabel('pressure')
    ax1.set_ylabel('subsidence')
    ax.legend()
    ax1.legend()
    plt.show()

def prediction_error(samples):
    p0 = 56.26
    c = 0.008719645302839404
    
    d = 0.7441129057361028
    diffuse_t = 12.20957927092153
    t_max = 1984.073774683781

    t = np.linspace(1953, 2012, 101)
    t0 = t[0]
    t1 = t[-1]
    dt = t[1]-t[0]
    q = interpolate_mass_extraction(t)
    dqdt = q.copy()
    for i in range(len(q)-1):
        dqdt[i] = (q[i+1] - q[i])/dt

    f,ax = plt.subplots(1,1)

    s = []

    for a,b in samples:
        t, pm = solve_pressure_ode(pressure_ode, t0, t1-dt, dt, p0, q, dqdt, [a,b,c])
        sub = subsidence_model(t, pm, d, diffuse_t, t_max)
        s.append(np.interp(1980,t,sub))

    t_data, s_data = load_subsidence_data()
    ax.hist(s)
    ax.axvline(np.interp(1980,t_data,s_data), color='b', label='true process')

    a = 0.0014699848529127948
    b = 0.06322104915197868
    c = 0.008719645302839404
    t,pm = solve_pressure_ode(pressure_ode, t0, t1-dt, dt, p0, q, dqdt, [a,b,c])

    ax.axvline(np.interp(1980,t,subsidence_model(t, pm, d, diffuse_t, t_max)), color='r', label='best model')

    ax.axvline(np.percentile(s, 5), color='k', linestyle=':', label='90% interval')
    ax.axvline(np.percentile(s, 90), color='k', linestyle=':')
    ax.set_xlabel('subsidence')
    ax.legend()
    plt.show()

if __name__=="__main__":
    #get_familiar_with_model()
    a,b,posterior = grid_search()
    N = 200
    samples = construct_samples(a, b, posterior, N)
    model_ensemble(samples)
    prediction_error(samples)