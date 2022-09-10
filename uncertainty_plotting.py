#########################################################################################
# Function library for plotting uncertainty analysis 
#
# 	Functions:
#		plot_pressure_posterior3D: Plot posterior distribution for each parameter combination in pressure model.
#       plot_subsidence_posterior3D: Plot posterior distribution for each parameter combination in subsidence model.
#       plot_pressure_samples3D: Plot posterior distribution for each parameter combination with samples chosen.
#       plot_subsidence_samples3D: Plot posterior distribution for each parameter combination with samples chosen.
#########################################################################################


# import modules and functions
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from model_solver import *


def plot_pressure_posterior3D(a, b, c, P, display=False):
    """Plot posterior distribution for each parameter combination in pressure model.

    Args:
        a (numpy array): a distribution vector
        b (numpy array): b distribution vector
        c (numpy array): c distribution vector
        P (numpy array): posterior matrix
        display : boolean, optional
        boolean to either display(TRUE) or save plots(FALSE), by default False
    """

    # plotting variables
    azim = 15.		# azimuth at which surfaces are shown

    # a and b combination
    Ab, Ba = np.meshgrid(a, b, indexing='ij')
    Pab = np.zeros(Ab.shape)
    for i in range(len(a)):
        for j in range(len(b)):
            Pab[i][j] = sum([P[i][j][k] for k in range(len(c))])

    # a and c combination
    Ac, Ca = np.meshgrid(a, c, indexing='ij')
    Pac = np.zeros(Ac.shape)
    for i in range(len(a)):
        for k in range(len(c)):
            Pac[i][k] = sum([P[i][j][k] for j in range(len(b))])

    # b and c combination
    Bc, Cb = np.meshgrid(b, c, indexing='ij')
    Pbc = np.zeros(Bc.shape)
    for j in range(len(b)):
        for k in range(len(c)):
            Pbc[j][k] = sum([P[i][j][k] for i in range(len(a))])

    # plotting
    fig = plt.figure(figsize=[20.0, 15.])
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.plot_surface(Ab, Ba, Pab, rstride=1, cstride=1,
                     cmap=cm.Oranges, lw=0.5)
    ax1.set_xlabel('a')
    ax1.set_ylabel('b')
    ax1.set_zlabel('P')
    ax1.set_xlim([a[0], a[-1]])
    ax1.set_ylim([b[0], b[-1]])
    ax1.set_zlim(0., )
    ax1.view_init(40, azim)

    ax1 = fig.add_subplot(222, projection='3d')
    ax1.plot_surface(Ac, Ca, Pac, rstride=1, cstride=1,
                     cmap=cm.Oranges, lw=0.5)
    ax1.set_xlabel('a')
    ax1.set_ylabel('c')
    ax1.set_zlabel('P')
    ax1.set_xlim([a[0], a[-1]])
    ax1.set_ylim([c[0], c[-1]])
    ax1.set_zlim(0., )
    ax1.view_init(40, azim)

    ax1 = fig.add_subplot(223, projection='3d')
    ax1.plot_surface(Bc, Cb, Pbc, rstride=1, cstride=1,
                     cmap=cm.Oranges, lw=0.5)
    ax1.set_xlabel('b')
    ax1.set_ylabel('c')
    ax1.set_zlabel('P')
    ax1.set_xlim([b[0], b[-1]])
    ax1.set_ylim([c[0], c[-1]])
    ax1.set_zlim(0., )
    ax1.view_init(40, azim)

    if display:
        plt.show()
    else:
        plt.savefig('posterior_pressure', dpi=300)
        plt.close()


def plot_subsidence_posterior3D(d, Tm, Td, P, display= False):
    """Plot posterior distribution for each parameter combination in subsidence model.

    Args:
        d (numpy array): d distribution vector
        Tm (numpy array): Tm distribution vector
        Td (numpy array): Td distribution vector
        P (numpy array): posterior matrix
        display : boolean, optional
        boolean to either display(TRUE) or save plots(FALSE), by default False
    """

    # plotting variables
    azim = 15.		# azimuth at which surfaces are shown

    # d and Tm combination
    Ab, Ba = np.meshgrid(d, Tm, indexing='ij')
    Pab = np.zeros(Ab.shape)
    for i in range(len(d)):
        for j in range(len(Tm)):
            Pab[i][j] = sum([P[i][j][k] for k in range(len(Td))])

    # d and Td combination
    Ac, Ca = np.meshgrid(d, Td, indexing='ij')
    Pac = np.zeros(Ac.shape)
    for i in range(len(d)):
        for k in range(len(Td)):
            Pac[i][k] = sum([P[i][j][k] for j in range(len(Tm))])

    # Tm and Td combination
    Bc, Cb = np.meshgrid(Tm, Td, indexing='ij')
    Pbc = np.zeros(Bc.shape)
    for j in range(len(Tm)):
        for k in range(len(Td)):
            Pbc[j][k] = sum([P[i][j][k] for i in range(len(d))])

    # plotting
    fig = plt.figure(figsize=[20.0, 15.])
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.plot_surface(Ab, Ba, Pab, rstride=1, cstride=1,
                     cmap=cm.BuPu, lw=0.5)
    ax1.set_xlabel('d')
    ax1.set_ylabel('Tm')
    ax1.set_zlabel('P')
    ax1.set_xlim([d[0], d[-1]])
    ax1.set_ylim([Tm[0], Tm[-1]])
    ax1.set_zlim(0., )
    ax1.view_init(40, azim)

    ax1 = fig.add_subplot(222, projection='3d')
    ax1.plot_surface(Ac, Ca, Pac, rstride=1, cstride=1,
                     cmap=cm.BuPu, lw=0.5)
    ax1.set_xlabel('d')
    ax1.set_ylabel('Td')
    ax1.set_zlabel('P')
    ax1.set_xlim([d[0], d[-1]])
    ax1.set_ylim([Td[0], Td[-1]])
    ax1.set_zlim(0., )
    ax1.view_init(40, azim)

    ax1 = fig.add_subplot(223, projection='3d')
    ax1.plot_surface(Bc, Cb, Pbc, rstride=1, cstride=1,
                     cmap=cm.BuPu, lw=0.5)
    ax1.set_xlabel('Tm')
    ax1.set_ylabel('Td')
    ax1.set_zlabel('P')
    ax1.set_xlim([Tm[0], Tm[-1]])
    ax1.set_ylim([Td[0], Td[-1]])
    ax1.set_zlim(0., )
    ax1.view_init(40, azim)

    if display:
        plt.show()
    else:
        plt.savefig('posterior_subsidence', dpi=300)
        plt.close()


def plot_pressure_samples3D(a, b, c, P, samples, display= False):
    """Plot posterior distribution for each parameter combination with samples chosen.

    Parameters
    ----------
    a : array-like
			Vector of 'a' parameter values.
    b : array-like
        Vector of 'b' parameter values.
    c : array-like
        Vector of 'c' parameter values.
    P : array-like
        Posterior probability distribution.
    samples : array-like
			parameter samples from the multivariate normal
    display : boolean, optional
        boolean to either display(TRUE) or save plots(FALSE), by default False
    """
    # plotting variables
    azim = 15.		# azimuth at which surfaces are shown

    # a and b combination
    Ab, Ba = np.meshgrid(a, b, indexing='ij')
    Pab = np.zeros(Ab.shape)
    for i in range(len(a)):
        for j in range(len(b)):
            Pab[i][j] = sum([P[i][j][k] for k in range(len(c))])

    # a and c combination
    Ac, Ca = np.meshgrid(a, c, indexing='ij')
    Pac = np.zeros(Ac.shape)
    for i in range(len(a)):
        for k in range(len(c)):
            Pac[i][k] = sum([P[i][j][k] for j in range(len(b))])

    # b and c combination
    Bc, Cb = np.meshgrid(b, c, indexing='ij')
    Pbc = np.zeros(Bc.shape)
    for j in range(len(b)):
        for k in range(len(c)):
            Pbc[j][k] = sum([P[i][j][k] for i in range(len(a))])

    #Fixed parameters
    p_init, p0 = 56.26, 56.26
    # data for calibration from 1953 to 2000
    t0, t1, dt = 1953, 2000, 1
    t_range = np.arange(t0, t1 + dt, dt)
    t_p_data, p_data = load_pressure_data()
    p_interp = interpolate_pressure_data(t_p_data, p_data, t_range)

    #Load in mass extraction values interpolated at t_range times
    t_data, q_data = load_mass_exraction_data()
    q = interpolate_mass_extraction(t_data, q_data, t_range)
    #Find the derivative at each point numerically
    dqdt = np.gradient(q, dt)

    #error variance - 2bar
    v = 2.

    s= []
    for a_s, b_s, c_s in samples:
        t, p_model = solve_pressure_ode(pressure_ode, t0, t1, dt, q, dqdt, p_init, p0, [a_s, b_s, c_s])
        s.append(np.sum((p_interp - p_model)**2)/v)
    s= np.array(s)
    p = np.exp(-s/2.)
    p = p/np.max(p)*np.max(P)*1.2

    # plotting
    fig = plt.figure(figsize=[20.0, 15.])
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.plot_surface(Ab, Ba, Pab, rstride=1, cstride=1,
                     cmap=cm.Oranges, lw=0.5)
    ax1.set_xlabel('a')
    ax1.set_ylabel('b')
    ax1.set_zlabel('P')
    ax1.set_xlim([a[0], a[-1]])
    ax1.set_ylim([b[0], b[-1]])
    ax1.set_zlim(0., )
    ax1.view_init(40, azim)
    ax1.plot(samples[:, 0], samples[:, 1], p, 'k.')

    ax1 = fig.add_subplot(222, projection='3d')
    ax1.plot_surface(Ac, Ca, Pac, rstride=1, cstride=1,
                     cmap=cm.Oranges, lw=0.5)
    ax1.set_xlabel('a')
    ax1.set_ylabel('c')
    ax1.set_zlabel('P')
    ax1.set_xlim([a[0], a[-1]])
    ax1.set_ylim([c[0], c[-1]])
    ax1.set_zlim(0., )
    ax1.view_init(40, azim)
    ax1.plot(samples[:, 0], samples[:, -1], p, 'k.')

    ax1 = fig.add_subplot(223, projection='3d')
    ax1.plot_surface(Bc, Cb, Pbc, rstride=1, cstride=1,
                     cmap=cm.Oranges, lw=0.5)
    ax1.set_xlabel('b')
    ax1.set_ylabel('c')
    ax1.set_zlabel('P')
    ax1.set_xlim([b[0], b[-1]])
    ax1.set_ylim([c[0], c[-1]])
    ax1.set_zlim(0., )
    ax1.view_init(40, azim)
    ax1.plot(samples[:, 1], samples[:, -1], p, 'k.')

    if display:
        plt.show()
    else:
        plt.savefig('pressure_samples', dpi=300)
        plt.close()


def plot_subsidence_samples3D(d, Tm, Td, P, samples, display= False):
    """Plot posterior distribution for each parameter combination with samples chosen.

    Parameters
    ----------
    d : array-like
		Vector of 'd' parameter values.
    Tm : array-like
        Vector of 'Tm' parameter values.
    Td : array-like
        Vector of 'Td' parameter values.
    P : array-like
        Posterior probability distribution.
    samples : array-like
			parameter samples from the multivariate normal
    display : boolean, optional
        boolean to either display(TRUE) or save plots(FALSE), by default False
    """
    # plotting variables
    azim = 15.		# azimuth at which surfaces are shown

    # a and b combination
    Ab, Ba = np.meshgrid(d, Tm, indexing='ij')
    Pab = np.zeros(Ab.shape)
    for i in range(len(d)):
        for j in range(len(Tm)):
            Pab[i][j] = sum([P[i][j][k] for k in range(len(Td))])

    # a and c combination
    Ac, Ca = np.meshgrid(d, Td, indexing='ij')
    Pac = np.zeros(Ac.shape)
    for i in range(len(d)):
        for k in range(len(Td)):
            Pac[i][k] = sum([P[i][j][k] for j in range(len(Tm))])

    # b and c combination
    Bc, Cb = np.meshgrid(Tm, Td, indexing='ij')
    Pbc = np.zeros(Bc.shape)
    for j in range(len(Tm)):
        for k in range(len(Td)):
            Pbc[j][k] = sum([P[i][j][k] for i in range(len(d))])

    # data for calibration from 1953 to 2000
    t0, t1, dt = 1953, 2000, 1
    p0 = 56.26
    t_range = np.arange(t0, t1 + dt, dt)
    t_p_data, p_data = load_pressure_data()
    p_interp = interpolate_pressure_data(t_p_data, p_data, t_range)
    t_s_data, s_data = load_subsidence_data()
    s_interp = interpolate_subsidence_data(t_s_data, s_data, t_range)

    #error variance - 2bar
    v = 2.

    s = []
    for d_s, Tm_s, Td_s in samples:
        t, s_model = solve_subsidence_model(
            pressure_ode, t0, t1, dt, p_interp, p0, [d_s, Tm_s, Td_s])
        s.append(np.sum((s_interp - s_model)**2)/v)

    s = np.array(s)
    p = np.exp(-s/2.)
    p = p/np.max(p)*np.max(P)*1.2

    # plotting
    fig = plt.figure(figsize=[20.0, 15.])
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.plot_surface(Ab, Ba, Pab, rstride=1, cstride=1,
                        cmap=cm.BuPu, lw=0.5)
    ax1.set_xlabel('d')
    ax1.set_ylabel('Tm')
    ax1.set_zlabel('P')
    ax1.set_xlim([d[0], d[-1]])
    ax1.set_ylim([Tm[0], Tm[-1]])
    ax1.set_zlim(0., )
    ax1.view_init(40, azim)
    ax1.plot(samples[:, 0], samples[:, 1], p, 'k.')

    ax1 = fig.add_subplot(222, projection='3d')
    ax1.plot_surface(Ac, Ca, Pac, rstride=1, cstride=1,
                        cmap=cm.BuPu, lw=0.5)
    ax1.set_xlabel('d')
    ax1.set_ylabel('Td')
    ax1.set_zlabel('P')
    ax1.set_xlim([d[0], d[-1]])
    ax1.set_ylim([Td[0], Td[-1]])
    ax1.set_zlim(0., )
    ax1.view_init(40, azim)
    ax1.plot(samples[:, 0], samples[:, -1], p, 'k.')

    ax1 = fig.add_subplot(223, projection='3d')
    ax1.plot_surface(Bc, Cb, Pbc, rstride=1, cstride=1,
                        cmap=cm.BuPu, lw=0.5)
    ax1.set_xlabel('Tm')
    ax1.set_ylabel('Td')
    ax1.set_zlabel('P')
    ax1.set_xlim([Tm[0], Tm[-1]])
    ax1.set_ylim([Td[0], Td[-1]])
    ax1.set_zlim(0., )
    ax1.view_init(40, azim)
    ax1.plot(samples[:, 1], samples[:, -1], p, 'k.')

    if display:
        plt.show()
    else:
        plt.savefig('subsidence_samples', dpi=300)
        plt.close()
