import numpy as np
from modelG15 import *

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

if __name__=="__main__":
    get_familiar_with_model()
