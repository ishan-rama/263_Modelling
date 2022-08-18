def ode_model(t, p, q, a, b, p0, c, dqdt):
    ''' Return the derivative dx/dt at time, t, for given parameters.

        Parameters:
        -----------
        t : float
            Independent variable.
        p : float
            Dependent variable.
        q : float
            Source/sink rate.
        a : float
            Source/sink strength parameter.
        b : float
            Recharge strength parameter.
        p0 : float
            Ambient value of dependent variable.
        c :
            Slow-drainage parameter.
        dqdt :
            slow-drainage rate.

        Returns:
        --------
        dxdt : float
            Derivative of dependent variable with respect to independent variable.

        Notes:
        ------
        None

        Examples:
        ---------
        *add this*
        >>> ode_model()
        some number

    '''

    return -a * q - b * (p - p0) - c * dqdt