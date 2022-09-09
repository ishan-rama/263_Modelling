#########################################################################################
# test_model_solver.py - Unit tests for model_solver.py library
#
# 	Tested Functions:
#		pressure_ode: Returns the derivative dp/dt for given parameters.
#		step_rk4: Performs one step of the Classic RK4 method.
#       subsidence_model: Computes subsidence at a given time point.
#########################################################################################

# import functions to be tested
from model_solver import *


def test_pressure_ode():
    # Normal Test case
    p, q, p0, a, b, c, dqdt = 1, 2, 3, 4, 5, 6, 0
    result1 = pressure_ode(p, q, p0, a, b, c, dqdt)

    # Normal Test case
    p, q, p0, a, b, c, dqdt = 3, 6, 7, 9, 10, 2, 11
    result2 = pressure_ode(p, q, p0, a, b, c, dqdt)

    # Edge Test case
    p, q, p0, a, b, c, dqdt = 0, 0, 0, 0, 0, 0, 0
    result3 = pressure_ode(p, q, p0, a, b, c, dqdt)

    assert result1 == 2
    assert result2 == -36
    assert result3 == 0


def test_step_rk4():
    f = pressure_ode
    yk = 10
    h = 5
    q, p0, a, b, c, dqdt1, dqdt2 = 1, 2, 3, 4, 5, 6, 2
    args = [q, p0, a, b, c, dqdt1, dqdt2]

    # Normal Test case
    result = step_rk4(f, yk, h, args)

    assert result == 90985


def test_subsidence_model():
    # Test division (0 as numerator)
    t, p_change, d, Tm, Td = 2000, 20, 0.5, 2000, 8
    result1 = subsidence_model(t, p_change, d, Tm, Td)

    # Test division (division by 0)
    t, p_change, d, Tm, Td = 1980, 50, 1, 2000, 0
    result2 = subsidence_model(t, p_change, d, Tm, Td)

    assert result1 == 5
    assert result2 == False
