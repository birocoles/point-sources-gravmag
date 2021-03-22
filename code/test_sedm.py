import numpy as np
import scipy as sp
from numpy.testing import assert_almost_equal as aae
from pytest import raises
import sedm


def test_sedm_functions_bad_arguments():
    'code must stop for bad arguments'
    # list of function in sedm
    functions = [sedm.naive, sedm.avoid_sqrt, sedm.avoid_sqrt_inner_loops,
                 sedm.naive_numba, sedm.avoid_sqrt_numba]

    # wrong number of rows (it must be 3)
    P = np.empty((2, 5))
    S = np.empty((3, 4))

    for f in functions:
        raises(AssertionError, f, P, S)
    for f in functions:
        raises(AssertionError, f, S, P)

    # wrong ndim (it must be 2)
    P =np.empty(3)

    for f in functions:
        raises(AssertionError, f, P, S)
    for f in functions:
        raises(AssertionError, f, S, P)


def test_comparison_functions():
    'check if all functions produce the same result'
    np.random.seed(13)
    P = np.random.rand(3, 8)
    S = np.random.rand(3,11)
    
    EDM = []
    EDM.append(sedm.naive(P, S))
    EDM.append(sedm.avoid_sqrt(P, S))
    EDM.append(sedm.avoid_sqrt_inner_loops(P, S))
    EDM.append(sedm.scipy_distance(P, S))
    EDM.append(sedm.naive_numba(P, S))
    EDM.append(sedm.avoid_sqrt_numba(P, S))
    
    aae(EDM[0], EDM[1], decimal=10)
    aae(EDM[0], EDM[2], decimal=10)
    aae(EDM[0], EDM[3], decimal=10)
    aae(EDM[0], EDM[4], decimal=10)
    aae(EDM[0], EDM[5], decimal=10)
