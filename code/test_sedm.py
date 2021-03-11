import numpy as np
import scipy as sp
from numpy.testing import assert_almost_equal as aae
import pytest
import sedm

def test_scalar_vec_real_known_values():
    'check if all functions produce the same result'
    np.random.seed(13)
    P = np.random.rand(3, 8)
    S = np.random.rand(3,11)
    EDM = []
    EDM.append(sedm.naive(P, S))
    EDM.append(sedm.avoid_sqrt(P, S))
    EDM.append(sedm.avoid_sqrt_inner(P, S))
    EDM.append(sedm.avoid_sqrt_inner_loops(P, S))
    EDM.append(sedm.scipy_distance(P, S))
    EDM.append(sedm.avoid_sqrt_inner_jit(P, S))
    for edmi in EDM[1:]:
        aae(EDM[0], edmi, decimal=10)
