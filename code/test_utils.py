import numpy as np
from numpy.testing import assert_almost_equal as aae
import pytest
import utils


def test_unit_vector_magnitude():
    'check if the unit vector has magnitude 1'
    I = [10, -30, 0, 90, -90, 180, 45, 73, -3]
    D = [-28, 47, 5, 18, 0, 90, -90, 7, 89]
    for inc, dec in zip(I, D):
        u = utils.unit_vector(inc, dec)
        aae(np.sum(u*u), 1, decimal=10)


def test_unit_vector_known_values():
    'compare computed unit vector with reference values'
    I = [0, 0, 90, -90, 0]
    D = [0, 90, 0, 0, 45]
    reference_outputs = [np.array([1, 0, 0]),
                         np.array([0, 1, 0]),
                         np.array([0, 0, 1]),
                         np.array([0, 0, -1]),
                         np.array([np.sqrt(2)/2, np.sqrt(2)/2, 0])]
    for inc, dec, ref in zip(I, D, reference_outputs):
        u = utils.unit_vector(inc, dec)
        aae(u, ref, decimal=10)


def test_direction_known_values():
    'compare computed direction with reference values'
    reference_I = [0, 0, 90, -90, 0]
    reference_D = [0, 90, 0, 0, 45]
    reference_inputs = [np.array([1, 0, 0]),
                        np.array([0, 1, 0]),
                        np.array([0, 0, 1]),
                        np.array([0, 0, -1]),
                        np.array([np.sqrt(2)/2, np.sqrt(2)/2, 0])]
    for ref_inc, ref_dec, ref_input in zip(reference_I, reference_D, 
                                           reference_inputs):
        intens, inc, dec = utils.direction(ref_input)
        aae(intens, 1, decimal=10)
        aae(inc, ref_inc, decimal=10)
        aae(dec, ref_dec, decimal=10)


def test_rotation_matrix_orthonormal():
    'check if the rotation matrix is orthonormal'
    I = [10, -30, 0, 90, -90, 180, 45, 73, -3]
    D = [-28, 47, 5, 18, 0, 90, -90, 7, 89]
    dI = [1, 18, 24, 13, 0, 40, 5, -3, -3]
    dD = [8, 7, -51, 108, 19.4, 0, 6, -7, 389]
    for inc, dec, dinc, ddec in zip(I, D, dI, dD):
        R = utils.rotation_matrix(inc, dec, dinc, ddec)
        aae(np.dot(R.T, R), np.identity(3), decimal=10)
        aae(np.dot(R, R.T), np.identity(3), decimal=10)