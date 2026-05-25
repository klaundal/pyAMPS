from __future__ import division

import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
import pyamps
from pyamps.coefficients import MODEL_COEFF_TEST

def test_model_vectors():

    fn = MODEL_COEFF_TEST

    model_vectors = pyamps.model_utils.get_model_vectors(v=0, By=0, Bz=1, tilt=0.5, f107=0.3, coeff_fn = fn)
    # const,cosca,tilt and f107 terms non-zero
    tor_c, tor_s, pol_c, pol_s, pol_index, tor_index = model_vectors

    assert_allclose(tor_c.flatten(), [-0.186221  , -0.18158214, -0.14932639,  0.15199847, -0.30006201])
    assert_allclose(tor_s.flatten(), [ 0.        ,  1.21234145,  0.        , -0.64112046,  0.12969112])  # tor_s_const = 0
    assert_allclose(pol_c.flatten(), [ 0.87205983, -0.37122322, -0.07961519, -0.50883854,  0.09931672])
    assert_allclose(pol_s.flatten(), [ 0.        ,  0.36922202,  0.        ,  0.29350035, -0.22020463])
    assert tor_index[4] == (2, 2)
    assert pol_index[2] == (2, 0)
    assert len(tor_index) == 5


def test_model_vectors_vector_input():

    fn = MODEL_COEFF_TEST

    v    = np.array([0.0, 0.4, 1.1])
    By   = np.array([0.0, 100.6, -40.0])
    Bz   = np.array([1.0, 200.7, 30.0])
    tilt = np.array([0.5, 0.11, -0.2])
    f107 = np.array([0.3, 75.2, 120.0])

    vector_model = pyamps.model_utils.get_model_vectors(v = v, By = By, Bz = Bz, tilt = tilt, f107 = f107, coeff_fn = fn)

    for i in range(v.size):
        scalar_model = pyamps.model_utils.get_model_vectors(v = v[i], By = By[i], Bz = Bz[i],
                                                            tilt = tilt[i], f107 = f107[i], coeff_fn = fn)
        for scalar_values, vector_values in zip(scalar_model[:4], vector_model[:4]):
            assert_allclose(vector_values[:, i], scalar_values.flatten())

    assert vector_model[0].shape == (5, 3)
    assert vector_model[1].shape == (5, 3)
    assert vector_model[2].shape == (5, 3)
    assert vector_model[3].shape == (5, 3)
    assert_array_equal(vector_model[4], scalar_model[4])
    assert_array_equal(vector_model[5], scalar_model[5])


def test_model_vectors_broadcast_scalar_input():

    fn = MODEL_COEFF_TEST

    vector_model = pyamps.model_utils.get_model_vectors(v = np.array([0.0, 0.4]),
                                                        By = 0.0,
                                                        Bz = 1.0,
                                                        tilt = 0.5,
                                                        f107 = 0.3,
                                                        coeff_fn = fn)

    assert vector_model[0].shape == (5, 2)
