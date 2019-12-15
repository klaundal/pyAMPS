from __future__ import division

import os
import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
import pyamps

def test_model_vectors():

    fn = os.path.abspath(os.path.join(pyamps.model_utils.basepath, 'coefficients','test_model.txt'))

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
