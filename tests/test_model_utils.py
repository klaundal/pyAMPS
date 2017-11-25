from __future__ import division

import os
import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
import pyamps


def test_model_vectors(model_coeff):

    model_vectors = pyamps.model_utils.get_model_vectors(v=0, By=0, Bz=1, tilt=0.5, f107=0.3)
    # const,cosca,tilt and f107 terms non-zero
    tor_c, tor_s, pol_c, pol_s, pol_index, tor_index = model_vectors

    assert_allclose(tor_c.flatten(), [(-0.1 - 0.9 - 0.5 * 2.5 - 0.5 * 3.3 - 0.3 * 7.3)] * 5)
    assert_allclose(tor_s.flatten(), [(0)] * 5)  # tor_s_const = 0
    assert_allclose(pol_c.flatten(), [(-0.3 - 1.1 - 0.5 * 2.7 - 0.5 * 3.5 - 0.3 * 7.5)] * 5)
    assert_allclose(pol_s.flatten(), [(0.4 + 1.2 + 0.5 * 2.8 + 0.5 * 3.6 + 0.3 * 7.6)] * 5)
    assert tor_index[4] == (2, 2)
    assert pol_index[2] == (2, 0)
    assert len(tor_index) == 5
