from __future__ import division

import pytest
import numpy as np

from numpy.testing import assert_array_equal,assert_allclose

import pyamps


@pytest.mark.xfail(reason="unfinished")
def test_model_vectors():
    #v, By, Bz, tilt, f107, epsilon_multiplier = 1.
    #
    tor_c, tor_s, pol_c, pol_s, pol_c_index_v, tor_c_index_v = pyamps.model_utils.get_model_vectors(v=0, By=0, Bz=1, tilt=0, f107=0)
    #print(tor_c,tor_s,pol_c,pol_s)
    assert False
    #_s is zero for n==m?

    pass