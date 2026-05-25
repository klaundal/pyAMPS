from __future__ import division

import os
import numpy as np
from numpy.testing import assert_allclose

import pyamps
from pyamps.coefficients import MODEL_VECTOR_TEST, MODEL_COEFF_0104


def test_get_model_vectors():
    model_vector = np.load(MODEL_VECTOR_TEST)

    path_txt = MODEL_COEFF_0104
    assert os.path.exists(path_txt)

    coeffs = pyamps.model_utils.get_coeffs(path_txt)
    vector_from_txt = []

    for param in pyamps.model_utils.names:
        vector_from_txt.extend(coeffs.loc[:, 'tor_c_' + param].dropna().values)
        vector_from_txt.extend(coeffs.loc[:, 'tor_s_' + param].dropna().values)
        vector_from_txt.extend(coeffs.loc[:, 'pol_c_' + param].dropna().values)
        vector_from_txt.extend(coeffs.loc[:, 'pol_s_' + param].dropna().values)

    assert_allclose(model_vector, np.array(vector_from_txt), atol=1e-6)
