from __future__ import division

import os
import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

import pyamps


def test_get_model_vectors():
    basepath = os.path.join(os.path.dirname(__file__), "..", "pyamps")
    model_vector = np.load(os.path.abspath(os.path.join(basepath, 'coefficients/model_vector_NT_MT_NV_MV_65_3_45_3.npy')))

    path_txt = os.path.abspath(os.path.join(basepath, 'coefficients/model_coefficients.txt'))
    assert os.path.exists(path_txt)
    
    coeffs = np.nan_to_num(np.genfromtxt(path_txt, skip_header=14, unpack=True))
    model_vectors = np.split(model_vector, 19)
    
    assert_allclose(model_vectors[0][:len(coeffs[0])], coeffs[2], atol=1e-6)
    assert_allclose(model_vector.sum(), coeffs[2:].sum(), atol=1e-5)

    pass
