from __future__ import division

import os
import pytest
import numpy as np

from numpy.testing import assert_array_equal,assert_allclose

import pyamps

@pytest.fixture(scope="function")
def switch_model():
    fake_csv_data = """ n,  m,  tor_c_const,    tor_s_const,    pol_c_const,    pol_s_const,    tor_c_sinca,    tor_s_sinca,    pol_c_sinca,    pol_s_sinca,    tor_c_cosca,    tor_s_cosca,    pol_c_cosca,    pol_s_cosca,    tor_c_epsilon,  tor_s_epsilon,  pol_c_epsilon,  pol_s_epsilon,  tor_c_epsilon_sinca,    tor_s_epsilon_sinca,    pol_c_epsilon_sinca,    pol_s_epsilon_sinca,    tor_c_epsilon_cosca,    tor_s_epsilon_cosca,    pol_c_epsilon_cosca,    pol_s_epsilon_cosca,    tor_c_tilt, tor_s_tilt, pol_c_tilt, pol_s_tilt, tor_c_tilt_sinca,   tor_s_tilt_sinca,   pol_c_tilt_sinca,   pol_s_tilt_sinca,   tor_c_tilt_cosca,   tor_s_tilt_cosca,   pol_c_tilt_cosca,   pol_s_tilt_cosca,   tor_c_tilt_epsilon, tor_s_tilt_epsilon, pol_c_tilt_epsilon, pol_s_tilt_epsilon, tor_c_tilt_epsilon_sinca,   tor_s_tilt_epsilon_sinca,   pol_c_tilt_epsilon_sinca,   pol_s_tilt_epsilon_sinca,   tor_c_tilt_epsilon_cosca,   tor_s_tilt_epsilon_cosca,   pol_c_tilt_epsilon_cosca,   pol_s_tilt_epsilon_cosca,   tor_c_tau,  tor_s_tau,  pol_c_tau,  pol_s_tau,  tor_c_tau_sinca,    tor_s_tau_sinca,    pol_c_tau_sinca,    pol_s_tau_sinca,    tor_c_tau_cosca,    tor_s_tau_cosca,    pol_c_tau_cosca,    pol_s_tau_cosca,    tor_c_tilt_tau, tor_s_tilt_tau, pol_c_tilt_tau, pol_s_tilt_tau, tor_c_tilt_tau_sinca,   tor_s_tilt_tau_sinca,   pol_c_tilt_tau_sinca,   pol_s_tilt_tau_sinca,   tor_c_tilt_tau_cosca,   tor_s_tilt_tau_cosca,   pol_c_tilt_tau_cosca,   pol_s_tilt_tau_cosca,   tor_c_f107, tor_s_f107, pol_c_f107, pol_s_f107
                        1,  0,  -0.1,           ,               -0.3,           0.4,            -0.5,           0.6,            -0.7,           0.8,            -0.9,           1.0,            -1.1,           1.2,            -1.3,           1.4,            -1.5,           1.6,            -1.7,                   1.8,                    -1.9,                   2.0,                    -2.1,                   2.2,                    -2.3,                   2.4,                    -2.5,       2.6,        -2.7,       2.8,        -2.9,               3.0,                -3.1,               3.2,                -3.3,               3.4,                -3.5,               3.6,                -3.7,               3.8,                -3.9,               4.0,                -4.1,                       4.2,                        -4.3,                       4.4,                        -4.5,                       4.6,                        -4.7,                       4.8,                        -4.9,       5.0,        -5.1,       5.2,        -5.3,               5.4,                -5.5,               5.6,                -5.7,               5.8,                -5.9,               6.0,                -6.1,           6.2,            -6.3,           6.4,            -6.5,                   6.6,                    -6.7,                   6.8,                    -6.9,                   7.0,                    -7.1,                   7.2,                    -7.3,       7.4,        -7.5,       7.6
                        1,  1,  -0.1,           ,               -0.3,           0.4,            -0.5,           0.6,            -0.7,           0.8,            -0.9,           1.0,            -1.1,           1.2,            -1.3,           1.4,            -1.5,           1.6,            -1.7,                   1.8,                    -1.9,                   2.0,                    -2.1,                   2.2,                    -2.3,                   2.4,                    -2.5,       2.6,        -2.7,       2.8,        -2.9,               3.0,                -3.1,               3.2,                -3.3,               3.4,                -3.5,               3.6,                -3.7,               3.8,                -3.9,               4.0,                -4.1,                       4.2,                        -4.3,                       4.4,                        -4.5,                       4.6,                        -4.7,                       4.8,                        -4.9,       5.0,        -5.1,       5.2,        -5.3,               5.4,                -5.5,               5.6,                -5.7,               5.8,                -5.9,               6.0,                -6.1,           6.2,            -6.3,           6.4,            -6.5,                   6.6,                    -6.7,                   6.8,                    -6.9,                   7.0,                    -7.1,                   7.2,                    -7.3,       7.4,        -7.5,       7.6
                        2,  0,  -0.1,           ,               -0.3,           0.4,            -0.5,           0.6,            -0.7,           0.8,            -0.9,           1.0,            -1.1,           1.2,            -1.3,           1.4,            -1.5,           1.6,            -1.7,                   1.8,                    -1.9,                   2.0,                    -2.1,                   2.2,                    -2.3,                   2.4,                    -2.5,       2.6,        -2.7,       2.8,        -2.9,               3.0,                -3.1,               3.2,                -3.3,               3.4,                -3.5,               3.6,                -3.7,               3.8,                -3.9,               4.0,                -4.1,                       4.2,                        -4.3,                       4.4,                        -4.5,                       4.6,                        -4.7,                       4.8,                        -4.9,       5.0,        -5.1,       5.2,        -5.3,               5.4,                -5.5,               5.6,                -5.7,               5.8,                -5.9,               6.0,                -6.1,           6.2,            -6.3,           6.4,            -6.5,                   6.6,                    -6.7,                   6.8,                    -6.9,                   7.0,                    -7.1,                   7.2,                    -7.3,       7.4,        -7.5,       7.6
                        2,  1,  -0.1,           ,               -0.3,           0.4,            -0.5,           0.6,            -0.7,           0.8,            -0.9,           1.0,            -1.1,           1.2,            -1.3,           1.4,            -1.5,           1.6,            -1.7,                   1.8,                    -1.9,                   2.0,                    -2.1,                   2.2,                    -2.3,                   2.4,                    -2.5,       2.6,        -2.7,       2.8,        -2.9,               3.0,                -3.1,               3.2,                -3.3,               3.4,                -3.5,               3.6,                -3.7,               3.8,                -3.9,               4.0,                -4.1,                       4.2,                        -4.3,                       4.4,                        -4.5,                       4.6,                        -4.7,                       4.8,                        -4.9,       5.0,        -5.1,       5.2,        -5.3,               5.4,                -5.5,               5.6,                -5.7,               5.8,                -5.9,               6.0,                -6.1,           6.2,            -6.3,           6.4,            -6.5,                   6.6,                    -6.7,                   6.8,                    -6.9,                   7.0,                    -7.1,                   7.2,                    -7.3,       7.4,        -7.5,       7.6
                        2,  2,  -0.1,           ,               -0.3,           0.4,            -0.5,           0.6,            -0.7,           0.8,            -0.9,           1.0,            -1.1,           1.2,            -1.3,           1.4,            -1.5,           1.6,            -1.7,                   1.8,                    -1.9,                   2.0,                    -2.1,                   2.2,                    -2.3,                   2.4,                    -2.5,       2.6,        -2.7,       2.8,        -2.9,               3.0,                -3.1,               3.2,                -3.3,               3.4,                -3.5,               3.6,                -3.7,               3.8,                -3.9,               4.0,                -4.1,                       4.2,                        -4.3,                       4.4,                        -4.5,                       4.6,                        -4.7,                       4.8,                        -4.9,       5.0,        -5.1,       5.2,        -5.3,               5.4,                -5.5,               5.6,                -5.7,               5.8,                -5.9,               6.0,                -6.1,           6.2,            -6.3,           6.4,            -6.5,                   6.6,                    -6.7,                   6.8,                    -6.9,                   7.0,                    -7.1,                   7.2,                    -7.3,       7.4,        -7.5,       7.6
    """.replace(" ","")
    true_name = pyamps.model_utils.coeff_fn
    fake_name = os.path.abspath("test_fake_model.csv")
    with open(fake_name,"w") as f:
        f.write(fake_csv_data)
    yield true_name,fake_name
    os.remove(fake_name) 
    pyamps.model_utils.coeff_fn = true_name


def test_fake_model(switch_model):
    #check that fake model has similar enough format to true model
    true_model,fake_model = switch_model
    with open(pyamps.model_utils.coeff_fn) as f:
        true_header = f.readline()
        true_data_line = f.readline()
    with open(fake_model) as f:
        fake_header = f.readline()
        fake_data_line = f.readline()
    
    assert true_header == fake_header
    assert len(true_data_line.split(",")) == len(fake_data_line.split(","))
    assert len(fake_data_line.split(",")) == len(fake_header.split(","))
    


def test_model_vectors(switch_model):
    true_model,fake_model = switch_model
    
    pyamps.model_utils.coeff_fn = fake_model

    model_vectors = pyamps.model_utils.get_model_vectors(v=0, By=0, Bz=1, tilt=0.5, f107=0.3)
    #const,cosca,tilt and f107 terms non-zero
    tor_c, tor_s, pol_c, pol_s, pol_c_index_v, tor_c_index_v = model_vectors

    assert_allclose(tor_c.flatten() ,[(-0.1 - 0.9 - 0.5*2.5 - 0.5*3.3 - 0.3*7.3)]*5)  
    assert_allclose(tor_s.flatten() ,[(                                       0)]*5)#tor_s_const = 0  
    assert_allclose(pol_c.flatten() ,[(-0.3 - 1.1 - 0.5*2.7 - 0.5*3.5 - 0.3*7.5)]*5)  
    assert_allclose(pol_s.flatten() ,[( 0.4 + 1.2 + 0.5*2.8 + 0.5*3.6 + 0.3*7.6)]*5)  
    assert tor_c_index_v[4] == (2,2)
    assert pol_c_index_v[2] == (2,0)
