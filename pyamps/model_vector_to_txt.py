""" script to read model vector from npy (numpy save) file, and store coefficients
    in rxt file, with wave number (n,m) along the rows, and external variables
    along the columns.

    This script is provided for transparency; when the txt file is made, it is
    no longer useful.
"""
from __future__ import absolute_import
import os.path
import time
import numpy as np
import pandas as pd
from .sh_utils import SHkeys

# header
t = time.ctime().split(' ')
date = ' '.join([t[1], t[-1]])
header = '# Sherical harmonic coefficients for the Average Magnetic field and Polar current System (AMPS) model\n'
header = header + '# Produced ' + date
header = header + """
#
# Based on magnetic field measurements from CHAMP (2001-08 to 2010-09) and Swarm (2013-12 to 2021-02).
# Reference: Laundal et al., "Solar wind and seasonal influence on ionospheric currents", Journal of Geophysical Research - Space Physics, doi:10.1029/2018JA025387, 2018
#
# Coefficient unit: nT
# Apex reference height: 110 km
# Earth radius: 6371.2 km
#
# Spherical harmonic degree, order: 65, 3 (for T) and 45, 3 (for V)
# 
# column names:
# n m tor_c_const tor_s_const pol_c_const pol_s_const tor_c_sinca tor_s_sinca pol_c_sinca pol_s_sinca tor_c_cosca tor_s_cosca pol_c_cosca pol_s_cosca tor_c_epsilon tor_s_epsilon pol_c_epsilon pol_s_epsilon tor_c_epsilon_sinca tor_s_epsilon_sinca pol_c_epsilon_sinca pol_s_epsilon_sinca tor_c_epsilon_cosca tor_s_epsilon_cosca pol_c_epsilon_cosca pol_s_epsilon_cosca tor_c_tilt tor_s_tilt pol_c_tilt pol_s_tilt tor_c_tilt_sinca tor_s_tilt_sinca pol_c_tilt_sinca pol_s_tilt_sinca tor_c_tilt_cosca tor_s_tilt_cosca pol_c_tilt_cosca pol_s_tilt_cosca tor_c_tilt_epsilon tor_s_tilt_epsilon pol_c_tilt_epsilon pol_s_tilt_epsilon tor_c_tilt_epsilon_sinca tor_s_tilt_epsilon_sinca pol_c_tilt_epsilon_sinca pol_s_tilt_epsilon_sinca tor_c_tilt_epsilon_cosca tor_s_tilt_epsilon_cosca pol_c_tilt_epsilon_cosca pol_s_tilt_epsilon_cosca tor_c_tau tor_s_tau pol_c_tau pol_s_tau tor_c_tau_sinca tor_s_tau_sinca pol_c_tau_sinca pol_s_tau_sinca tor_c_tau_cosca tor_s_tau_cosca pol_c_tau_cosca pol_s_tau_cosca tor_c_tilt_tau tor_s_tilt_tau pol_c_tilt_tau pol_s_tilt_tau tor_c_tilt_tau_sinca tor_s_tilt_tau_sinca pol_c_tilt_tau_sinca pol_s_tilt_tau_sinca tor_c_tilt_tau_cosca tor_s_tilt_tau_cosca pol_c_tilt_tau_cosca pol_s_tilt_tau_cosca tor_c_f107 tor_s_f107 pol_c_f107 pol_s_f107
"""


basepath = os.path.dirname(__file__)

# load model vector and define truncation levels and external parametrisation
model_vector = np.load(os.path.abspath(os.path.join(basepath,'coefficients/model_vector_NT_MT_NV_MV_65_3_45_3.npy')))
NT, MT, NV, MV = 65, 3, 45, 3

external_parameters = ['const', 'sinca', 'cosca', 'epsilon', 'epsilon_sinca', 'epsilon_cosca', 'tilt', 
                       'tilt_sinca', 'tilt_cosca', 'tilt_epsilon', 'tilt_epsilon_sinca', 'tilt_epsilon_cosca', 
                       'tau', 'tau_sinca', 'tau_cosca', 'tilt_tau', 'tilt_tau_sinca', 'tilt_tau_cosca', 'f107']

NTERMS = len(external_parameters) # number of terms in the expansion of each spherical harmonic coefficient


""" make spherical harmonic keys """
keys = {}
keys['cos_T'] = SHkeys(NT, MT).setNmin(1).MleN().Mge(0)
keys['sin_T'] = SHkeys(NT, MT).setNmin(1).MleN().Mge(1)
keys['cos_V'] = SHkeys(NV, MV).setNmin(1).MleN().Mge(0)
keys['sin_V'] = SHkeys(NV, MV).setNmin(1).MleN().Mge(1)

m_cos_V = keys['cos_V'].m
m_sin_V = keys['sin_V'].m
m_cos_T = keys['cos_T'].m
m_sin_T = keys['sin_T'].m


def vector_to_df(m_vec):
    """ convert model vector to dataframes, one for each coefficient (cos and sin for T and V). The index is (n,m) """

    tor_c = pd.Series(m_vec[                                           : m_cos_T.size                              ], index = keys['cos_T'], name = 'tor_c')
    tor_s = pd.Series(m_vec[m_cos_T.size                               : m_cos_T.size + m_sin_T.size               ], index = keys['sin_T'], name = 'tor_s')
    pol_c = pd.Series(m_vec[m_cos_T.size + m_sin_T.size                : m_cos_T.size + m_sin_T.size + m_cos_V.size], index = keys['cos_V'], name = 'pol_c')
    pol_s = pd.Series(m_vec[m_cos_T.size + m_sin_T.size + m_cos_V.size :                                           ], index = keys['sin_V'], name = 'pol_s')

    # merge the series into one DataFrame, and fill in zeros where the terms are undefined
    return pd.concat((tor_c, tor_s, pol_c, pol_s), axis = 1)

# get one set of coefficients per external parameter, and store in dataframes where columns are 'tor_c', 'tor_s', 'pol_c', and 'pol_s'
# _c and _s refer to cos and sin terms, respectively, and tor and pol to toroidal and poloidal
dataframes = [vector_to_df(m) for m in np.split(model_vector, NTERMS)]

# add the external parameter to the column names:
for m, param in zip(dataframes, external_parameters):
    m.columns = [n + '_' + param for n in m.columns]

# merge them all
coefficients = pd.concat(dataframes, axis = 1)

# write txt file
with open(os.path.abspath(os.path.join(basepath,'coefficients/SW_OPER_MIO_SHA_2E_00000000T000000_99999999T999999_0104.txt')), 'w') as file:
    # header:
    file.write(header)
    # data:
    coefficients.to_string(buf = file, float_format = lambda x: '{:.7f}'.format(x), 
                           header = False, sparsify = False,
                           index_names = False)

