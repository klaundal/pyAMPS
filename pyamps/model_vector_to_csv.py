""" script to read model vector from npy (numpy save) file, and store coefficients
    in csv file, with wave number (n,m) along the rows, and external variables
    along the columns.

    This script is provided for transparency; when the csv file is made, it is
    no longer useful.
"""
import os.path
import numpy as np
import pandas as pd
from sh_utils import SHkeys

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

# merge them all and save to csv:
coefficients = pd.concat(dataframes, axis = 1)
coefficients.to_csv(os.path.abspath(os.path.join(basepath,'coefficients/model_coefficients.csv'), index_label = ('n', 'm')

# to read the file: pd.read_csv('coefficents/model_coefficients.csv', index_col=('n','m'))


