from pyamps.model_vector_to_txt import vector_to_df
import numpy as np
import time
import pandas as pd
external_parameters = ['const', 'sinca', 'cosca', 'epsilon', 'epsilon_sinca', 'epsilon_cosca', 'tilt',
                       'tilt_sinca', 'tilt_cosca', 'tilt_epsilon', 'tilt_epsilon_sinca', 'tilt_epsilon_cosca',
                       'tau', 'tau_sinca', 'tau_cosca', 'tilt_tau', 'tilt_tau_sinca', 'tilt_tau_cosca', 'f107']

# number of terms in the expansion of each spherical harmonic coefficient
NTERMS = len(external_parameters)

t = time.ctime().split(' ')
date = ' '.join([t[1], t[-1]])
header = '# Sherical harmonic coefficients for the Average Magnetic field and Polar current System (AMPS) model\n'
header = header + '# Produced ' + date
header = header + """
#
# Based on magnetic field measurements from CHAMP (2001-08 to 2010-09) and Swarm (2013-12 to 2023-12).
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
model_vector = np.load(
    '/Users/fasilkebede/Downloads/model_v1_iteration_13.npy')
dataframes = [vector_to_df(m) for m in np.split(model_vector, NTERMS)]

# add the external parameter to the column names:

for m, param in zip(dataframes, external_parameters):
    m.columns = [n + '_' + param for n in m.columns]
# merge them all
coefficients = pd.concat(dataframes, axis=1)

# write txt file
with open('/Users/fasilkebede/Downloads/SW_OPER_MIO_SHA_2E_00000000T000000_99999999T999999_0106.txt', 'w') as file:
    # header:
    file.write(header)
    # data:
    coefficients.to_string(buf=file, float_format=lambda x: '{:.7f}'.format(x),
                           header=False, sparsify=False,
                           index_names=False)
