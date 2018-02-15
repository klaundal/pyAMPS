import numpy as np
import pandas as pd
import os
from functools import reduce

basepath = os.path.dirname(__file__)

coeff_fn = os.path.abspath(os.path.join(basepath,'coefficients','model_coefficients.txt'))

# read coefficient file and store in pandas DataFrame - with column names from last row of header:
names = ([x for x in open(coeff_fn).readlines() if x.startswith('#')][-1][1:]).strip().split(' ') 
coeffs = pd.read_table(coeff_fn, skipinitialspace = True, comment = '#', sep = ' ', names = names, index_col = [0, 1])



# organize the coefficients in arrays that are used to calculate magnetic field values
names = ['const', 'sinca', 'cosca', 'epsilon', 'epsilon_sinca', 'epsilon_cosca', 
         'tilt', 'tilt_sinca', 'tilt_cosca', 'tilt_epsilon', 'tilt_epsilon_sinca', 
         'tilt_epsilon_cosca', 'tau', 'tau_sinca', 'tau_cosca', 'tilt_tau', 
         'tilt_tau_sinca', 'tilt_tau_cosca', 'f107']

m_matrix = np.array([np.hstack((coeffs['tor_c_' + ss].dropna().values,
                                coeffs['tor_s_' + ss].dropna().values,
                                coeffs['pol_c_' + ss].dropna().values,
                                coeffs['pol_s_' + ss].dropna().values)) for ss in names]).T

m_matrix_pol = np.array([np.hstack((coeffs['pol_c_' + ss].dropna().values,
                                    coeffs['pol_s_' + ss].dropna().values)) for ss in names]).T


def get_model_vectors(v, By, Bz, tilt, f107, epsilon_multiplier = 1.):
    """ tor_c, tor_s, pol_c, pol_s = get_model_vectors(v, By, Bz, tilt, F107)

        returns column vectors ((K,1)-shaped) corresponding to the spherical harmonic coefficients of the toroidal
        and poloidal parts, with _c and _s denoting cos and sin terms, respectively.

        This function is used by amps.AMPS class
    """


    ca = np.arctan2(By, Bz)
    epsilon = v**(4/3.) * np.sqrt(By**2 + Bz**2)**(2/3.) * (np.sin(ca/2)**(8))**(1/3.) / 1000 * epsilon_multiplier # Newell coupling           
    tau     = v**(4/3.) * np.sqrt(By**2 + Bz**2)**(2/3.) * (np.cos(ca/2)**(8))**(1/3.) / 1000 # Newell coupling - inverse 

    # make a dict of the 19 external parameters, where the keys are postfixes in the column names of coeffs:
    external_params = {'const'             : 1                          ,                            
                       'sinca'             : 1              * np.sin(ca),
                       'cosca'             : 1              * np.cos(ca),
                       'epsilon'           : epsilon                    ,
                       'epsilon_sinca'     : epsilon        * np.sin(ca),
                       'epsilon_cosca'     : epsilon        * np.cos(ca),
                       'tilt'              : tilt                       ,
                       'tilt_sinca'        : tilt           * np.sin(ca),
                       'tilt_cosca'        : tilt           * np.cos(ca),
                       'tilt_epsilon'      : tilt * epsilon             ,
                       'tilt_epsilon_sinca': tilt * epsilon * np.sin(ca),
                       'tilt_epsilon_cosca': tilt * epsilon * np.cos(ca),
                       'tau'               : tau                        ,
                       'tau_sinca'         : tau            * np.sin(ca),
                       'tau_cosca'         : tau            * np.cos(ca),
                       'tilt_tau'          : tilt * tau                 ,
                       'tilt_tau_sinca'    : tilt * tau     * np.sin(ca),
                       'tilt_tau_cosca'    : tilt * tau     * np.cos(ca),
                       'f107'              : f107                        }

    # The SH coefficients are the sums in the expansion in terms of external parameters, scaled by the ext. params.:
    tor_c = reduce(lambda x, y: x+y, [coeffs['tor_c_' + param] * external_params[param] for param in external_params.keys()]).dropna()
    tor_s = reduce(lambda x, y: x+y, [coeffs['tor_s_' + param] * external_params[param] for param in external_params.keys()]).fillna(0)
    pol_c = reduce(lambda x, y: x+y, [coeffs['pol_c_' + param] * external_params[param] for param in external_params.keys()]).dropna()
    pol_s = reduce(lambda x, y: x+y, [coeffs['pol_s_' + param] * external_params[param] for param in external_params.keys()]).fillna(0)
    pol_s = pol_s.ix[pol_c.index] # equal number of sin and cos terms, but sin coeffs will be 0 where m = 0
    tor_s = tor_s.ix[tor_c.index] # 


    return tor_c[:, np.newaxis], tor_s[:, np.newaxis], pol_c[:, np.newaxis], pol_s[:, np.newaxis], pol_c.index.values, tor_c.index.values


