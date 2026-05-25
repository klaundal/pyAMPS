import numpy as np
import pandas as pd
from .coefficients import MODEL_COEFF_LATEST

default_coeff_fn = MODEL_COEFF_LATEST

# read coefficient file and store in pandas DataFrame - with column names from last row of header:
colnames = ([x for x in open(default_coeff_fn).readlines() if x.startswith('#')][-1][1:]).strip().split(' ') 

get_coeffs = lambda fn: pd.read_table(fn, skipinitialspace = True, comment = '#', sep = ' ', names = colnames, index_col = [0, 1])


# organize the coefficients in arrays that are used to calculate magnetic field values
names = ['const', 'sinca', 'cosca', 'epsilon', 'epsilon_sinca', 'epsilon_cosca', 
         'tilt', 'tilt_sinca', 'tilt_cosca', 'tilt_epsilon', 'tilt_epsilon_sinca', 
         'tilt_epsilon_cosca', 'tau', 'tau_sinca', 'tau_cosca', 'tilt_tau', 
         'tilt_tau_sinca', 'tilt_tau_cosca', 'f107']


def get_truncation_levels(coeff_fn = default_coeff_fn):
    """ read model truncation levels from coefficient file 
        returns NT, MT, NV, MV (spherical harmonic degree (N) and order (M)
        for toroidal (T) and poloidal (V) fields, respectively)
    """

    # read relevant line and split in words:
    words = [l for l in open(coeff_fn).readlines() if 'Spherical harmonic degree' in l][0].split(' ')

    # remove commas from each word
    words = [w.replace(',', '') for w in words]

    # pick out the truncation levels and convert to ints
    NT, MT, NV, MV = [int(num) for num in words if num.isdigit()]

    return NT, MT, NV, MV


def get_m_matrix(coeff_fn = default_coeff_fn):
    """ make matrix of model coefficients - used in get_B_space for fast calculations
        of model field time series along trajectory with changing input
    """
    coeffs = get_coeffs(coeff_fn)

    m_matrix = np.array([np.hstack((coeffs.loc[:, 'tor_c_' + ss].dropna().values,
                                    coeffs.loc[:, 'tor_s_' + ss].dropna().values,
                                    coeffs.loc[:, 'pol_c_' + ss].dropna().values,
                                    coeffs.loc[:, 'pol_s_' + ss].dropna().values)) for ss in names]).T
    return m_matrix


def get_m_matrix_pol(coeff_fn = default_coeff_fn):
    """ make matrix of model coefficients - only poloidal part
        used in get_B_ground for fast calculations of model field 
        time series on ground
    """
    coeffs = get_coeffs(coeff_fn)

    m_matrix_pol = np.array([np.hstack((coeffs.loc[:, 'pol_c_' + ss].dropna().values,
                                        coeffs.loc[:, 'pol_s_' + ss].dropna().values)) for ss in names]).T
    return m_matrix_pol



def get_model_vectors(v, By, Bz, tilt, f107, epsilon_multiplier = 1., coeff_fn = default_coeff_fn):
    """ tor_c, tor_s, pol_c, pol_s = get_model_vectors(v, By, Bz, tilt, F107, epsilon_multiplier = 1., coeffs = coeffs)

        Returns 2D arrays corresponding to the spherical harmonic coefficients of the
        toroidal and poloidal parts, with _c and _s denoting cos and sin terms,
        respectively. Scalar inputs produce column vectors with shape (K, 1).
        Vector-valued inputs produce arrays with shape (K, N), where N is the
        number of input values after broadcasting and flattening.

        This function is used by amps.AMPS class
    """

    coeffs = get_coeffs(coeff_fn)

    v, By, Bz, tilt, f107, epsilon_multiplier = np.broadcast_arrays(v, By, Bz, tilt, f107, epsilon_multiplier)
    v                  = v.flatten()
    By                 = By.flatten()
    Bz                 = Bz.flatten()
    tilt               = tilt.flatten()
    f107               = f107.flatten()
    epsilon_multiplier = epsilon_multiplier.flatten()

    ca = np.arctan2(By, Bz)
    epsilon = np.abs(v)**(4/3.) * np.sqrt(By**2 + Bz**2)**(2/3.) * (np.sin(ca/2)**(8))**(1/3.) / 1000 * epsilon_multiplier # Newell coupling           
    tau     = np.abs(v)**(4/3.) * np.sqrt(By**2 + Bz**2)**(2/3.) * (np.cos(ca/2)**(8))**(1/3.) / 1000 # Newell coupling - inverse 

    # make a dict of the 19 external parameters, where the keys are postfixes in the column names of coeffs:
    external_params = {'const'             : np.ones_like(ca)           ,                            
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

    external_params = np.vstack([external_params[param] for param in names])

    tor_c_index = coeffs.loc[:, 'tor_c_const'].dropna().index
    pol_c_index = coeffs.loc[:, 'pol_c_const'].dropna().index

    tor_c_coeffs = coeffs.loc[tor_c_index, ['tor_c_' + param for param in names]].to_numpy()
    tor_s_coeffs = coeffs.loc[tor_c_index, ['tor_s_' + param for param in names]].fillna(0).to_numpy()
    pol_c_coeffs = coeffs.loc[pol_c_index, ['pol_c_' + param for param in names]].to_numpy()
    pol_s_coeffs = coeffs.loc[pol_c_index, ['pol_s_' + param for param in names]].fillna(0).to_numpy()

    # The SH coefficients are the sums in the expansion in terms of external parameters.
    tor_c = tor_c_coeffs.dot(external_params)
    tor_s = tor_s_coeffs.dot(external_params)
    pol_c = pol_c_coeffs.dot(external_params)
    pol_s = pol_s_coeffs.dot(external_params)


    return tor_c, tor_s, pol_c, pol_s, pol_c_index.values, tor_c_index.values
