from __future__ import division

import pytest
import datetime
import numpy as np
from numpy import pi, cos, sin
from numpy.testing import assert_array_equal, assert_allclose

import pyamps
from pyamps.amps import AMPS, get_B_space, get_B_ground
from pyamps import model_utils
from pyamps.coefficients import MODEL_COEFF_TEST, MODEL_COEFF_0103

test_coeff_fn = MODEL_COEFF_TEST

@pytest.fixture()
def amps_model(model_coeff):
    model_args = [
        0.4,    # v
        100.6,  # By
        200.7,  # Bz
        0.11,   # tilt
        75.2    # F107
    ]
    model_kwargs = dict(
        minlat=71.2,
        maxlat=85.1,
        height=90.1,
        dr=4,
        M0=8,
        resolution=21,
        coeff_fn = test_coeff_fn
    )
    try:
        model = AMPS(*model_args, **model_kwargs)
    except Exception:
        # allow test_init to fail instead
        model = None
    return model, model_args, model_kwargs


class Test_AMPS(object):

    def test_init(self, amps_model):
        model, m_args, m_kwargs = amps_model

        model = AMPS(*m_args, **m_kwargs)
        model_vectors = pyamps.model_utils.get_model_vectors(*m_args, coeff_fn = m_kwargs['coeff_fn'])

        assert_allclose(model.tor_s, model_vectors[1])
        assert_allclose(model.pol_c, model_vectors[2])
        assert_array_equal(model.tor_keys, model_vectors[5])

        assert model.N == 2
        assert model.M == 2

        assert model.plotgrid_scalar[0].shape == (m_kwargs['resolution'], m_kwargs['resolution'])

    def test_update_model(self, amps_model):
        model, m_args, m_kwargs = amps_model

        old_tor_c = model.tor_c.copy()
        m_args[0] += 1
        model.update_model(*m_args)
        new_tor_c = model.tor_c
        with pytest.raises(AssertionError):
            assert_allclose(old_tor_c, new_tor_c, atol=1e-5)

    def test_vector_model_inputs_match_scalar_models(self):
        vector_args = [np.array([  0.4,   1.4]),
                       np.array([100.6, -20.5]),
                       np.array([200.7,  50.2]),
                       np.array([  0.11, -5.0]),
                       np.array([ 75.2, 120.0])]

        model_kwargs = dict(
            minlat     = 71.2,
            maxlat     = 85.1,
            height     = 90.1,
            dr         = 4,
            M0         = 8,
            resolution = 7,
            coeff_fn   = test_coeff_fn
        )

        model         = AMPS(*vector_args, **model_kwargs)
        scalar_models = [AMPS(*[arg[i] for arg in vector_args], **model_kwargs) for i in range(2)]

        def assert_model_columns_match_scalar_outputs(vector_values, scalar_values, shape = None):
            vector_values = np.asarray(vector_values)
            if shape is not None:
                assert vector_values.shape == shape + (2,)
            else:
                assert vector_values.shape[-1] == 2

            for i, scalar_value in enumerate(scalar_values):
                assert_allclose(vector_values[..., i], np.asarray(scalar_value))

        scalar_grid_methods = ['get_toroidal_scalar',
                               'get_poloidal_scalar',
                               'get_divergence_free_current_function',
                               'get_upward_current',
                               'get_curl_free_current_potential',
                               'get_total_current_magnitude',
                               'get_ground_Beqd',
                               'get_ground_Bnqd',
                               'get_ground_Buqd']

        for method_name in scalar_grid_methods:
            vector_values = getattr(model, method_name)()
            scalar_values = [getattr(scalar_model, method_name)().reshape(-1) for scalar_model in scalar_models]
            assert_model_columns_match_scalar_outputs(vector_values, scalar_values,
                                                      shape = (2 * model_kwargs['resolution']**2,))

        mlat = np.array([[72., 74.],
                         [78., 82.]])
        mlt  = np.array([[ 1.,  6.],
                         [12., 18.]])

        custom_scalar_methods = ['get_toroidal_scalar',
                                 'get_poloidal_scalar',
                                 'get_divergence_free_current_function',
                                 'get_upward_current',
                                 'get_curl_free_current_potential']

        for method_name in custom_scalar_methods:
            vector_values = getattr(model, method_name)(mlat, mlt)
            scalar_values = [getattr(scalar_model, method_name)(mlat, mlt) for scalar_model in scalar_models]
            assert_model_columns_match_scalar_outputs(vector_values, scalar_values, shape = mlat.shape)

        grid_mlat = np.array([72., 76., 82.])
        grid_mlt  = np.array([ 1.,  6., 12., 18.])

        for method_name in custom_scalar_methods:
            vector_values = getattr(model, method_name)(grid_mlat, grid_mlt, grid = True)
            scalar_values = [getattr(scalar_model, method_name)(grid_mlat, grid_mlt, grid = True) for scalar_model in scalar_models]
            assert_model_columns_match_scalar_outputs(vector_values, scalar_values,
                                                      shape = (grid_mlat.size, grid_mlt.size))

        vector_current_methods = ['get_divergence_free_current',
                                  'get_curl_free_current',
                                  'get_total_current',
                                  'get_ground_perturbation']

        for method_name in vector_current_methods:
            vector_components = getattr(model, method_name)()
            scalar_components = [getattr(scalar_model, method_name)() for scalar_model in scalar_models]

            for component_index, vector_values in enumerate(vector_components):
                scalar_values = [scalar_components[i][component_index].reshape(-1) for i in range(2)]
                assert_model_columns_match_scalar_outputs(vector_values, scalar_values, shape = (model.vectorgrid[0].size,))

        for method_name in vector_current_methods[:-1]:
            vector_components = getattr(model, method_name)(mlat, mlt)
            scalar_components = [getattr(scalar_model, method_name)(mlat, mlt) for scalar_model in scalar_models]

            for component_index, vector_values in enumerate(vector_components):
                scalar_values = [scalar_components[i][component_index] for i in range(2)]
                assert_model_columns_match_scalar_outputs(vector_values, scalar_values, shape = mlat.shape)

        for method_name in vector_current_methods[:-1]:
            vector_components = getattr(model, method_name)(grid_mlat, grid_mlt, grid = True)
            scalar_components = [getattr(scalar_model, method_name)(grid_mlat, grid_mlt, grid = True) for scalar_model in scalar_models]

            for component_index, vector_values in enumerate(vector_components):
                scalar_values = [scalar_components[i][component_index] for i in range(2)]
                assert_model_columns_match_scalar_outputs(vector_values, scalar_values,
                                                          shape = (grid_mlat.size, grid_mlt.size))

        for summary_method in ['get_integrated_upward_current', 'get_AE_indices']:
            vector_values = getattr(model, summary_method)()
            scalar_values = [getattr(scalar_model, summary_method)() for scalar_model in scalar_models]

            for component_index, values in enumerate(vector_values):
                assert values.shape == (2,)
                assert_allclose(values, [scalar_values[0][component_index], scalar_values[1][component_index]])

        with pytest.raises(ValueError, match = 'plot_currents'):
            model.plot_currents()

        with pytest.raises(ValueError, match = 'plot_deltaB_space_dipole'):
            model.plot_deltaB_space_dipole()

    def test__get_vectorgrid(self, amps_model):
        model, _, _ = amps_model

        mlat, mlt = model._get_vectorgrid()

        mlat_, mlt_, mlt_res = pyamps.plot_utils.equal_area_grid(dr=model.dr, M0=model.M0)
        assert (np.abs(mlat) >= model.minlat).all()
        assert (-model.maxlat <= mlat).all() and (mlat <= model.maxlat).all()
        assert (0 <= mlt).all() and (mlt <= 24).all()
        assert mlat.shape == mlt.shape

    def test__get_scalargrid(self, amps_model):
        model, _, m_kwargs = amps_model

        resolution = m_kwargs['resolution'] + 1
        mlat, mlt = model._get_scalargrid(resolution)

        assert model.scalar_resolution == resolution
        assert mlat.shape == mlt.shape
        assert mlat.shape == (2 * resolution**2, 1)

        assert (np.abs(mlat) >= model.minlat).all()
        assert (-model.maxlat <= mlat).all() and (mlat <= model.maxlat).all()
        assert (0 <= mlt).all() and (mlt <= 24).all()

    def test_calculate_matrices(self, amps_model):
        model, _, m_kwargs = amps_model
        assert model.tor_sinmphi_scalar.shape == (882, 5)
        assert model.pol_dP_vector.shape == (144, 5)

        assert_allclose(model.tor_sinmphi_vector[2],
                        [0., 0.47139674, 0., 0.47139674, 0.83146961],atol=1e-6)
        assert_allclose(model.pol_cosmphi_vector[0],
                        [1., 0.99518473, 1., 0.99518473, 0.98078528], atol=1e-6)

        assert_allclose(model.pol_P_scalar[1],
                        [0.95048862, 0.31075938, 0.85514291, 0.51160148, 0.08363328], atol=1e-6)

        assert_allclose(model.tor_dP_vector[5],
                        [ 0.27563736, -0.9612617 ,  0.7948789 , -1.46886239, -0.45892354], atol=1e-6)

    @pytest.mark.parametrize("mlat, mlt", [(np.array([60.]), np.array([0.])),
                                           (np.array([71.]), np.array([6.]))])
    def test_toroidal_scalar(self, amps_model, mlat, mlt):
        model, _, m_kwargs = amps_model
        mlt2r = np.pi / 12

        P, dP = pyamps.sh_utils.legendre(model.N, model.M, 90 - mlat)

        T = 0
        for i, (n, m) in enumerate(model.keys_T):
            T += P[n, m] * (model.tor_c[i] * cos(m * mlt * mlt2r) +
                            model.tor_s[i] * sin(m * mlt * mlt2r))

        assert_allclose(T.reshape(mlat.shape), model.get_toroidal_scalar(mlat, mlt))

        assert_allclose(np.split(model.get_toroidal_scalar(), 2)[0].reshape(model.plotgrid_scalar[0].shape),
                        model.get_toroidal_scalar(*model.plotgrid_scalar))
        pass

    @pytest.mark.parametrize("mlat, mlt", [(np.array([60.]), np.array([0.])),
                                           (np.array([71.]), np.array([6.]))])
    def test_poloidal_scalar(self, amps_model, mlat, mlt):
        model, _, m_kwargs = amps_model
        mlt2r = pi / 12
        REFRE = 6371.2

        P, dP = pyamps.sh_utils.legendre(model.N, model.M, 90 - mlat)

        V = 0
        for i, (n, m) in enumerate(model.keys_P):
            V += (REFRE / (REFRE + m_kwargs['height']))**(n + 1) * P[n, m] * (
                model.pol_c[i] * cos(m * mlt * mlt2r) +
                model.pol_s[i] * sin(m * mlt * mlt2r))
        V *= REFRE
        assert_allclose(V.reshape(mlat.shape), model.get_poloidal_scalar(mlat, mlt))

        assert_allclose(np.split(model.get_poloidal_scalar(), 2)[0].reshape(model.plotgrid_scalar[0].shape),
                        model.get_poloidal_scalar(*model.plotgrid_scalar))
        pass

    @pytest.mark.parametrize("mlat, mlt", [(np.array([[60.]]), np.array([[0.]])),
                                           (np.array([[71.]]), np.array([[6.]]))])
    def test_get_divergence_free_current_function(self, amps_model, mlat, mlt):
        model, _, m_kwargs = amps_model
        mlt2r = pi / 12
        REFRE = 6371.2
        MU0 = pi * 4e-7

        P, dP = pyamps.sh_utils.legendre(model.N, model.M, 90 - mlat)

        Psi = 0
        for i, (n, m) in enumerate(model.keys_P):
            Psi += (REFRE / (REFRE + m_kwargs['height']))**(n + 1) \
                * (2 * n + 1) / n * P[n, m] * (
                    model.pol_c[i] * cos(m * mlt * mlt2r) +
                    model.pol_s[i] * sin(m * mlt * mlt2r))
        Psi *= -REFRE / MU0 * 1e-9
        mlats, mlts = np.meshgrid(mlat,mlt)
        assert_allclose(Psi, model.get_divergence_free_current_function(mlat, mlt))
        assert_allclose(
            model.get_divergence_free_current_function(mlats, mlts),
            model.get_divergence_free_current_function(mlat.flatten(), mlt.flatten(), True)
        )

        assert_allclose(np.split(model.get_divergence_free_current_function(), 2)[0].reshape(
                            m_kwargs['resolution'],m_kwargs['resolution']),
                        model.get_divergence_free_current_function(*model.plotgrid_scalar))
        pass

    @pytest.mark.parametrize("mlat, mlt", [(np.array([[60.]]), np.array([[0.]])),
                                           (np.array([[71.]]), np.array([[6.]]))])
    def test_get_upward_current(self, amps_model, mlat, mlt):
        model, _, m_kwargs = amps_model
        mlt2r = pi / 12
        REFRE = 6371.2
        MU0 = pi * 4e-7

        P, dP = pyamps.sh_utils.legendre(model.N, model.M, 90 - mlat)

        Ju = 0
        for i, (n, m) in enumerate(model.keys_T):
            Ju += n * (n + 1) * P[n, m] * (
                model.tor_c[i] * cos(m * mlt * mlt2r) +
                model.tor_s[i] * sin(m * mlt * mlt2r))
        Ju *= -1 / MU0 / (REFRE + m_kwargs['height']) * 1e-6
        mlats, mlts = np.meshgrid(mlat,mlt)
        assert_allclose(Ju, model.get_upward_current(mlat, mlt))
        assert_allclose(
            model.get_upward_current(mlats, mlts),
            model.get_upward_current(mlat.flatten(), mlt.flatten(), True)
        )

        assert_allclose(np.split(model.get_upward_current(), 2)[0].reshape(
                            m_kwargs['resolution'],m_kwargs['resolution']),
                        model.get_upward_current(*model.plotgrid_scalar))
        pass

    @pytest.mark.parametrize("mlat, mlt", [(np.array([[60.]]), np.array([[0.]])),
                                           (np.array([[71.]]), np.array([[6.]]))])
    def test_get_curl_free_current_potential(self, amps_model, mlat, mlt):
        model, _, m_kwargs = amps_model
        mlt2r = pi / 12
        REFRE = 6371.2
        MU0 = pi * 4e-7

        P, dP = pyamps.sh_utils.legendre(model.N, model.M, 90 - mlat)

        alpha = 0
        for i, (n, m) in enumerate(model.keys_T):
            alpha += P[n, m] * (
                model.tor_c[i] * cos(m * mlt * mlt2r) +
                model.tor_s[i] * sin(m * mlt * mlt2r))
        alpha *= -(REFRE + m_kwargs['height']) / MU0 * 1e-9
        mlats, mlts = np.meshgrid(mlat,mlt)
        assert_allclose(alpha, model.get_curl_free_current_potential(mlat, mlt))
        assert_allclose(
            model.get_curl_free_current_potential(mlats, mlts),
            model.get_curl_free_current_potential(mlat.flatten(), mlt.flatten(), True)
        )
        assert_allclose(np.split(model.get_curl_free_current_potential(), 2)[0].reshape(
                            m_kwargs['resolution'],m_kwargs['resolution']),
                        model.get_curl_free_current_potential(*model.plotgrid_scalar))
        pass

    @pytest.mark.parametrize("mlat, mlt", [(np.array([[60.]]), np.array([[0.]])),
                                           (np.array([[71.]]), np.array([[6.]]))])
    def test_get_divergence_free_current(self, amps_model, mlat, mlt):
        model, _, m_kwargs = amps_model
        mlt2r = pi / 12
        lat2r = pi / 180
        REFRE = 6371.2
        MU0 = pi * 4e-7

        P, dP = pyamps.sh_utils.legendre(model.N, model.M, 90 - mlat)

        east = 0
        north = 0
        for i, (n, m) in enumerate(model.keys_P):
            east += (REFRE / (REFRE + m_kwargs['height'])) ** (n + 2) \
                * (2 * n + 1) / n * (-dP[n, m]) * (
                    model.pol_c[i] * cos(m * mlt * mlt2r) +
                    model.pol_s[i] * sin(m * mlt * mlt2r))
            north += - (REFRE / (REFRE + m_kwargs['height'])) ** (n + 2) \
                * (2 * n + 1) / n * P[n, m] * (
                    m * model.pol_s[i] * cos(m * mlt * mlt2r) -
                    m * model.pol_c[i] * sin(m * mlt * mlt2r)) / cos(mlat * lat2r)

        east *= 1 / MU0 * 1e-6
        north *= 1 / MU0 * 1e-6

        out = np.array([east, north]).reshape(-1, 1, 1)
        mlats, mlts = np.meshgrid(mlat,mlt)
        assert_allclose(out, model.get_divergence_free_current(mlat, mlt))
        assert_allclose(
            np.array(model.get_divergence_free_current(mlats, mlts)).flatten(),
            np.array(model.get_divergence_free_current(mlat.flatten(), mlt.flatten(), True)).flatten()
        )
        
        assert_allclose(np.split(model.get_divergence_free_current()[0], 2)[0].reshape(model.plotgrid_vector[0].shape),
                        model.get_divergence_free_current(*model.plotgrid_vector)[0])
        assert_allclose(np.split(model.get_divergence_free_current()[1], 2)[0].reshape(model.plotgrid_vector[0].shape),
                        model.get_divergence_free_current(*model.plotgrid_vector)[1])

        pass

    @pytest.mark.parametrize("mlat, mlt", [(np.array([[60.]]), np.array([[0.]])),
                                           (np.array([[71.]]), np.array([[6.]]))])
    def test_get_curl_free_current(self, amps_model, mlat, mlt):
        model, _, m_kwargs = amps_model
        mlt2r = pi / 12
        lat2r = pi / 180
        REFRE = 6371.2
        MU0 = pi * 4e-7

        P, dP = pyamps.sh_utils.legendre(model.N, model.M, 90 - mlat)

        east = 0
        north = 0
        for i, (n, m) in enumerate(model.keys_T):
            east += P[n, m] * (
                m * model.tor_s[i] * cos(m * mlt * mlt2r) -
                m * model.tor_c[i] * sin(m * mlt * mlt2r)) / cos(mlat * lat2r)
            north += (-dP[n, m]) * (
                model.tor_c[i] * cos(m * mlt * mlt2r) +
                model.tor_s[i] * sin(m * mlt * mlt2r))

        east *= -1 / MU0 * 1e-6
        north *= -1 / MU0 * 1e-6

        out = np.array([east, north]).reshape(-1, 1, 1)
        mlats, mlts = np.meshgrid(mlat,mlt)
        assert_allclose(out, model.get_curl_free_current(mlat, mlt))
        assert_allclose(out, model.get_curl_free_current(mlat, mlt))
        assert_allclose(
            np.array(model.get_curl_free_current(mlats, mlts)).flatten(),
            np.array(model.get_curl_free_current(mlat.flatten(), mlt.flatten(), True)).flatten()
        )
        assert_allclose(np.split(model.get_curl_free_current()[0], 2)[0].reshape(model.plotgrid_vector[0].shape),
                        model.get_curl_free_current(*model.plotgrid_vector)[0])
        assert_allclose(np.split(model.get_curl_free_current()[1], 2)[0].reshape(model.plotgrid_vector[0].shape),
                        model.get_curl_free_current(*model.plotgrid_vector)[1])

        pass

    @pytest.mark.parametrize("mlat, mlt", [(np.array([[60.]]), np.array([[0.]])),
                                           (np.array([[71.]]), np.array([[6.]]))])
    def test_get_total_current(self, amps_model, mlat, mlt):
        model, _, m_kwargs = amps_model

        mlats, mlts = np.meshgrid(mlat,mlt)
        assert_allclose(np.array(model.get_curl_free_current(mlat, mlt)) +
                        np.array(model.get_divergence_free_current(mlat, mlt)),
                        model.get_total_current(mlat, mlt))
        assert_allclose(
            np.array(model.get_total_current(mlats, mlts)).flatten(),
            np.array(model.get_total_current(mlat.flatten(), mlt.flatten(), True)).flatten()
        )
        assert_allclose(np.split(model.get_total_current()[0], 2)[0].reshape(model.plotgrid_vector[0].shape),
                        model.get_total_current(*model.plotgrid_vector)[0])
        assert_allclose(np.split(model.get_total_current()[1], 2)[0].reshape(model.plotgrid_vector[0].shape),
                        model.get_total_current(*model.plotgrid_vector)[1])

        pass

    def test_get_integrated_upward_current(self, amps_model):
        model, _, m_kwargs = amps_model
        mlt2r = pi / 12
        lat2r = pi / 180
        REFRE = 6371.2

        mlat, mlt = model.scalargrid
        mlat_res = (model.maxlat - model.minlat) / (m_kwargs['resolution'] - 1) * lat2r
        mlt_res = (mlt.max() - mlt.min()) / (m_kwargs['resolution'] - 1) * mlt2r
        dS = ((REFRE + m_kwargs['height']) * 1e3)**2 * np.cos(mlat * lat2r) * mlat_res * mlt_res

        ju = model.get_upward_current() * 1e-6
        J_n, J_s = np.split(dS * ju * 1e-6, 2)
        J = model.get_integrated_upward_current()

        assert_allclose(J_n[J_n > 0].sum(), J[0])
        assert_allclose(J_n[J_n < 0].sum(), J[1])
        assert_allclose(J_s[J_s > 0].sum(), J[2])
        assert_allclose(J_s[J_s < 0].sum(), J[3])

        pass

    @pytest.mark.parametrize("mlat, mlt", [(np.array([[60.]]), np.array([[0.]])),
                                           (np.array([[71.]]), np.array([[6.]]))])
    def test_get_ground_perturbation(self, amps_model, mlat, mlt):
        model, _, m_kwargs = amps_model
        mlt2r = pi / 12
        lat2r = pi / 180
        REFRE = 6371.2
        MU0 = pi * 4e-7

        P, dP = pyamps.sh_utils.legendre(model.N, model.M, 90 - mlat)

        G_north = 0
        G_east = 0
        for i, (n, m) in enumerate(model.keys_P):
            c = (REFRE / (REFRE + m_kwargs['height']))**(2 * n + 1) * (n + 1) / n
            G_north += c * (-dP[n, m]) * (
                model.pol_c[i] * cos(m * mlt * mlt2r) +
                model.pol_s[i] * sin(m * mlt * mlt2r))

            G_east += c * P[n, m] * (
                -m * model.pol_c[i] * sin(m * mlt * mlt2r) +
                m * model.pol_s[i] * cos(m * mlt * mlt2r)) / cos(mlat * lat2r)

        G = model.get_ground_perturbation(mlat, mlt)
        assert_allclose(G_east, G[0])
        assert_allclose(G_north, G[1])

        pass

    def test_get_AE_indices(self, amps_model):
        model, _, m_kwargs = amps_model

        Bn = model.get_AE_indices()
        assert_allclose(Bn, [-3.93173, -2.55411, 4.280122, 3.508369], atol=1e-6)

        pass

    @pytest.mark.skip(reason="No test performed")
    def test_plot_currents(self, amps_model):
        pass

@pytest.mark.apex_dep
def test_get_B_space():
    # params, input for get_B_ground: glat, glon, height, time, v, By, Bz, tilt, f107 
    params = list(map(lambda x: np.array(x).reshape(-1), [
        [80, 75],
        [0, 20],
        [110, 110],
        [datetime.datetime(2015,1,2), datetime.datetime(2015,4,2)],
        [0, 0],
        [0, 0],
        [1, 1],
        [0.5, 0.5],
        [0.3, 0.3]]))
    B_e, B_n, B_r = get_B_space(*params, coeff_fn = MODEL_COEFF_0103)
    assert_allclose(B_e, [1.214, -14.165], atol=1e-3)
    assert_allclose(B_n, [0.445, 7.911], atol=1e-3)
    assert_allclose(B_r, [-6.659, -7.802], atol=1e-3)
    pass


def test_get_B_ground():
    # params, input for get_B_space: qdlat, mlt, height, v, By, Bz, tilt, f107
    params = list(map(lambda x: np.array(x), [90, 0, 110, 0, 0, 1, 0.5, 0.3]))
    Bqphi, Bqlambda, Bqr = get_B_ground(*params, coeff_fn = test_coeff_fn)
    assert_allclose(Bqphi, 0, atol=1e-3)
    assert_allclose(Bqlambda, 1.973, atol=1e-3)
    assert_allclose(Bqr, 1.433, atol=1e-3)
    pass
