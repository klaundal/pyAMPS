from __future__ import division

import pytest
import numpy as np


import matplotlib.pyplot as plt
from numpy.testing import assert_array_equal, assert_allclose

from pyamps.plot_utils import Polarsubplot, equal_area_grid

plt.switch_backend('Agg')  

class Test_Polarsubplot(object):

    def test_init(self):

        fig = plt.figure()
        ax = fig.add_subplot(111)
        pax = Polarsubplot(ax, minlat=65)

        assert ax == pax.ax
        assert pax.minlat == 65

        pass

    @pytest.mark.skip(reason="No test performed")
    def test_plot(self):
        pass

    @pytest.mark.skip(reason="No test performed")
    def test_write(self):
        pass

    @pytest.mark.skip(reason="No test performed")
    def test_scatter(self):
        pass

    @pytest.mark.skip(reason="No test performed")
    def test_plotgrid(self):
        pass

    @pytest.mark.skip(reason="No test performed")
    def test_writeMLTlabels(self):
        pass

    @pytest.mark.skip(reason="No test performed")
    def test_featureplot(self):
        pass

    @pytest.mark.skip(reason="No test performed")
    def test_contour(self):
        pass

    @pytest.mark.skip(reason="No test performed")
    def test_contourf(self):
        pass

    def test__mlat_mlt_to_xy(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        pax = Polarsubplot(ax, minlat=77)
        pax1 = Polarsubplot(ax, minlat=0)

        xy1 = pax1._mlat_mlt_to_xy(0, np.arange(0, 24, 6))
        assert_allclose(xy1, [[0, 1, 0, -1], [-1, 0, 1, 0]], atol=1e-15)

        xy2 = pax._mlat_mlt_to_xy(-80, 6)
        assert_allclose(xy2, [10. / (90 - 77), 0], atol=1e-15)

        xy3 = pax._mlat_mlt_to_xy(90, 5.33)
        assert_allclose(xy3, [0, 0], atol=1e-15)

        xy4 = pax._mlat_mlt_to_xy(88.2, 0)
        xy5 = pax._mlat_mlt_to_xy(88.2, 24)

        assert_allclose(xy4, xy5, atol=1e-15)

    def test__xy_to_mlat_mlt(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        pax = Polarsubplot(ax, minlat=77)
        pax1 = Polarsubplot(ax, minlat=0)

        x_in, y_in = np.array([0, 1, 0, -1]), np.array([-1, 0, 1, 0])
        mlat1, mlt1 = pax1._xy_to_mlat_mlt(x_in, y_in)
        assert_allclose(mlt1, [0, 6, 12, 18])
        assert_allclose(mlat1, [0, 0, 0, 0])

        mlt_mlat2 = pax._xy_to_mlat_mlt(0, 0)
        assert_allclose(mlt_mlat2[0], [90])

        xy = pax._mlat_mlt_to_xy(*pax._xy_to_mlat_mlt(0.34, 0.56))
        assert_allclose(xy, [[0.34], [0.56]])

    def test__north_east_to_cartesian(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        pax = Polarsubplot(ax)

        mlt_values = np.arange(0, 24, 6)
        north, east = 43.2, 67.8
        xy1 = pax._north_east_to_cartesian(north, east, mlt_values)
        assert_allclose(xy1, [[east, -north, -east, north], [north, east, -north, -east]])
        xy2 = pax._north_east_to_cartesian(0, 0, mlt_values)
        assert_allclose(xy2, np.zeros((2, 4)))
        pass


@pytest.mark.parametrize("dr, K, M0, N", [(1.5, 1, 2, 6), (2, 0, 8, 20), (0.5, 1, 2, 3)])
def test_equal_area_grid(dr, K, M0, N):

    mlats, mlts, mltres = equal_area_grid(dr, K, M0, N)

    mlat_values, mlat_counts = np.unique(mlats, return_counts=True)
    assert mlats.shape == mlts.shape
    assert mlats.shape == mltres.shape

    assert mlat_values.shape[0] == N
    assert mlat_counts[-1] == M0
    assert mlat_counts[-2] == int(M0 * (1 + 1. / (K + 1)))
    assert_allclose(mltres.sum(), 24 * N)
    assert_allclose(mlats[-1], 90 - (dr + dr * (2 * K + 1) / 2))  # 90-(dr+r0)

    with pytest.raises(AssertionError):
        equal_area_grid(K=1, M0=9)

    pass
