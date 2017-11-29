from __future__ import division

import pytest
import datetime
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

import pyamps
from pyamps.mlt_utils import mlon_to_mlt, sph_to_car, car_to_sph, subsol, is_leapyear, geo2mag


@pytest.mark.parametrize("mlon, times, mlt",
                         [(27.6361193, datetime.datetime(2014, 10, 29, 10, 18, 1), 7.746497),
                          (358.9850748, datetime.datetime(2016, 3, 1, 22, 57, 2), 17.742899)])
def test_mlon_to_mlt(mlon, times, mlt):
    atol = 1e-5
    mlon_np = np.array(mlon)

    assert_allclose(mlon_to_mlt(mlon=mlon_np, times=times, epoch=times.year),
                    mlt,
                    atol=atol)
    assert_allclose(mlon_to_mlt(mlon=(mlon_np + 180) % 360, times=times, epoch=times.year),
                    (mlt + 12) % 24,
                    atol=atol)

    mlon_arr = np.arange(mlon, mlon + 361, 15) % 360
    times_arr = np.repeat(times, len(mlon_arr))
    mlt_arr = np.arange(mlt, mlt + 25) % 24

    assert_allclose(mlon_to_mlt(mlon=mlon_arr, times=times, epoch=times.year),
                    mlt_arr,
                    atol=atol)
    assert_allclose(mlon_to_mlt(mlon=mlon_arr, times=times_arr, epoch=times.year),
                    mlt_arr,
                    atol=atol)


@pytest.mark.parametrize("r,theta,phi,deg,xyz",
                         [(1.0, 90.0, 45.0, True, np.array([np.sqrt(2) / 2, np.sqrt(2) / 2, 0])),
                          (2.12, np.pi / 4, 3 * np.pi / 4, False, np.array([-1.06, 1.06, 2.12 * (np.sqrt(2) / 2)]))])
def test_sph_to_car(r, theta, phi, deg, xyz):
    rtol, atol = 1e-5, 1e-15
    sph = np.array([r, theta, phi])

    assert_allclose(sph_to_car(sph[:, np.newaxis], deg),
                    xyz[:, np.newaxis],
                    rtol=rtol, atol=atol)

    sph5 = np.repeat(sph, 5).reshape(3, -1)
    xyz5 = np.repeat(xyz, 5).reshape(3, -1)

    assert car_to_sph(sph5, deg).shape == sph5.shape
    assert_allclose(sph_to_car(sph5, deg),
                    xyz5,
                    rtol=rtol, atol=atol)


@pytest.mark.parametrize("r, theta, phi, deg, xyz",
                         [(1.0, 90.0, 45.0, True, np.array([np.sqrt(2) / 2, np.sqrt(2) / 2, 0])),
                          (2.12, np.pi / 4, 3 * np.pi / 4, False, np.array([-1.06, 1.06, 2.12 * (np.sqrt(2) / 2)]))])
def test_car_to_sph(r, theta, phi, deg, xyz):
    rtol, atol = 1e-5, 1e-15
    sph = np.array([r, theta, phi])
    convert = (np.pi / 180)**deg
    assert_allclose(phi * convert, np.arctan2(xyz[1], xyz[0]))
    assert_allclose(car_to_sph(xyz[:, np.newaxis], deg),
                    sph[:, np.newaxis],
                    rtol=rtol, atol=atol)

    sph5 = np.repeat(sph, 5).reshape(3, -1)
    xyz5 = np.repeat(xyz, 5).reshape(3, -1)

    assert car_to_sph(xyz5, deg).shape == xyz5.shape
    assert_allclose(car_to_sph(xyz5, deg),
                    sph5,
                    rtol=rtol, atol=atol)


def test_subsol():
    atol = 0.025
    times = datetime.datetime(2014, 10, 29, 10, 18, 1)
    subsolcoor = np.array([-13.4882, 21.4283])

    assert type(subsol(times)[0]) == np.ndarray
    assert_allclose(subsol(times), *subsolcoor, atol=atol)

    times5 = np.repeat(times, 5)
    subsolcoor5 = np.repeat(subsolcoor, 5).reshape(2, -1)

    assert_allclose(subsol(times5), subsolcoor5, atol=atol)


def test_is_leapyear():

    assert is_leapyear(2000)
    assert not is_leapyear(2001)
    assert_array_equal(is_leapyear(np.arange(1600, 1605)), np.array([1, 0, 0, 0, 1]))
    assert is_leapyear(np.arange(1600, 2010)).sum() == 100


def test_geo2mag():
    time = 1925.0

    assert_allclose(geo2mag(glat=np.array(50), glon=np.array(50), epoch=time, deg=True),
                    [(43.660603,), (128.508908,)]
                    )
    assert_allclose(geo2mag(glat=np.array(50 * np.pi / 180),
                            glon=np.array(50 * np.pi / 180),
                            epoch=time, deg=False),
                    [(43.660603,), (128.508908,)])
    assert_allclose(geo2mag(glat=np.array([50, 60]),
                            glon=np.array([50, 60]),
                            epoch=time, deg=True),
                    [(43.660603, 51.937547), (128.508908, 140.454543)])
