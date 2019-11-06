from __future__ import division

import datetime
import pytest
import numpy as np

from numpy.testing import assert_array_equal, assert_allclose
from pyamps.sh_utils import SHkeys, nterms, legendre, getG0, get_ground_field_G0


@pytest.mark.incremental
@pytest.mark.parametrize("N, M, idx", ([5, 3, 13],
                                       [3, 0, 1],
                                       [15, 15, 255],
                                       [2, 0, 0]),
                         scope="class")
class Test_SHkeys(object):

    def test_init(self, N, M, idx):
        k = SHkeys(N, M)
        assert len(k.keys) == (N + 1) * (M + 1)
        assert k.keys[idx] == (idx // (M + 1), idx % (M + 1))

    def test_getitem(self, N, M, idx):
        k = SHkeys(N, M)

        assert k['n'] == [n for n in range(N + 1) for m in range(M + 1)]
        assert k['m'] == [m for n in range(N + 1) for m in range(M + 1)]
        assert k[idx] == k.keys[idx]

    def test_len(self, N, M, idx):
        k = SHkeys(N, M)

        assert len(k) == len(k.keys)

    def test_repr(self, N, M, idx):
        k = SHkeys(N, M)
        lines = repr(k).split('\n')

        assert lines[0] == 'n, m'
        assert lines[idx + 1] == repr(k.keys[idx]).strip('()')

    def test_str(self, N, M, idx):
        k = SHkeys(N, M)
        lines = str(k).split('\n')

        assert lines[0] == 'n, m'
        assert lines[idx + 1] == str(k.keys[idx]).strip('()')

    def test_setNmin(self, N, M, idx):
        k = SHkeys(N, M).setNmin(2)
        if N < 2:
            with pytest.raises(IndexError):
                bool(k.n[0, 0])
            return
        else:
            assert k.n[0, 0] == 2
            assert k.n.shape[1] == (N + 1 - 2) * (M + 1)
            assert len(k.keys) == (N + 1 - 2) * (M + 1)
        if M > 0:
            assert k.keys[M + 1] == (3, 0)

    def test_MleN(self, N, M, idx):
        k = SHkeys(N, M).MleN()
        m_gt_n = [m > n for n, m in k.keys]

        assert len(k.keys) == (N + 1) + N * M - M * (M - 1) / 2
        assert sum(m_gt_n) == 0

    def test_Mge(self, N, M, idx):
        limit = 4
        k = SHkeys(N, M).Mge(limit)

        if M < limit:
            assert k.n.shape == (1, 0)
        else:
            assert k.m.shape[1] == (N + 1) * (M + 1 - limit)
            assert k.m[0, 0] == limit

    def test_NminusModd(self, N, M, idx):
        k = SHkeys(N, M)
        nm = np.c_[k.n[0], k.m[0]]
        k.NminusModd()
        print(k.n[0], nm.sum(1) % 2 == (1, 0),nm)
        assert k.n.shape[1] == (nm.sum(1) % 2).sum()
        assert_array_equal(k.n[0], nm[(nm.sum(1) % 2 == 1), 0])

    def test_NminusMeven(self, N, M, idx):
        k = SHkeys(N, M)
        nm = np.c_[k.n[0], k.m[0]]
        k.NminusMeven()

        assert k.n.shape[1] == ((nm.sum(1) + 1) % 2).sum()
        assert_array_equal(k.n[0], nm[(nm.sum(1) % 2 == 0), 0])

    def test_negative_m(self, N, M, idx):
        k = SHkeys(N, M).negative_m()

        assert k.m.sum() == 0
        assert k.m.shape[1] == (N + 1) * (2 * M + 1)

    def test_make_array(self, N, M, idx):
        k = SHkeys(N, M)

        assert type(k.n) == np.ndarray
        assert k.n.dtype == np.int
        assert k.n.shape == k.m.shape
        assert k.n.shape[1] == (N + 1) * (M + 1)
        assert_array_equal(k.n[0], [n for n, m in k.keys])
        assert_array_equal(k.m[0], [m for n, m in k.keys])
        assert k.n[0, idx] == idx // (M + 1)
        assert k.m[0, idx] == idx % (M + 1)


@pytest.mark.parametrize("N,M", [(10, 6)])
def test_SHkeys_composite(N, M):
    k1 = SHkeys(N, M)
    k2 = SHkeys(N, M)

    assert k1.negative_m().Mge(2).n.shape[1] == 110
    assert k1.MleN().m.shape[1] == 70
    assert_array_equal(k1, k2.MleN().Mge(2).negative_m())
    assert k1.NminusModd().m.shape[1] == 32
    assert k1.NminusMeven().m.shape[1] == 0


@pytest.mark.parametrize("N,M", [(20, 15), (5, 0), (8, 8)])
def test_nterms(N, M):
    mlen = N * M + (N + 1) - M * (M - 1) / 2
    nterms_ = 2 * (mlen - 1) - N

    assert nterms(NT=N, MT=M) == nterms_
    assert nterms(NVi=N, MVi=M) == nterms_
    assert nterms(NVe=N, MVe=M) == nterms_
    assert nterms(NT=N, MT=M, NVi=N, MVi=M, NVe=N, MVe=M) == 3 * nterms_


@pytest.mark.parametrize("inp,out", [
    [(2, 1, 180, False, 1, 0), (-1, 0, 2. / 3, -1)],
    [(20, 15, 31.5, True, 15, 14), (2.83735e-4, 6.30832e-3, 4.39178e-2, -1.27220)]
])
def test_legendre(inp, out):
    P, dP = legendre(
        nmax=inp[0],
        mmax=inp[1],
        theta=np.array(inp[2]),
        schmidtnormalize=inp[3])
    PdP = legendre(
        nmax=inp[0],
        mmax=inp[1],
        theta=np.array(inp[2]),
        schmidtnormalize=inp[3],
        keys=SHkeys(inp[0], inp[1]).MleN())

    

    assert len(P) >= len(SHkeys(inp[0], inp[1]).MleN())
    assert PdP.shape == (1, 2 * len(SHkeys(inp[0], inp[1]).MleN()))
    assert_allclose(P[inp[4], inp[5]], out[0], rtol=1e-4, atol=1e-14)
    assert_allclose(dP[inp[4], inp[5]], out[1], rtol=1e-4, atol=1e-14)

    # Tests cannot be performed until clarification on legendre API 
    # assert_allclose(PdP[inp[4], inp[5]].sum(), out[0], rtol=1e-4, atol=1e-14)
    # assert_allclose(PdP[inp[4], inp[1] + inp[5]].sum(), out[1], rtol=1e-4, atol=1e-14)

    # assert_allclose(PdP[:, inp[5]].sum(), out[2], rtol=1e-4, atol=1e-14)
    # assert_allclose(PdP[inp[4], inp[1]:].sum(), out[3], rtol=1e-4, atol=1e-14)

@pytest.mark.apex_dep
def test_getG0():
    glat = np.array([80, 10])
    glon = np.array([63, 175])
    time = np.array([datetime.datetime(2000, 1, 2, 3, 4, 5, 6),
                     datetime.datetime(2000, 1, 2, 4, 5, 6, 7)])
    height = np.array([110, 110])
    G0 = getG0(glat, glon, height, time, epoch=2000)

    assert G0.shape == (3 * 2, 758)
    assert_allclose(np.abs(G0[3, :]).sum(), 665.387696, atol=1e-4)
    assert_allclose(G0[2, ::200], [-0.03689656, -1.82078773, -0.6418252 ,  1.74947096], atol=1e-4)


def test_get_ground_field_G0():

    qdlat = np.array([90, 0])
    mlt = np.array([0, 24])
    height = 30
    current_height = 110
    G0 = get_ground_field_G0(qdlat, mlt, height, current_height)

    assert G0.shape == (3 * 2, 309)
    assert_allclose(G0[0], np.zeros(309), atol=1e-15)
    assert_allclose(G0[4, :8], [1.8999, 0.0000, 2.7669,
                                0.0000, 0.0000, 3.5818,
                                0.0000, 0.0000], atol=1e-4)
