""" tools that are useful for spherical harmonic analysis

    SHkeys       -- class to contain n and m - the indices of the spherical harmonic terms
    nterms       -- function which calculates the number of terms in a 
                    real expansion of a poloidal (internal + external) and toroidal expansion 
    legendre -- calculate associated legendre functions - with option for Schmidt semi-normalization



MIT License

Copyright (c) 2017 Karl M. Laundal

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import absolute_import, division
import numpy as np
import apexpy
from .mlt_utils import mlon_to_mlt
from builtins import range

d2r = np.pi/180

DEFAULT = object()
refre = 6371.2 # reference radius

class SHkeys(object):
    """ container for n and m in spherical harmonics

        keys = SHkeys(Nmax, Mmax)

        keys will behave as a tuple of tuples, more or less
        keys['n'] will return a list of the n's
        keys['m'] will return a list of the m's
        keys[3] will return the fourth n,m tuple

        keys is also iterable

    """

    def __init__(self, Nmax, Mmax):
        keys = []
        for n in range(Nmax + 1):
            for m in range(Mmax + 1):
                keys.append((n, m))

        self.keys = tuple(keys)
        self.make_arrays()

    def __getitem__(self, index):
        if index == 'n':
            return [key[0] for key in self.keys]
        if index == 'm':
            return [key[1] for key in self.keys]

        return self.keys[index]

    def __iter__(self):
        for key in self.keys:
            yield key

    def __len__(self):
        return len(self.keys)

    def __repr__(self):
        return ''.join(['n, m\n'] + [str(key)[1:-1] + '\n' for key in self.keys])[:-1]

    def __str__(self):
        return ''.join(['n, m\n'] + [str(key)[1:-1] + '\n' for key in self.keys])[:-1]

    def setNmin(self, nmin):
        """ set minimum n """
        self.keys = tuple([key for key in self.keys if key[0] >= nmin])
        self.make_arrays()
        return self

    def MleN(self):
        """ set m <= n """
        self.keys = tuple([key for key in self.keys if abs(key[1]) <= key[0]])
        self.make_arrays()
        return self

    def Mge(self, limit):
        """ set m >= limit  """
        self.keys = tuple([key for key in self.keys if abs(key[1]) >= limit])
        self.make_arrays()
        return self

    def NminusModd(self):
        """ remove keys if n - m is even """
        self.keys = tuple([key for key in self.keys if (key[0] - abs(key[1])) % 2 == 1])
        self.make_arrays()
        return self

    def NminusMeven(self):
        """ remove keys if n - m is odd """
        self.keys = tuple([key for key in self.keys if (key[0] - abs(key[1])) % 2 == 0])
        self.make_arrays()
        return self

    def negative_m(self):
        """ add negative m to the keys """
        keys = []
        for key in self.keys:
            keys.append(key)
            if key[1] != 0:
                keys.append((key[0], -key[1]))
        
        self.keys = tuple(keys)
        self.make_arrays()
        
        return self


    def make_arrays(self):
        """ prepare arrays with shape ( 1, len(keys) )
            these are used when making G matrices
        """

        if len(self) > 0:
            self.m = np.array(self)[:, 1][np.newaxis, :]
            self.n = np.array(self)[:, 0][np.newaxis, :]
        else:
            self.m = np.array([])[np.newaxis, :]
            self.n = np.array([])[np.newaxis, :]



def nterms(NT = 0, MT = 0, NVi = 0, MVi = 0, NVe = 0, MVe = 0):
    """ return number of coefficients in an expansion in real spherical harmonics of
        toroidal magnetic potential truncated at NT, MT
        poloidal magnetic potential truncated at NVi, MVi for internal sources
        poloidal magnetic potential truncated at NVe, MVe for external sources
    """

    return len(SHkeys(NT , MT ).setNmin(1).MleN().Mge(0)) + \
           len(SHkeys(NT , MT ).setNmin(1).MleN().Mge(1)) + \
           len(SHkeys(NVe, MVe).setNmin(1).MleN().Mge(0)) + \
           len(SHkeys(NVe, MVe).setNmin(1).MleN().Mge(1)) + \
           len(SHkeys(NVi, MVi).setNmin(1).MleN().Mge(0)) + \
           len(SHkeys(NVi, MVi).setNmin(1).MleN().Mge(1))



def legendre(nmax, mmax, theta, schmidtnormalize = True, keys = None):
    """ Calculate associated Legendre function P and its derivative

        Algorithm from "Spacecraft Attitude Determination and Control" by James Richard Wertz


        Parameters
        ----------
        nmax : int
            highest spherical harmonic degree
        mmax : int
            hightest spherical harmonic order
        theta : array, float
            colatitude in degrees (shape is not preserved)
        schmidtnormalize : bool, optional
            True if Schmidth seminormalization is wanted, False otherwise. Default True
        keys : SHkeys, optional
            If this parameter is set, an array will be returned instead of a dict. 
            The array will be (N, 2M), where N is the number of elements in `theta`, and 
            M is the number of keys. The first M columns represents a matrix of P values, 
            and the last M columns represent values of dP/dtheta

        Returns
        -------
        P : dict
            dictionary of Legendre function evalulated at theta. Dictionary keys are spherical harmonic
            wave number tuples (n, m), and values will have shape (N, 1), where N is number of 
            elements in `theta`. 
        dP : dict
            dictionary of Legendre function derivatives evaluated at theta. Dictionary keys are spherical
            harmonic wave number tuples (n, m), and values will have shape (N, 1), where N is number of 
            elements in theta. 
        PdP : array (only if keys != None)
            if keys != None, PdP is returned instaed of P and dP. PdP is an (N, 2M) array, where N is 
            the number of elements in `theta`, and M is the number of keys. The first M columns represents 
            a matrix of P values, and the last M columns represent values of dP/dtheta

    """

    theta = theta.flatten()[:, np.newaxis]

    P = {}
    dP = {}
    sinth = np.sin(d2r*theta)
    costh = np.cos(d2r*theta)

    if schmidtnormalize:
        S = {}
        S[0, 0] = 1.

    # initialize the functions:
    for n in range(nmax +1):
        for m in range(nmax + 1):
            P[n, m] = np.zeros_like(theta, dtype = np.float64)
            dP[n, m] = np.zeros_like(theta, dtype = np.float64)

    P[0, 0] = np.ones_like(theta, dtype = np.float64)
    for n in range(1, nmax +1):
        for m in range(0, min([n + 1, mmax + 1])):
            # do the legendre functions and derivatives
            if n == m:
                P[n, n]  = sinth * P[n - 1, m - 1]
                dP[n, n] = sinth * dP[n - 1, m - 1] + costh * P[n - 1, n - 1]
            else:

                if n == 1:
                    Knm = 0.
                    P[n, m]  = costh * P[n -1, m]
                    dP[n, m] = costh * dP[n - 1, m] - sinth * P[n - 1, m]

                elif n > 1:
                    Knm = ((n - 1)**2 - m**2) / ((2*n - 1)*(2*n - 3))
                    P[n, m]  = costh * P[n -1, m] - Knm*P[n - 2, m]
                    dP[n, m] = costh * dP[n - 1, m] - sinth * P[n - 1, m] - Knm * dP[n - 2, m]

            if schmidtnormalize:
                # compute Schmidt normalization
                if m == 0:
                    S[n, 0] = S[n - 1, 0] * (2.*n - 1)/n
                else:
                    S[n, m] = S[n, m - 1] * np.sqrt((n - m + 1)*(int(m == 1) + 1.)/(n + m))


    if schmidtnormalize:
        # now apply Schmidt normalization
        for n in range(1, nmax + 1):
            for m in range(0, min([n + 1, mmax + 1])):
                P[n, m]  *= S[n, m]
                dP[n, m] *= S[n, m]


    if keys is None:
        return P, dP
    else:
        Pmat  = np.hstack(tuple(P[key] for key in keys))
        dPmat = np.hstack(tuple(dP[key] for key in keys)) 
    
        return np.hstack((Pmat, dPmat))




def getG0(glat, glon, height, time, epoch = 2015., h_R = 110., NT = 65, MT = 3, NV = 45, MV = 3):
    """ calculate the G matrix for the constant term in the AMPS model. The constant term is the 
        term that depends only on the spherical harmonic coefficients that are not scaled by 
        external parameters. This G matrix can be used to produce the full matrix.

        The structure of the matrix is such that G0.dot(m), the product of the matrix with the 
        first 1/19 of the model vector, will be model values of the eastward, northward, and
        upward components of the magnetic field, stacked in a column vector.

        glat, glon, time, and height must all have the same number of elements (let's call this N)

        Parameters
        ----------
        glat : array
            Geodetic latitude (degrees)
        glon : array
            Geographic/geodetic longitude (degrees)
        height : array
            Geodetic heights, in km
        time : array
            Array of datetimes corresponding to each point. This is needed to calculate 
            magnetic local time.
        epoch : float, optional
            The epoch used for conversion to apex coordinates. Default 2015.
        h_R : float, optional
            Reference height used in conversion to modified apex coordinates. Default 110 km.
        NT, MT, NV, MV: int, optional
            Truncation level. Must match coefficient file

        Returns
        -------
        G0 : array
            an 3N by M matrix, where N is the number of elements in the input coordinates. There will be
            3 times as many rows G0, since there are 3 components. Partionining G0 in thirds, from top to
            bottom, gives the parts that correspond to east, north, and up, respectively. M
            is the number of terms in the spherical harmonic expansion of B

    """
    glat   = np.asarray(glat).flatten()
    glon   = np.asarray(glon).flatten()
    height = np.asarray(height).flatten()

    # convert to magnetic coords and get base vectors
    a = apexpy.Apex(epoch, refh = h_R)
    qlat, qlon = a.geo2qd(  glat.flatten(), glon.flatten(), height.flatten())
    alat, alon = a.geo2apex(glat.flatten(), glon.flatten(), height.flatten())
    f1, f2, f3, g1, g2, g3, d1, d2, d3, e1, e2, e3 = a.basevectors_apex(qlat, qlon, height, coords  = 'qd')
    f1e = f1[0].reshape(-1, 1) # base vector components as column vectors
    f1n = f1[1].reshape(-1, 1)
    f2e = f2[0].reshape(-1, 1)
    f2n = f2[1].reshape(-1, 1)
    d1e = d1[0].reshape(-1, 1)
    d1n = d1[1].reshape(-1, 1)
    d2e = d2[0].reshape(-1, 1)
    d2n = d2[1].reshape(-1, 1)

    # calculate magnetic local time
    phi = mlon_to_mlt(qlon, time, a.year)[:, np.newaxis]*15 # multiply by 15 to get degrees

    # turn the coordinate arrays into column vectors:
    alat, qlat, h = map(lambda x: x.flatten()[:, np.newaxis], [alat, qlat, height])

    # generate spherical harmonic keys    
    keys = {} # dictionary of spherical harmonic keys
    keys['cos_T'] = SHkeys(NT, MT).setNmin(1).MleN().Mge(0)
    keys['sin_T'] = SHkeys(NT, MT).setNmin(1).MleN().Mge(1)
    keys['cos_V'] = SHkeys(NV, MV).setNmin(1).MleN().Mge(0)
    keys['sin_V'] = SHkeys(NV, MV).setNmin(1).MleN().Mge(1)
    m_cos_V = keys['cos_V'].m
    m_sin_V = keys['sin_V'].m
    m_cos_T = keys['cos_T'].m
    m_sin_T = keys['sin_T'].m

    nV = np.hstack((keys['cos_V'].n, keys['sin_V'].n))

    # generate Legendre matrices - first get dicts of arrays, and then stack them in the appropriate fashion
    legendre_T = legendre(NT, MT, 90 - alat, keys = keys['cos_T'])
    legendre_V = legendre(NV, MV, 90 - qlat, keys = keys['cos_V'])
    P_cos_T  =  legendre_T[:, :len(keys['cos_T']) ] # split
    dP_cos_T = -legendre_T[:,  len(keys['cos_T']):]
    P_cos_V  =  legendre_V[:, :len(keys['cos_V']) ] # split
    dP_cos_V = -legendre_V[:,  len(keys['cos_V']):]
    P_sin_T  =  P_cos_T[ :, keys['cos_T'].m.flatten() != 0] 
    dP_sin_T =  dP_cos_T[:, keys['cos_T'].m.flatten() != 0]
    P_sin_V  =  P_cos_V[ :, keys['cos_V'].m.flatten() != 0]
    dP_sin_V =  dP_cos_V[:, keys['cos_V'].m.flatten() != 0]  

    # trig matrices:
    cos_T  =  np.cos(phi * d2r * m_cos_T)
    sin_T  =  np.sin(phi * d2r * m_sin_T)
    cos_V  =  np.cos(phi * d2r * m_cos_V)
    sin_V  =  np.sin(phi * d2r * m_sin_V)
    dcos_T = -np.sin(phi * d2r * m_cos_T)
    dsin_T =  np.cos(phi * d2r * m_sin_T)
    dcos_V = -np.sin(phi * d2r * m_cos_V)
    dsin_V =  np.cos(phi * d2r * m_sin_V)


    cos_qlat   = np.cos(qlat * d2r)
    cos_alat   = np.cos(alat * d2r)

    sinI  = 2 * np.sin( alat * d2r )/np.sqrt(4 - 3*cos_alat**2)

    r  = refre + h
    Rtor  = refre/r

    F = f1e*f2n - f1n*f2e

    # matrix with horizontal spherical harmonic functions in QD coordinates
    V        = np.hstack((P_cos_V * cos_V, P_sin_V * sin_V ))

    # matrices with partial derivatives in QD coordinates:
    dV_dqlon  = np.hstack(( P_cos_V * dcos_V * m_cos_V,  P_sin_V * dsin_V * m_sin_V ))
    dV_dqlat  = np.hstack((dP_cos_V *  cos_V          , dP_sin_V *  sin_V           ))

    # matrices with partial derivatives in MA coordinates:
    dT_dalon  = np.hstack(( P_cos_T * dcos_T * m_cos_T,  P_sin_T * dsin_T * m_sin_T))
    dT_dalat  = np.hstack((dP_cos_T *  cos_T          , dP_sin_T *  sin_T          ))

    # Toroidal field components
    B_T_e  =   -d1n * dT_dalon / cos_alat + d2n * dT_dalat / sinI
    B_T_n  =    d1e * dT_dalon / cos_alat - d2e * dT_dalat / sinI
    B_T_u  =    np.zeros(B_T_n.shape)

    # Poloidal field components:
    B_V_e = (-f2n / (cos_qlat * r) * dV_dqlon + f1n * dV_dqlat / r) * refre * Rtor ** (nV + 1)
    B_V_n = ( f2e / (cos_qlat * r) * dV_dqlon - f1e * dV_dqlat / r) * refre * Rtor ** (nV + 1)
    B_V_u = np.sqrt(F) * V  * (nV + 1) * Rtor ** (nV + 2)

    # combine:
    G     = np.hstack((np.vstack((B_T_e  , 
                                  B_T_n  , 
                                  B_T_u  )),   np.vstack((B_V_e, 
                                                          B_V_n, 
                                                          B_V_u))
                      ))

    return G


def get_ground_field_G0(qdlat, mlt, height, current_height, N = 45, M = 3):
    """ calculate the G matrix for the constant term in the AMPS model needed to calculate
        corresponding ground magnetic field perturbations. The constant term is the 
        term that depends only on the spherical harmonic coefficients that are not scaled by 
        external parameters. This G matrix can be used to produce the full matrix.


        Parameters
        ----------
        qdlat : array
            Quasi-dipole latitude (degrees)
        MLT : array
            Magnetic local time (MLT)
        height : float
            Geodetic height, in km (0 <= height <= current_height)
        current_height : float
            height of the current
        N, M: truncation level (int, optional) - must match coefficient file

        Returns
        -------
        G0 : array
            an 3N by M matrix, where N is the number of elements in the input coordinates. There will be
            3 times as many rows G0, since there are 3 components. Partionining G0 in thirds, from top to
            bottom, gives the parts that correspond to east, north, and up, respectively. M
            is the number of terms in the spherical harmonic expansion of B

    """
    qdlat  = qdlat.flatten() [:, np.newaxis]
    mlt    = mlt.flatten()   [:, np.newaxis]

    # convert mlt to degreees
    phi = mlt*15 # multiply by 15 to get degrees


    # generate spherical harmonic keys    
    keys = {} # dictionary of spherical harmonic keys
    keys['cos'] = SHkeys(N, M).setNmin(1).MleN().Mge(0)
    keys['sin'] = SHkeys(N, M).setNmin(1).MleN().Mge(1)
    m_cos = keys['cos'].m
    m_sin = keys['sin'].m

    n = np.hstack((keys['cos'].n, keys['sin'].n))

    # generate Legendre matrices - first get dicts of arrays, and then stack them in the appropriate fashion
    legendre_arr = legendre(N, M, 90 - qdlat, keys = keys['cos'])
    P_cos  =  legendre_arr[: , :len(keys['cos']) ] # split
    dP_cos = -legendre_arr[: ,  len(keys['cos']):]
    P_sin  =  P_cos       [: , keys['cos'].m.flatten() != 0]
    dP_sin =  dP_cos      [: , keys['cos'].m.flatten() != 0]  

    # trig matrices:
    cos  =  np.cos(phi * d2r * m_cos)
    sin  =  np.sin(phi * d2r * m_sin)
    dcos = -np.sin(phi * d2r * m_cos)
    dsin =  np.cos(phi * d2r * m_sin)

    cos_qlat   = np.cos(qdlat * d2r)

    r = refre + height
    r_ratio_horizontal = (r / (refre + current_height)) ** (n    ) * (refre / (refre + current_height)) ** (n + 1) * (n + 1) / n
    r_ratio_vertical   = (r / (refre + current_height)) ** (n - 1) * (refre / (refre + current_height)) ** (n + 2) * (n + 1)

    # matrix with trig functions
    trigs   = np.hstack((cos, sin ))

    # matrix with derivative of trig functions
    trigs_d = np.hstack(( dcos * m_cos, dsin * m_sin ))

    # matrices with P and dP stacked
    P  = np.hstack((P_cos , P_sin ))
    dP = np.hstack((dP_cos, dP_sin))

    G0east  = P  * trigs_d * r_ratio_horizontal / cos_qlat
    G0north = dP * trigs   * r_ratio_horizontal 
    G0up    = P  * trigs   * r_ratio_vertical

    # stack vertically and return 
    return np.vstack((G0east, G0north, G0up))

