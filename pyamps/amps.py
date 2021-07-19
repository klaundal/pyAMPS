""" 
Python interface for the Average Magnetic field and Polar current System (AMPS) model

This module can be used to 
1) Calculate and plot average magnetic field and current parameters 
   on a grid. This is done through the AMPS class. The parameters 
   that are available for calculation/plotting are:
    - field aligned current (scalar)
    - equivalent current function (scalar)
    - divergence-free part of horizontal current (vector)
    - curl-free part of horizontal current (vector)
    - total horizontal current (vector)
    - eastward or northward ground perturbation 
      corresponding to equivalent current (scalars)



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
import dask.array as da
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from .plot_utils import equal_area_grid, Polarsubplot
from .sh_utils import legendre, getG0, get_ground_field_G0
from .model_utils import get_model_vectors, get_m_matrix, get_m_matrix_pol, get_coeffs, default_coeff_fn, get_truncation_levels
from functools import reduce
from builtins import range



rc('text', usetex=False)

MU0   = 4*np.pi*1e-7 # Permeability constant
REFRE = 6371.2 # Reference radius used in geomagnetic modeling

DEFAULT = object()


class AMPS(object):
    """
    Calculate and plot maps of the model Average Magnetic field and Polar current System (AMPS)

    Parameters
    ---------
    v : float
        solar wind velocity in km/s
    By : float
        IMF GSM y component in nT
    Bz : float
        IMF GSM z component in nT
    tilt : float
        dipole tilt angle in degrees
    f107 : float
        F10.7 index in s.f.u.
    minlat : float, optional
        low latitude boundary of grids  (default 60)
    maxlat : float, optional
        low latitude boundary of grids  (default 89.99)
    height : float, optional
        altitude of the ionospheric currents in km (default 110)
    dr : int, optional
        latitudinal spacing between equal area grid points (default 2 degrees)
    M0 : int, optional
        number of grid points in the most poleward circle of equal area grid points (default 4)
    resolution: int, optional
        resolution in both directions of the scalar field grids (default 100)
    coeff_fn: str, optional
        file name of model coefficients - must be in format produced by model_vector_to_txt.py
        (default is latest version)


    Examples
    --------
    >>> # initialize by supplying a set of external conditions:
    >>> m = AMPS(solar_wind_velocity_in_km_per_s, 
                 IMF_By_in_nT, IMF_Bz_in_nT, 
                 dipole_tilt_in_deg, 
                 F107_index)
    
    >>> # make summary plot:
    >>> m.plot_currents()
        
    >>> # calculate field-aligned currents on a pre-defined grid
    >>> Ju = m.get_upward_current()

    >>> # Ju will be evaluated at the following coords:
    >>> mlat, mlt = m.scalargrid

    >>> # It is also possible to specify coordinates (can be slow with 
    >>> # repeated calls)
    >>> Ju = m.get_upward_current(mlat = my_mlats, mlt = my_mlts)

    >>> # get components of the total height-integrated horizontal current,
    >>> # calculated on a pre-defined grid
    >>> j_east, j_north = m.get_total_current()

    >>> # j_east, and j_north will be evaluated at the following coords 
    >>> # (default grids are different with vector quantities)
    >>> mlat, mlt = m.vectorgrid

    >>> # update model vectors (tor_c, tor_s, etc.) without 
    >>> # recalculating the other matrices:
    >>> m.update_model(new_v, new_By, new_Bz, new_tilt, new_f107)

    Attributes
    ----------
    tor_c : numpy.ndarray
        vector of cos term coefficents in the toroidal field expansion
    tor_s : numpy.ndarray
        vector of sin term coefficents in the toroidal field expansion
    pol_c : numpy.ndarray
        vector of cos term coefficents in the poloidal field expansion
    pol_s : numpy.ndarray
        vector of sin term coefficents in the poloidal field expansion
    keys_P : list
        list of spherical harmonic wave number pairs (n,m) corresponding to elements of pol_c and pol_s 
    keys_T : list
        list of spherical harmonic wave number pairs (n,m) corresponding to elements of tor_c and tor_s 
    vectorgrid : tuple
        grid used to calculate and plot vector fields
    scalargrid : tuple
        grid used to calculate and plot scalar fields
                   
        The grid formats are as follows (see also example below):
        (np.hstack((mlat_north, mlat_south)), np.hstack((mlt_north, mlt_south)))
        
        The grids can be changed directly, but member function calculate_matrices() 
        must then be called for the change to take effect. 

    """



    def __init__(self, v, By, Bz, tilt, f107, minlat = 60, maxlat = 89.99, height = 110., dr = 2, M0 = 4, resolution = 100, coeff_fn = default_coeff_fn):
        """ __init__ function for class AMPS
        """

        self.coeff_fn = coeff_fn
        self.tor_c, self.tor_s, self.pol_c, self.pol_s, self.pol_keys, self.tor_keys = get_model_vectors(v, By, Bz, tilt, f107, coeff_fn = self.coeff_fn)

        self.height = height

        self.dr = dr
        self.M0 = M0


        assert (len(self.pol_s) == len(self.pol_c)) and (len(self.pol_s) == len(self.pol_c))

        self.minlat = minlat
        self.maxlat = maxlat

        self.keys_P = [c for c in self.pol_keys]
        self.keys_T = [c for c in self.tor_keys]
        self.m_P = np.array(self.keys_P).T[1][np.newaxis, :]
        self.m_T = np.array(self.keys_T).T[1][np.newaxis, :]
        self.n_P = np.array(self.keys_P).T[0][np.newaxis, :]
        self.n_T = np.array(self.keys_T).T[0][np.newaxis, :]


        # find highest degree and order:
        self.N, self.M = np.max( np.hstack((np.array([c for c in self.tor_keys]).T, np.array([c for c in self.tor_keys]).T)), axis = 1)

        self.vectorgrid = self._get_vectorgrid()
        self.scalargrid = self._get_scalargrid(resolution = resolution)

        mlats = np.split(self.scalargrid[0], 2)[0].reshape((self.scalar_resolution, self.scalar_resolution))
        mlts  = np.split(self.scalargrid[1], 2)[0].reshape((self.scalar_resolution, self.scalar_resolution))
        mlatv = np.split(self.vectorgrid[0], 2)[0]
        mltv  = np.split(self.vectorgrid[1], 2)[0]

        self.plotgrid_scalar = (mlats, mlts)
        self.plotgrid_vector = (mlatv, mltv)



        self.calculate_matrices()


    def update_model(self, v, By, Bz, tilt, f107, coeff_fn = DEFAULT):
        """
        Update the model vectors without updating all the other matrices. This leads to better
        performance than just making a new AMPS object.

        Parameters
        ----------
        v : float
            solar wind velocity in km/s
        By : float
            IMF GSM y component in nT
        Bz : float
            IMF GSM z component in nT
        tilt : float
            dipole tilt angle in degrees
        f107 : float
            F10.7 index in s.f.u.

        Examples
        --------
        If model currents shall be calculated on the same grid for a range of 
        external conditions, it is faster to do this:
        
        >>> m1 = AMPS(solar_wind_velocity_in_km_per_s, IMF_By_in_nT, IMF_Bz_in_nT, dipole_tilt_in_deg, F107_index)
        >>> # ... current calculations ...
        >>> m1.update_model(new_v, new_By, new_Bz, new_tilt, new_f107)
        >>> # ... new current calcuations ...
        
        than to make a new object:
        
        >>> m2 = AMPS(new_v, new_By, new_Bz, new_tilt, new_f107)
        >>> # ... new current calculations ...
        
        Also note that the inputs are scalars in both cases. It is possible to optimize the calculations significantly
        by allowing the inputs to be arrays. That is not yet implemented.

        """

        if coeff_fn is DEFAULT:
            self.tor_c, self.tor_s, self.pol_c, self.pol_s, self.pol_keys, self.tor_keys = get_model_vectors(v, By, Bz, tilt, f107, coeff_fn = self.coeff_fn)
        else:
            self.tor_c, self.tor_s, self.pol_c, self.pol_s, self.pol_keys, self.tor_keys = get_model_vectors(v, By, Bz, tilt, f107, coeff_fn = coeff_fn)
       



    def _get_vectorgrid(self, **kwargs):
        """ 
        Make grid for plotting vectors

        kwargs are passed to equal_area_grid(...)
        """

        grid = equal_area_grid(dr = self.dr, M0 = self.M0, **kwargs)
        mlt  = grid[1] + grid[2]/2. # shift to the center points of the bins
        mlat = grid[0] + (grid[0][1] - grid[0][0])/2  # shift to the center points of the bins

        mlt  = mlt[ (mlat >= self.minlat) & (mlat <= self.maxlat)]# & (mlat <=60 )]
        mlat = mlat[(mlat >= self.minlat) & (mlat <= self.maxlat)]# & (mlat <= 60)]

        mlat = np.hstack((mlat, -mlat)) # add southern hemisphere points
        mlt  = np.hstack((mlt ,  mlt)) # add southern hemisphere points


        return mlat[:, np.newaxis], mlt[:, np.newaxis] # reshape to column vectors and return


    def _get_scalargrid(self, resolution = 100):
        """ 
        Make grid for calculations of scalar fields 

        Parameters
        ----------
        resolution : int, optional
            resolution in both directions of the scalar field grids (default 100)
        """

        mlat, mlt = map(np.ravel, np.meshgrid(np.linspace(self.minlat , self.maxlat, resolution), np.linspace(-179.9, 179.9, resolution)))
        mlat = np.hstack((mlat, -mlat)) # add southern hemisphere points
        mlt  = np.hstack((mlt ,   mlt)) * 12/180 # add points for southern hemisphere and scale to mlt
        self.scalar_resolution = resolution

        return mlat[:, np.newaxis], mlt[:, np.newaxis] + 12 # reshape to column vectors and return


    def calculate_matrices(self):
        """ 
        Calculate the matrices that are needed to calculate currents and potentials 

        Call this function if and only if the grid has been changed manually
        """

        mlt2r = np.pi/12

        # cos(m * phi) and sin(m * phi):
        self.pol_cosmphi_vector = np.cos(self.m_P * self.vectorgrid[1] * mlt2r)
        self.pol_cosmphi_scalar = np.cos(self.m_P * self.scalargrid[1] * mlt2r)
        self.pol_sinmphi_vector = np.sin(self.m_P * self.vectorgrid[1] * mlt2r)
        self.pol_sinmphi_scalar = np.sin(self.m_P * self.scalargrid[1] * mlt2r)
        self.tor_cosmphi_vector = np.cos(self.m_T * self.vectorgrid[1] * mlt2r)
        self.tor_cosmphi_scalar = np.cos(self.m_T * self.scalargrid[1] * mlt2r)
        self.tor_sinmphi_vector = np.sin(self.m_T * self.vectorgrid[1] * mlt2r)
        self.tor_sinmphi_scalar = np.sin(self.m_T * self.scalargrid[1] * mlt2r)

        self.coslambda_vector = np.cos(self.vectorgrid[0] * np.pi/180)
        self.coslambda_scalar = np.cos(self.scalargrid[0] * np.pi/180)

        # P and dP ( shape  NEQ, NED):
        vector_P, vector_dP = legendre(self.N, self.M, 90 - self.vectorgrid[0])
        scalar_P, scalar_dP = legendre(self.N, self.M, 90 - self.scalargrid[0])

        self.pol_P_vector  =  np.array([vector_P[ key] for key in self.keys_P ]).squeeze().T
        self.pol_dP_vector = -np.array([vector_dP[key] for key in self.keys_P ]).squeeze().T # change sign since we use lat - not colat
        self.pol_P_scalar  =  np.array([scalar_P[ key] for key in self.keys_P ]).squeeze().T
        self.pol_dP_scalar = -np.array([scalar_dP[key] for key in self.keys_P ]).squeeze().T
        self.tor_P_vector  =  np.array([vector_P[ key] for key in self.keys_T ]).squeeze().T
        self.tor_dP_vector = -np.array([vector_dP[key] for key in self.keys_T ]).squeeze().T
        self.tor_P_scalar  =  np.array([scalar_P[ key] for key in self.keys_T ]).squeeze().T
        self.tor_dP_scalar = -np.array([scalar_dP[key] for key in self.keys_T ]).squeeze().T


    def get_toroidal_scalar(self, mlat = DEFAULT, mlt = DEFAULT, grid = False):
        """ 
        Calculate the toroidal scalar values (unit is nT). 

        Parameters
        ----------
        mlat : numpy.ndarray, optional
            array of mlats at which to calculate the toroidal scalar. Will be ignored if mlt is not 
            also specified. If not specified, the calculations will be done using the coords of the 
            `scalargrid` attribute.
        mlt : numpy.ndarray, optional
            array of mlts at which to calculate the toroidal scalar. Will be ignored if mlat is not
            also specified. If not specified, the calculations will be done using the coords of the 
            `scalargrid` attribute.
        grid : bool, optional, default False
            if True, mlat and mlt are interpreted as coordinates in a regular grid. They must be 
            1-dimensional, and the output will have dimensions len(mlat) x len(mlt). If mlat and mlt 
            are not set, this keyword is ignored.

        Returns
        -------
        T : numpy.ndarray
            Toroidal scalar evaluated at self.scalargrid, or, if specified, mlat/mlt 
        """

        if mlat is DEFAULT or mlt is DEFAULT:
            T = (  np.dot(self.tor_P_scalar * self.tor_cosmphi_scalar, self.tor_c)
                 + np.dot(self.tor_P_scalar * self.tor_sinmphi_scalar, self.tor_s) ) 

        else: # calculate at custom coordinates
            if grid:
                assert len(mlat.shape) == len(mlt.shape) == 1 # enforce 1D input arrays

                P, dP = legendre(self.N, self.M, 90 - mlat)
                P  =  np.transpose(np.array([ P[ key] for key in self.keys_T]), (1,2,0)) # (nlat, 1, 257)
                mlt = mlt.reshape(1,-1,1)
                m_T = self.m_T[np.newaxis, ...] # (1, 1, 257)

                cosmphi = np.cos(m_T *  mlt * np.pi/12 ) # (1, nmlt, 257)
                sinmphi = np.sin(m_T *  mlt * np.pi/12 ) # (1, nmlt, 257)

                T = np.dot(P * cosmphi, self.tor_c) + \
                    np.dot(P * sinmphi, self.tor_s)

                T = T.squeeze()

            else:
                shape = mlat.shape

                mlat = mlat.flatten()[:, np.newaxis]
                mlt  = mlt.flatten()[:, np.newaxis]

                P, dP = legendre(self.N, self.M, 90 - mlat)
                P  =  np.array([ P[ key] for key in self.keys_T]).T.squeeze()
                cosmphi   = np.cos(self.m_T *  mlt * np.pi/12 )
                sinmphi   = np.sin(self.m_T *  mlt * np.pi/12 )

                T = np.dot(P * cosmphi, self.tor_c) + \
                    np.dot(P * sinmphi, self.tor_s) 

                T = T.reshape(shape)


        return T


    def get_poloidal_scalar(self, mlat = DEFAULT, mlt = DEFAULT, grid = False):
        """ 
        Calculate the poloidal scalar potential values (unit is microTm).

        Parameters
        ----------
        mlat : numpy.ndarray, optional
            array of mlats at which to calculate the poloidal scalar. Will be ignored if mlt is not 
            also specified. If not specified, the calculations will be done using the coords of the 
            `scalargrid` attribute.
        mlt : numpy.ndarray, optional
            array of mlts at which to calculate the poloidal scalar. Will be ignored if mlat is not
            also specified. If not specified, the calculations will be done using the coords of the 
            `scalargrid` attribute.
        grid : bool, optional, default False
            if True, mlat and mlt are interpreted as coordinates in a regular grid. They must be 
            1-dimensional, and the output will have dimensions len(mlat) x len(mlt). If mlat and mlt 
            are not set, this keyword is ignored.

        Returns
        -------
        V : numpy.ndarray
            Poloidal scalar evalulated at self.scalargrid, or, if specified, mlat/mlt
        """

        rtor = (REFRE / (REFRE + self.height)) ** (self.n_P + 1)

        if mlat is DEFAULT or mlt is DEFAULT:
            V = REFRE * (  np.dot(rtor * self.pol_P_scalar * self.pol_cosmphi_scalar, self.pol_c ) 
                         + np.dot(rtor * self.pol_P_scalar * self.pol_sinmphi_scalar, self.pol_s ) )
        else: # calculate at custom coordinates
            if grid:
                assert len(mlat.shape) == len(mlt.shape) == 1 # enforce 1D input arrays

                P, dP = legendre(self.N, self.M, 90 - mlat)
                P  =  np.transpose(np.array([ P[ key] for key in self.keys_P]), (1,2,0)) # (nlat, 1, 177)
                mlt = mlt.reshape(1,-1,1)
                m_P, n_P = self.m_P[np.newaxis, ...], self.n_P[np.newaxis, ...] # (1, 1, 177)

                cosmphi = np.cos(m_P *  mlt * np.pi/12 ) # (1, nmlt, 177)
                sinmphi = np.sin(m_P *  mlt * np.pi/12 ) # (1, nmlt, 177)

                rtor = (REFRE / (REFRE + self.height)) ** (n_P + 1)

                V = REFRE * (  np.dot(rtor * P * cosmphi, self.pol_c ) 
                             + np.dot(rtor * P * sinmphi, self.pol_s ) )
                V = V.squeeze()

            else:
                shape = mlat.shape

                mlat = mlat.flatten()[:, np.newaxis]
                mlt  = mlt.flatten()[:, np.newaxis]

                P, dP = legendre(self.N, self.M, 90 - mlat)
                P  =  np.array([ P[ key] for key in self.keys_P]).T.squeeze()
                cosmphi   = np.cos(self.m_P *  mlt * np.pi/12 )
                sinmphi   = np.sin(self.m_P *  mlt * np.pi/12 )
                V = REFRE * (  np.dot(rtor * P * cosmphi, self.pol_c ) 
                             + np.dot(rtor * P * sinmphi, self.pol_s ) )
                V = V.reshape(shape)


        return V


    def get_divergence_free_current_function(self, mlat = DEFAULT, mlt = DEFAULT, grid = False):
        """
        Calculate the divergence-free current function

        Isocontours of the divergence-free current function indicates the alignment of the 
        divergence-free part of the horizontal current. Its direction is given by the cross
        product between a vertical vector and the gradient of the divergence-free current function. 
        A fixed amount of current flows between isocontours. The calculations refer to 
        the height chosen upon initialization of the AMPS object (default 110 km). Divergence-free
        current function unit is kA.

        Note
        ----
        The divergence-free current is similar to what is often termed the `equivalent current`, that
        is derived from ground based magnetic field measurements. However, the present divergence-free 
        current is derived from measurements above the ionosphere, and thus it contains signal both from 
        ionospheric currents below low Earth orbit, and from subsurface induced currents. 
        See Laundal et al. (2016) [1]_ where this current is called
        `Psi` for more detail.


        Parameters
        ----------
        mlat : numpy.ndarray, optional
            array of mlats at which to calculate the current. Will be ignored if mlt is not also specified. If 
            not specified, the calculations will be done using the coords of the `scalargrid` attribute.
        mlt : numpy.ndarray, optional
            array of mlts at which to calculate the current. Will be ignored if mlat is not also specified. If 
            not specified, the calculations will be done using the coords of the `scalargrid` attribute.
        grid : bool, optional, default False
            if True, mlat and mlt are interpreted as coordinates in a regular grid. They must be 
            1-dimensional, and the output will have dimensions len(mlat) x len(mlt). If mlat and mlt 
            are not set, this keyword is ignored.

        Returns
        -------
        Psi : numpy.ndarray
            Divergence-free current function evaluated at self.scalargrid, or, if specified, mlat/mlt

        References
        ----------
        .. [1] K. M. Laundal, C. C. Finlay, and N. Olsen, "Sunlight effects on the 3D polar current 
           system determined from low Earth orbit measurements" Earth, Planets and Space, 2016,
           https://doi.org/10.1186/s40623-016-0518-x
        """

        rtor = (REFRE / (REFRE + self.height)) ** (self.n_P + 1.) * (2.*self.n_P + 1.)/self.n_P

        if mlat is DEFAULT or mlt is DEFAULT:
            Psi = - REFRE / MU0 * (  np.dot(rtor * self.pol_P_scalar * self.pol_cosmphi_scalar, self.pol_c ) 
                                   + np.dot(rtor * self.pol_P_scalar * self.pol_sinmphi_scalar, self.pol_s ) ) * 1e-9  # kA
        else: # calculate at custom coordinates
            if grid:
                assert len(mlat.shape) == len(mlt.shape) == 1 # enforce 1D input arrays

                P, dP = legendre(self.N, self.M, 90 - mlat)
                P  =  np.transpose(np.array([ P[ key] for key in self.keys_P]), (1,2,0)) # (nlat, 1, 177)
                mlt = mlt.reshape(1,-1,1)
                m_P, n_P = self.m_P[np.newaxis, ...], self.n_P[np.newaxis, ...] # (1, 1, 177)

                rtor = (REFRE / (REFRE + self.height)) ** (n_P + 1.) * (2.*n_P + 1.)/n_P
 
                cosmphi = np.cos(m_P *  mlt * np.pi/12 ) # (1, nmlt, 177)
                sinmphi = np.sin(m_P *  mlt * np.pi/12 ) # (1, nmlt, 177)

                Psi = - REFRE / MU0 * (  np.dot(rtor * P * cosmphi, self.pol_c ) 
                                       + np.dot(rtor * P * sinmphi, self.pol_s ) ) * 1e-9  # kA
                Psi = Psi.squeeze()
 
            else:
                shape = mlat.shape
                mlat = mlat.flatten()[:, np.newaxis]
                mlt  = mlt.flatten()[:, np.newaxis]

                P, dP = legendre(self.N, self.M, 90 - mlat)
                P  =  np.array([ P[ key] for key in self.keys_P]).T.squeeze()
                cosmphi   = np.cos(self.m_P *  mlt * np.pi/12 )
                sinmphi   = np.sin(self.m_P *  mlt * np.pi/12 )
                Psi = - REFRE / MU0 * (  np.dot(rtor * P * cosmphi, self.pol_c ) 
                                       + np.dot(rtor * P * sinmphi, self.pol_s ) ) * 1e-9  # kA
                Psi = Psi.reshape(shape)


        
        return Psi


    def get_upward_current(self, mlat = DEFAULT, mlt = DEFAULT, grid = False):
        """
        Calculate the upward current (unit is microAmps per square meter). The 
        calculations refer to the height chosen upon initialization of the 
        AMPS object (default 110 km).

        Parameters
        ----------
        mlat : numpy.ndarray, optional
            array of mlats at which to calculate the current. Will be ignored if mlt is not also specified. If 
            not specified, the calculations will be done using the coords of the `scalargrid` attribute.
        mlt : numpy.ndarray, optional
            array of mlts at which to calculate the current. Will be ignored if mlat is not also specified. If 
            not specified, the calculations will be done using the coords of the `scalargrid` attribute.
        grid : bool, optional, default False
            if True, mlat and mlt are interpreted as coordinates in a regular grid. They must be 
            1-dimensional, and the output will have dimensions len(mlat) x len(mlt). If mlat and mlt 
            are not set, this keyword is ignored.

        Returns
        -------
        Ju : numpy.ndarray
            Upward current evaulated at self.scalargrid, or, if specified, mlat/mlt
        """

        if mlat is DEFAULT or mlt is DEFAULT:
            Ju = -1e-6/(MU0 * (REFRE + self.height) ) * (   np.dot(self.n_T * (self.n_T + 1) * self.tor_P_scalar * self.tor_cosmphi_scalar, self.tor_c) 
                                                          + np.dot(self.n_T * (self.n_T + 1) * self.tor_P_scalar * self.tor_sinmphi_scalar, self.tor_s) )

        else: # calculate at custom coordinates
            if grid:
                assert len(mlat.shape) == len(mlt.shape) == 1 # enforce 1D input arrays

                P, dP = legendre(self.N, self.M, 90 - mlat)
                P  =  np.transpose(np.array([ P[ key] for key in self.keys_T]), (1,2,0)) # (nlat, 1, 257)
                mlt = mlt.reshape(1,-1,1)
                n_T, m_T = self.n_T[np.newaxis, ...], self.m_T[np.newaxis, ...] # (1, 1, 257)
                
                cosmphi = np.cos(m_T *  mlt * np.pi/12 ) # (1, nmlt, 257)
                sinmphi = np.sin(m_T *  mlt * np.pi/12 ) # (1, nmlt, 257)
                Ju = -1e-6/(MU0 * (REFRE + self.height) ) * ( np.dot(n_T * (n_T + 1) * P * cosmphi, self.tor_c) 
                                                          +   np.dot(n_T * (n_T + 1) * P * sinmphi, self.tor_s) )

                Ju = Ju.squeeze() # (nmlat, nmlt), transpose of original  
    
            else:    
                shape = mlat.shape
                mlat = mlat.flatten()[:, np.newaxis]
                mlt  = mlt.flatten()[:, np.newaxis]

                P, dP = legendre(self.N, self.M, 90 - mlat)
                P  =  np.array([ P[ key] for key in self.keys_T]).T.squeeze()
                cosmphi   = np.cos(self.m_T *  mlt * np.pi/12 )
                sinmphi   = np.sin(self.m_T *  mlt * np.pi/12 )
                Ju = -1e-6/(MU0 * (REFRE + self.height) ) * ( np.dot(self.n_T * (self.n_T + 1) * P * cosmphi, self.tor_c) 
                                                          +   np.dot(self.n_T * (self.n_T + 1) * P * sinmphi, self.tor_s) )
                Ju = Ju.reshape(shape)

        return Ju


    def get_curl_free_current_potential(self, mlat = DEFAULT, mlt = DEFAULT, grid = False):
        """ 
        Calculate the curl-free current potential (unit is kA). The curl-free
        current potential is a scalar alpha which relates to the curl-free part
        of the horizontal current by J_{cf} = grad(alpha). The calculations 
        refer to the height chosen upon initialization of the AMPS object (default 
        110 km). 

        Parameters
        ----------
        mlat : numpy.ndarray, optional
            array of mlats at which to calculate the current. Will be ignored if mlt is not also specified. If 
            not specified, the calculations will be done using the coords of the `scalargrid` attribute.
        mlt : numpy.ndarray, optional
            array of mlts at which to calculate the current. Will be ignored if mlat is not also specified. If 
            not specified, the calculations will be done using the coords of the `scalargrid` attribute.
        grid : bool, optional, default False
            if True, mlat and mlt are interpreted as coordinates in a regular grid. They must be 
            1-dimensional, and the output will have dimensions len(mlat) x len(mlt). If mlat and mlt 
            are not set, this keyword is ignored.

        Returns
        -------
        alpha : numpy.ndarray
            Curl-free current potential evaulated at self.scalargrid, or, if specified, mlat/mlt

        """

        if mlat is DEFAULT or mlt is DEFAULT:
            alpha = -(REFRE + self.height) / MU0 * (   np.dot(self.tor_P_scalar * self.tor_cosmphi_scalar, self.tor_c) 
                                                     + np.dot(self.tor_P_scalar * self.tor_sinmphi_scalar, self.tor_s) ) * 1e-9

        else: # calculate at custom coordinates
            if grid:
                assert len(mlat.shape) == len(mlt.shape) == 1 # enforce 1D input arrays

                P, dP = legendre(self.N, self.M, 90 - mlat)
                P  =  np.transpose(np.array([ P[ key] for key in self.keys_T]), (1,2,0)) # (nlat, 1, 257)
                mlt = mlt.reshape(1,-1,1)
                n_T, m_T = self.n_T[np.newaxis, ...], self.m_T[np.newaxis, ...] # (1, 1, 257)

                
                cosmphi = np.cos(m_T *  mlt * np.pi/12 ) # (1, nmlt, 257)
                sinmphi = np.sin(m_T *  mlt * np.pi/12 ) # (1, nmlt, 257)

                alpha = -(REFRE + self.height) / MU0 * (   np.dot(P * cosmphi, self.tor_c) 
                                                         + np.dot(P * sinmphi, self.tor_s) ) * 1e-9
                alpha = alpha.squeeze()
                

            else:
                shape = mlat.shape
                mlat = mlat.flatten()[:, np.newaxis]
                mlt  = mlt.flatten()[:, np.newaxis]

                P, dP = legendre(self.N, self.M, 90 - mlat)
                P  =  np.array([ P[ key] for key in self.keys_T]).T.squeeze()
                cosmphi   = np.cos(self.m_T *  mlt * np.pi/12 )
                sinmphi   = np.sin(self.m_T *  mlt * np.pi/12 )

                alpha = -(REFRE + self.height) / MU0 * (   np.dot(P * cosmphi, self.tor_c) 
                                                         + np.dot(P * sinmphi, self.tor_s) ) * 1e-9
                alpha = alpha.reshape(shape)


        return alpha


    def get_divergence_free_current(self, mlat = DEFAULT, mlt = DEFAULT, grid = False):
        """ 
        Calculate the divergence-free part of the horizontal current, in units of mA/m.
        The calculations refer to the height chosen upon initialization of the AMPS 
        object (default 110 km).

        Parameters
        ----------
        mlat : numpy.ndarray, optional
            array of mlats at which to calculate the current. Will be ignored if mlt is not also specified. If 
            not specified, the calculations will be done using the coords of the `vectorgrid` attribute.
        mlt : numpy.ndarray, optional
            array of mlts at which to calculate the current. Will be ignored if mlat is not also specified. If 
            not specified, the calculations will be done using the coords of the `vectorgrid` attribute.
        grid : bool, optional, default False
            if True, mlat and mlt are interpreted as coordinates in a regular grid. They must be 
            1-dimensional, and the output will have dimensions len(mlat) x len(mlt). If mlat and mlt 
            are not set, this keyword is ignored.


        Return
        ------
        jdf_eastward : numpy.ndarray, float
            eastward component of the divergence-free current evalulated at the coordinates given by the `vectorgrid` attribute
        jdf_northward : numpy.ndarray, float
            northward component of the divergence-free current evalulated at the coordinates given by the `vectorgrid` attribute

        See Also
        --------
        get_curl_free_current : Calculate curl-free part of the current
        get_total_current : Calculate total horizontal current

        """
        
        rtor = (REFRE / (REFRE + self.height)) ** (self.n_P + 2.) * (2.*self.n_P + 1.)/self.n_P /MU0 * 1e-6

        if mlat is DEFAULT or mlt is DEFAULT:
            east  =    (  np.dot(rtor * self.pol_dP_vector * self.pol_cosmphi_vector, self.pol_c) 
                        + np.dot(rtor * self.pol_dP_vector * self.pol_sinmphi_vector, self.pol_s) )
    
            north =  - (  np.dot(rtor * self.pol_P_vector * self.m_P * self.pol_cosmphi_vector, self.pol_s)
                        - np.dot(rtor * self.pol_P_vector * self.m_P * self.pol_sinmphi_vector, self.pol_c) ) / self.coslambda_vector

            return east.flatten(), north.flatten()


        else: # calculate at custom mlat, mlt
            if grid:
                assert len(mlat.shape) == len(mlt.shape) == 1 # enforce 1D input arrays

                P, dP = legendre(self.N, self.M, 90 - mlat)
                P  =  np.transpose(np.array([ P[ key] for key in self.keys_P]), (1,2,0)) # (nlat, 1, 177)
                dP = -np.transpose(np.array([dP[ key] for key in self.keys_P]), (1,2,0)) # (nlat, 1, 177)
                mlt = mlt.reshape(1, -1, 1)
                mlat = mlat.reshape(-1, 1, 1)
                m_P, n_P = self.m_P[np.newaxis, ...], self.n_P[np.newaxis, ...] # (1, 1, 177)

                rtor = (REFRE / (REFRE + self.height)) ** (n_P + 2.) * (2.*n_P + 1.)/n_P /MU0 * 1e-6

                coslambda = np.cos(      mlat * np.pi/180) # (nmlat, 1   , 177)
                cosmphi   = np.cos(m_P * mlt  * np.pi/12 ) # (1    , nmlt, 177)
                sinmphi   = np.sin(m_P * mlt  * np.pi/12 ) # (1    , nmlt, 177)

                east  = (  np.dot(rtor * dP       * cosmphi, self.pol_c) \
                         + np.dot(rtor * dP       * sinmphi, self.pol_s) )
                north = (- np.dot(rtor *  P * m_P * cosmphi, self.pol_s) \
                         + np.dot(rtor *  P * m_P * sinmphi, self.pol_c) ) / coslambda

                return east.squeeze(), north.squeeze()


            else:
                shape = mlat.shape
                mlat = mlat.flatten()[:, np.newaxis]
                mlt  = mlt.flatten()[:, np.newaxis]

                P, dP = legendre(self.N, self.M, 90 - mlat)
                P  =  np.array([ P[ key] for key in self.keys_P]).T.squeeze()
                dP = -np.array([dP[ key] for key in self.keys_P]).T.squeeze()
                cosmphi   = np.cos(self.m_P *  mlt * np.pi/12 )
                sinmphi   = np.sin(self.m_P *  mlt * np.pi/12 )
                coslambda = np.cos(           mlat * np.pi/180)

                east  = (  np.dot(rtor * dP            * cosmphi, self.pol_c) \
                         + np.dot(rtor * dP            * sinmphi, self.pol_s) )
                north = (- np.dot(rtor *  P * self.m_P * cosmphi, self.pol_s) \
                         + np.dot(rtor *  P * self.m_P * sinmphi, self.pol_c) ) / coslambda

                return east.reshape(shape), north.reshape(shape)



    def get_curl_free_current(self, mlat = DEFAULT, mlt = DEFAULT, grid = False):
        """ 
        Calculate the curl-free part of the horizontal current, in units of mA/m.
        The calculations refer to the height chosen upon initialization of the AMPS 
        object (default 110 km).


        Parameters
        ----------
        mlat : numpy.ndarray, optional
            array of mlats at which to calculate the current. Will be ignored if mlt is not also specified. If 
            not specified, the calculations will be done using the coords of the `vectorgrid` attribute.
        mlt : numpy.ndarray, optional
            array of mlts at which to calculate the current. Will be ignored if mlat is not also specified. If 
            not specified, the calculations will be done using the coords of the `vectorgrid` attribute.
        grid : bool, optional, default False
            if True, mlat and mlt are interpreted as coordinates in a regular grid. They must be 
            1-dimensional, and the output will have dimensions len(mlat) x len(mlt). If mlat and mlt 
            are not set, this keyword is ignored.


        Return
        ------
        jcf_eastward : numpy.ndarray, float
            eastward component of the curl-free current evalulated at the coordinates given by the `vectorgrid` attribute
        jcf_northward : numpy.ndarray, float
            northward component of the curl-free current evalulated at the coordinates given by the `vectorgrid` attribute

        See Also
        --------
        get_divergence_free_current : Calculate divergence-free part of the horizontal current
        get_total_current : Calculate total horizontal current
        """

        rtor = -1.e-6/MU0

        if mlat is DEFAULT or mlt is DEFAULT:
            east = rtor * (    np.dot(self.tor_P_vector * self.m_T * self.tor_cosmphi_vector, self.tor_s )
                             - np.dot(self.tor_P_vector * self.m_T * self.tor_sinmphi_vector, self.tor_c )) / self.coslambda_vector
    
            north = rtor * (   np.dot(self.tor_dP_vector * self.tor_cosmphi_vector, self.tor_c)
                             + np.dot(self.tor_dP_vector * self.tor_sinmphi_vector, self.tor_s))

            return east.flatten(), north.flatten()

        else: # calculate at custom mlat, mlt
            if grid:
                assert len(mlat.shape) == len(mlt.shape) == 1 # enforce 1D input arrays

                P, dP = legendre(self.N, self.M, 90 - mlat)
                P  =  np.transpose(np.array([ P[ key] for key in self.keys_T]), (1,2,0)) # (nlat, 1, 257)
                dP = -np.transpose(np.array([dP[ key] for key in self.keys_T]), (1,2,0)) # (nlat, 1, 257)
                mlt = mlt.reshape(1,-1,1)
                mlat = mlat.reshape(-1, 1, 1)
                n_T, m_T = self.n_T[np.newaxis, ...], self.m_T[np.newaxis, ...] # (1, 1, 257)

                coslambda = np.cos(      mlat * np.pi/180) # (nmlat, 1   , 177)                
                cosmphi   = np.cos(m_T *  mlt * np.pi/12 ) # (1, nmlt, 257)
                sinmphi   = np.sin(m_T *  mlt * np.pi/12 ) # (1, nmlt, 257)

                east  = (  np.dot(rtor *  P * m_T * cosmphi, self.tor_s) \
                         - np.dot(rtor *  P * m_T * sinmphi, self.tor_c) ) / coslambda
                north = (  np.dot(rtor * dP       * cosmphi, self.tor_c) \
                         + np.dot(rtor * dP       * sinmphi, self.tor_s) ) 

                return east.squeeze(), north.squeeze()

            else:
                shape = mlat.shape
                mlat = mlat.flatten()[:, np.newaxis]
                mlt  = mlt.flatten()[ :, np.newaxis]

                P, dP = legendre(self.N, self.M, 90 - mlat)
                P  =  np.array([ P[ key] for key in self.keys_T]).T.squeeze()
                dP = -np.array([dP[ key] for key in self.keys_T]).T.squeeze()
                cosmphi   = np.cos(self.m_T *  mlt * np.pi/12 )
                sinmphi   = np.sin(self.m_T *  mlt * np.pi/12 )
                coslambda = np.cos(           mlat * np.pi/180)

                east  = (  np.dot(rtor *  P * self.m_T * cosmphi, self.tor_s) \
                         - np.dot(rtor *  P * self.m_T * sinmphi, self.tor_c) ) / coslambda
                north = (  np.dot(rtor * dP            * cosmphi, self.tor_c) \
                         + np.dot(rtor * dP            * sinmphi, self.tor_s) ) 
                return east.reshape(shape), north.reshape(shape)




    def get_total_current(self, mlat = DEFAULT, mlt = DEFAULT, grid = False):
        """ 
        Calculate the total horizontal current, in units of mA/m. This is calculated as 
        the sum of the curl-free and divergence-free parts. The calculations refer to 
        the height chosen upon initialization of the AMPS object (default 110 km).

        Parameters
        ----------
        mlat : numpy.ndarray, optional
            array of mlats at which to calculate the current. Will be ignored if mlt is not also specified. If 
            not specified, the calculations will be done using the coords of the `vectorgrid` attribute.
        mlt : numpy.ndarray, optional
            array of mlts at which to calculate the current. Will be ignored if mlat is not also specified. If 
            not specified, the calculations will be done using the coords of the `vectorgrid` attribute.
        grid : bool, optional, default False
            if True, mlat and mlt are interpreted as coordinates in a regular grid. They must be 
            1-dimensional, and the output will have dimensions len(mlat) x len(mlt). If mlat and mlt 
            are not set, this keyword is ignored.


        Return
        ------
        j_eastward : numpy.ndarray, float
            eastward component of the horizontal current evalulated at the coordinates given by the `vectorgrid` attribute
        j_northward : numpy.ndarray, float
            northward component of the horizontal current evalulated at the coordinates given by the `vectorgrid` attribute


        See Also
        --------
        get_divergence_free_current : Calculate divergence-free part of the horizontal current
        get_curl_free_current : Calculate curl-free part of the horizontal current
        """
        
        return [x + y for x, y in zip(self.get_curl_free_current(      mlat = mlat, mlt = mlt, grid = grid), 
                                      self.get_divergence_free_current(mlat = mlat, mlt = mlt, grid = grid))]


    def get_total_current_magnitude(self):
        """ 
        Calculate the total horizontal current density magnitude, in units of mA/m. 
        This is calculated as the sum of the curl-free and divergence-free parts. 
        The calculations refer to the height chosen upon initialization of the AMPS 
        object (default 110 km). The calculations are performed on the coordinates of
        self.scalargrid. This is useful for making contour plots of the horizontal
        current density magnitude, and faster than calculating the magnitude 
        from the output of get_total_current


        Return
        ------
        j : numpy.ndarray, float
            horizontal current density magnitude, evalulated at the coordinates given by the `scalargrid` attribute

        See Also
        --------
        get_total_current : Calculate total current density vector components
        """

        # curl-free part:
        C = -1.e-6/MU0

        je_cf = C * (   np.dot(self.tor_P_scalar * self.m_T * self.tor_cosmphi_scalar, self.tor_s )
                         - np.dot(self.tor_P_scalar * self.m_T * self.tor_sinmphi_scalar, self.tor_c )) / self.coslambda_scalar

        jn_cf = C * (   np.dot(self.tor_dP_scalar * self.tor_cosmphi_scalar, self.tor_c)
                         + np.dot(self.tor_dP_scalar * self.tor_sinmphi_scalar, self.tor_s))

        # divergence-free part:
        rtor = (REFRE / (REFRE + self.height)) ** (self.n_P + 2.) * (2.*self.n_P + 1.)/self.n_P /MU0 * 1e-6

        je_df =    (  np.dot(rtor * self.pol_dP_scalar * self.pol_cosmphi_scalar, self.pol_c) 
                    + np.dot(rtor * self.pol_dP_scalar * self.pol_sinmphi_scalar, self.pol_s) )

        jn_df =  - (  np.dot(rtor * self.pol_P_scalar * self.m_P * self.pol_cosmphi_scalar, self.pol_s)
                    - np.dot(rtor * self.pol_P_scalar * self.m_P * self.pol_sinmphi_scalar, self.pol_c) ) / self.coslambda_scalar

        # return magntitude of vector sum:
        return np.sqrt((je_cf + je_df)**2 + (jn_cf + jn_df)**2)


    def get_integrated_upward_current(self):
        """ 
        Calculate the integrated upward and downward current, poleward of `minlat`,
        in units of MA. 

        Note
        ----
        This calculation uses the scalargrid attribute. By default this is a *regular* grid, 
        with coordinates from north and south tiled side-by-side (equal number of coords,
        with north first). If scalargrid has been changed, and has a different structure,
        this function will return wrong values.

        Return
        ------
        J_up_n : float
            Total upward current in the northern hemisphere
        J_down_n : float
            Total downward current in the northern hemisphere
        J_up_s : float
            Total upward current in the southern hemisphere
        J_down_s : float
            Total downward current in the southern hemisphere
        """

        ju = self.get_upward_current() * 1e-6 # unit A/m^2

        # get surface area element in each cell:
        mlat, mlt = self.scalargrid
        mlt_sorted = np.sort(np.unique(mlt))
        mltres = (mlt_sorted[1] - mlt_sorted[0]) * np.pi / 12
        mlat_sorted = np.sort(np.unique(mlat))
        mlatres = (mlat_sorted[1] - mlat_sorted[0]) * np.pi / 180
        R = (REFRE + self.height) * 1e3  # radius in meters
        dS = R**2 * np.cos(mlat * np.pi/180) * mlatres * mltres


        J_n, J_s = np.split(dS * ju * 1e-6, 2) # convert to MA and split to north and south

        #      J_up_north            J_down_north          J_up_south            J_down_south
        return np.sum(J_n[J_n > 0]), np.sum(J_n[J_n < 0]), np.sum(J_s[J_s > 0]), np.sum(J_s[J_s < 0])


    def get_ground_Beqd(self, height = 0):
        """ 
        Calculate ground magnetic field perturbations in the QD east direction, in units of nT. 


        Note
        ----
        These calculations are made by assuming that the divergende-free current function calculated
        with the AMPS model correspond to the equivalent current function of an external 
        magnetic potential, as described by Chapman & Bartels 1940 [2]_. Induced components are 
        thus ignored. The height of the current function also becomes important when propagating
        the model values to the ground. 

        Also note that the output parameters will be QD components, and that they can be converted
        to geographic by use of QD base vectors [3]_

        This function is not optimized for calculating long time series of model ground
        magnetic field perturbations, although it is possible to use for that.

        Parameters
        ----------
        height : float, optional
            height, in km, where the magnetic field is evaluated. Must be less than self.height, which 
            is the height of the current. Default is 0 (ground).

        Return
        ------
        dB_east : numpy.ndarray
            Eastward component of the magnetic field disturbance on ground

        References
        ----------
        .. [2] S. Chapman & J. Bartels "Geomagnetism Vol 2" Oxford University Press 1940
        
        .. [3] A. D. Richmond, "Ionospheric Electrodynamics Using Magnetic Apex Coordinates", 
           Journal of geomagnetism and geoelectricity Vol. 47, 1995, http://doi.org/10.5636/jgg.47.191


        See Also
        --------
        get_ground_Bnqd : Calculate ground perturbation in northward qd direction on scalargrid
        get_ground_Buqd : Calculate ground perturbation in upward qd direction on scalargrid
        get_ground_perturbation: Calculate ground perturbation in east/north qd direction
        """

        rr   = REFRE / (REFRE + self.height) # ratio of current radius to earth radius
        hh   = REFRE + height

        G_ce    = rr ** (2 * self.n_P + 1) * (hh / REFRE) ** self.n_P * (self.n_P + 1.) / self.n_P * self.pol_P_scalar * self.m_P / self.coslambda_scalar
        G = np.hstack((-G_ce * self.pol_sinmphi_scalar, G_ce * self.pol_cosmphi_scalar))

        return G.dot(np.vstack((self.pol_c, self.pol_s)))


    def get_ground_Bnqd(self, height = 0):
        """ 
        Calculate ground magnetic field perturbations in the QD north direction, in units of nT. 


        Note
        ----
        These calculations are made by assuming that the divergende-free current function calculated
        with the AMPS model correspond to the equivalent current function of an external 
        magnetic potential, as described by Chapman & Bartels 1940 [2]_. Induced components are 
        thus ignored. The height of the current function also becomes important when propagating
        the model values to the ground. 

        Also note that the output parameters will be QD components, and that they can be converted
        to geographic by use of QD base vectors [3]_

        This function is not optimized for calculating long time series of model ground
        magnetic field perturbations, although it is possible to use for that.

        Parameters
        ----------
        height : float, optional
            height, in km, where the magnetic field is evaluated. Must be less than self.height, which 
            is the height of the current. Default is 0 (ground).

        Return
        ------
        dB_north : numpy.ndarray
            Northward component of the magnetic field disturbance on ground

        References
        ----------
        .. [2] S. Chapman & J. Bartels "Geomagnetism Vol 2" Oxford University Press 1940
        
        .. [3] A. D. Richmond, "Ionospheric Electrodynamics Using Magnetic Apex Coordinates", 
           Journal of geomagnetism and geoelectricity Vol. 47, 1995, http://doi.org/10.5636/jgg.47.191


        See Also
        --------
        get_ground_Beqd : Calculate ground perturbation in easthward qd direction on scalargrid
        get_ground_Buqd : Calculate ground perturbation in upward qd direction on scalargrid
        get_ground_perturbation: Calculate ground perturbation in east/north qd direction
        """

        rr   = REFRE / (REFRE + self.height) # ratio of current radius to earth radius
        hh   = REFRE + height

        G_cn    = rr ** (2 * self.n_P + 1) * (hh / REFRE) ** self.n_P * (self.n_P + 1.) / self.n_P * self.pol_dP_scalar 
        G = np.hstack(( G_cn * self.pol_cosmphi_scalar, G_cn * self.pol_sinmphi_scalar))

        return G.dot(np.vstack((self.pol_c, self.pol_s)))


    def get_ground_Buqd(self, height = 0.):
        """ 
        Calculate ground magnetic field perturbations in the QD up direction, in units of nT. 


        Note
        ----
        These calculations are made by assuming that the divergende-free current function calculated
        with the AMPS model correspond to the equivalent current function of an external 
        magnetic potential, as described by Chapman & Bartels 1940 [2]_. Induced components are 
        thus ignored. The height of the current function also becomes important when propagating
        the model values to the ground. 

        Also note that the output parameters will be QD components, and that they can be converted
        to geographic by use of QD base vectors [3]_

        This function is not optimized for calculating long time series of model ground
        magnetic field perturbations, although it is possible to use for that.

        Parameters
        ----------
        height : float, optional
            height, in km, where the magnetic field is evaluated. Must be less than self.height, which 
            is the height of the current. Default is 0 (ground).

        Return
        ------
        dB_up : numpy.ndarray
            Upward component of the magnetic field disturbance on ground

        References
        ----------
        .. [2] S. Chapman & J. Bartels "Geomagnetism Vol 2" Oxford University Press 1940
        
        .. [3] A. D. Richmond, "Ionospheric Electrodynamics Using Magnetic Apex Coordinates", 
           Journal of geomagnetism and geoelectricity Vol. 47, 1995, http://doi.org/10.5636/jgg.47.191


        See Also
        --------
        get_ground_Beqd : Calculate ground perturbation in easthward qd direction on scalargrid
        get_ground_Bnqd : Calculate ground perturbation in northward qd direction on scalargrid
        get_ground_perturbation: Calculate ground perturbation in east/north qd direction
        """

        rr   = REFRE / (REFRE + self.height) # ratio of current radius to earth radius
        hh   = REFRE + height

        G_ce = rr ** (2 * self.n_P + 1) * (hh / REFRE) ** (self.n_P - 1) * (self.n_P + 1.) * self.pol_P_scalar 
        G = np.hstack(( G_ce * self.pol_cosmphi_scalar, G_ce * self.pol_sinmphi_scalar))

        return G.dot(np.vstack((self.pol_c, self.pol_s)))


    def get_ground_perturbation(self, mlat = DEFAULT, mlt = DEFAULT, height = 0):
        """ 
        Calculate magnetic field perturbations on ground, in units of nT, that corresponds 
        to the divergence-free current function.

        Parameters
        ----------
        mlat : numpy.ndarray, float, optional
            magnetic latitude of the output. The array shape will not be preserved, and 
            the results will be returned as a 1-dimensional array. Default value is 
            from self.vectorgrid
        mlt : numpy.ndarray, float, optional
            magnetic local time of the output. The array shape will not be preserved, and 
            the results will be returned as a 1-dimensional array. Default value is 
            from self.vectorgrid
        height: numpy.ndarray, optional
            geodetic height at which the field should be evalulated. Should be < current height
            set at initialization. Default 0 (ground)

        Note
        ----
        These calculations are made by assuming that the divergende-free current function calculated
        with the AMPS model correspond to the equivalent current function of an external 
        magnetic potential, as described by Chapman & Bartels 1940 [2]_. Induced components are 
        thus ignored. The height of the current function also becomes important when propagating
        the model values to the ground. 

        Also note that the output parameters will be QD components, and that they can be converted
        to geographic by use of QD base vectors [3]_

        This function is not optimized for calculating long time series of model ground
        magnetic field perturbations, although it is possible to use for that.


        Return
        ------
        dB_east : numpy.ndarray
            Eastward component of the magnetic field disturbance on ground
        dB_north : numpy.ndarray
            Northward component of the magnetic field disurubance on ground

        References
        ----------
        .. [2] S. Chapman & J. Bartels "Geomagnetism Vol 2" Oxford University Press 1940
        
        .. [3] A. D. Richmond, "Ionospheric Electrodynamics Using Magnetic Apex Coordinates", 
           Journal of geomagnetism and geoelectricity Vol. 47, 1995, http://doi.org/10.5636/jgg.47.191

        """

        # if mlat and mlt are not given, call function again with vectorgrid
        if mlat is DEFAULT or mlt is DEFAULT:
            return self.get_ground_perturbation(self.vectorgrid[0], self.vectorgrid[1], height = height)

        mlt  = mlt. flatten()[:, np.newaxis]
        mlat = mlat.flatten()[:, np.newaxis]
        rr   = REFRE / (REFRE + self.height) # ratio of current radius to earth radius
        hh   = REFRE + height

        m = self.m_P
        n = self.n_P


        P, dP = legendre(self.N, self.M, 90 - mlat)
        P  = np.array([ P[ key] for key in self.keys_P]).T.squeeze()
        dP = np.array([dP[ key] for key in self.keys_P]).T.squeeze()
        cosmphi = np.cos(m * mlt * np.pi/12)
        sinmphi = np.sin(m * mlt * np.pi/12)

        # G matrix for north component
        G_cn   =  - rr ** (2 * n + 1) * (hh / REFRE) ** n * (n + 1.)/n * dP
        Gn     =  np.hstack(( G_cn * cosmphi, G_cn * sinmphi))
        
        # G matrix for east component
        G_ce   =  rr ** (2 * n + 1) * (hh / REFRE) ** n * (n + 1.)/n * P * m / np.cos(mlat * np.pi / 180)
        Ge     =  np.hstack((-G_ce * sinmphi, G_ce * cosmphi))

        model = np.vstack((self.pol_c, self.pol_s))

        return Ge.dot(model), Gn.dot(model)


    def get_AE_indices(self):
        """ 
        Calculate model synthetic auroral electrojet (AE) indices: AL and AU. The unit is nT

        Note
        ----
        Here, AL and AU are defined as the lower/upper envelope curves for the northward component
        of the ground magnetic field perturbation that is equivalent with the divergence-free current,
        evaluated on `scalargrid`. Thus all the caveats for the `get_ground_perturbation()` function
        applies to these calculations as well. An additional caveat is that we have in principle
        perfect coverage with the model, while the true AE indices are derived using a small set of
        magnetometers in the auroral zone. The model values are also based on QD northward component,
        instead of the "H component", which is used in the official measured AL index. It is possible
        to calculate model AE indices that are more directly comparable to the measured indices.

        Returns
        -------
        AL_n : float
            Model AL index in the northerm hemisphere
        AL_s : float
            Model AL index in the southern hemisphere
        AU_n : float
            Model AU index in the northerm hemisphere
        AU_s : float
            Model AU index in the southern hemisphere
        """

        rr   = REFRE / (REFRE + self.height) # ratio of current radius to earth radius
        n = self.n_P

        dP = self.pol_dP_scalar

        G_cn   =  rr ** (2 * n + 1) * (n + 1.)/n * dP
        Gn     =  np.hstack(( G_cn * self.pol_cosmphi_scalar, G_cn * self.pol_sinmphi_scalar))

        Bn     = Gn.dot(np.vstack((self.pol_c, self.pol_s)))
        Bn_n, Bn_s = np.split(Bn, 2)

        return Bn_n.min(), Bn_s.min(), Bn_n.max(), Bn_s.max()


    def plot_currents(self, vector_scale = 200):
        """ 
        Create a summary plot of the current fields

        Parameters
        ----------
        vector_scale : optional
            Current vector lengths will be shown relative to a template. This parameter determines
            the magnitude of that template, in mA/m. Default is 200 mA/m

        Examples
        --------
        >>> # initialize by supplying a set of external conditions:
        >>> m = AMPS(300, # solar wind velocity in km/s 
                     -4, # IMF By in nT
                     -3, # IMF Bz in nT
                     20, # dipole tilt angle in degrees
                     150) # F10.7 index in s.f.u.
        >>> # make summary plot:
        >>> m.plot_currents()

        """

        # get the grids:
        mlats, mlts = self.plotgrid_scalar
        mlatv, mltv = self.plotgrid_vector

        # set up figure and polar coordinate plots:
        plt.figure(figsize = (15, 7))
        pax_n = Polarsubplot(plt.subplot2grid((1, 15), (0,  0), colspan = 7), minlat = self.minlat, linestyle = ':', linewidth = .3, color = 'lightgrey')
        pax_s = Polarsubplot(plt.subplot2grid((1, 15), (0,  7), colspan = 7), minlat = self.minlat, linestyle = ':', linewidth = .3, color = 'lightgrey')
        pax_c = plt.subplot2grid((1, 150), (0, 149), colspan = 1)
        
        # labels
        pax_n.writeMLTlabels(mlat = self.minlat, size = 14)
        pax_s.writeMLTlabels(mlat = self.minlat, size = 14)
        pax_n.write(self.minlat, 3,    str(self.minlat) + r'$^\circ$' , ha = 'left', va = 'top', size = 14)
        pax_s.write(self.minlat, 3,    r'$-$' + str(self.minlat) + '$^\circ$', ha = 'left', va = 'top', size = 14)
        pax_n.write(self.minlat-5, 12, r'North' , ha = 'center', va = 'center', size = 18)
        pax_s.write(self.minlat-5, 12, r'South' , ha = 'center', va = 'center', size = 18)

        # calculate and plot FAC
        Jun, Jus = np.split(self.get_upward_current(), 2)
        faclevels = np.r_[-.925:.926:.05]
        pax_n.contourf(mlats, mlts, Jun, levels = faclevels, cmap = plt.cm.bwr, extend = 'both')
        pax_s.contourf(mlats, mlts, Jus, levels = faclevels, cmap = plt.cm.bwr, extend = 'both')

        # Total horizontal
        j_e, j_n = self.get_total_current()
        nn, ns = np.split(j_n, 2)
        en, es = np.split(j_e, 2)
        pax_n.featherplot(mlatv, mltv, nn , en, SCALE = vector_scale, markersize = 10, unit = 'mA/m', linewidth = '.5', color = 'gray', markercolor = 'grey')
        pax_s.featherplot(mlatv, mltv, -ns, es, SCALE = vector_scale, markersize = 10, unit = None  , linewidth = '.5', color = 'gray', markercolor = 'grey')


        # colorbar
        pax_c.contourf(np.vstack((np.zeros_like(faclevels), np.ones_like(faclevels))), 
                       np.vstack((faclevels, faclevels)), 
                       np.vstack((faclevels, faclevels)), 
                       levels = faclevels, cmap = plt.cm.bwr)
        pax_c.set_xticks([])
        pax_c.set_ylabel(r'downward    $\mu$A/m$^2$      upward', size = 18)
        pax_c.yaxis.set_label_position("right")
        pax_c.yaxis.tick_right()

        # print AL index values and integrated up/down currents
        AL_n, AL_s, AU_n, AU_s = self.get_AE_indices()
        ju_n, jd_n, ju_s, jd_s = self.get_integrated_upward_current()

        pax_n.ax.text(pax_n.ax.get_xlim()[0], pax_n.ax.get_ylim()[0], 
                      'AL: \t${AL_n:+}$ nT\nAU: \t${AU_n:+}$ nT\n $\int j_{uparrow:}$:\t ${jn_up:+.1f}$ MA\n $\int j_{downarrow:}$:\t ${jn_down:+.1f}$ MA'.format(AL_n = int(np.round(AL_n)), AU_n = int(np.round(AU_n)), jn_up = ju_n, jn_down = jd_n, uparrow = r'\uparrow',downarrow = r'\downarrow'), ha = 'left', va = 'bottom', size = 12)
        pax_s.ax.text(pax_s.ax.get_xlim()[0], pax_s.ax.get_ylim()[0], 
                      'AL: \t${AL_s:+}$ nT\nAU: \t${AU_s:+}$ nT\n $\int j_{uparrow:}$:\t ${js_up:+.1f}$ MA\n $\int j_{downarrow:}$:\t ${js_down:+.1f}$ MA'.format(AL_s = int(np.round(AL_s)), AU_s = int(np.round(AU_s)), js_up = ju_s, js_down = jd_s, uparrow = r'\uparrow',downarrow = r'\downarrow'), ha = 'left', va = 'bottom', size = 12)


        plt.subplots_adjust(hspace = 0, wspace = 0.4, left = .05, right = .935, bottom = .05, top = .945)
        plt.show()


def get_B_space(glat, glon, height, time, v, By, Bz, tilt, f107, epoch = 2015., h_R = 110., chunksize = 15000, coeff_fn = default_coeff_fn):
    """ Calculate model magnetic field in space 

    This function uses dask to parallelize computations. That means that it is quite
    fast and that the memory consumption will not explode unless `chunksize` is too large.

    Parameters
    ----------
    glat : array_like
        array of geographic latitudes (degrees)
    glon : array_like
        array of geographic longitudes (degrees)
    height : array_like
        array of geodetic heights (km)
    time : array_like
        list/array of datetimes, needed to calculate magnetic local time
    v : array_like
        array of solar wind velocities in GSM/GSE x direction (km/s)
    By : array_like
        array of solar wind By values (nT)
    Bz : array_like
        array of solar wind Bz values (nT)
    tilt : array_like
        array of dipole tilt angles (degrees)
    f107 : array_like
        array of F10.7 index values (SFU)
    epoch : float, optional
        epoch (year) used in conversion to magnetic coordinates with the IGRF. Default = 2015.
    h_R : float, optional
        reference height (km) used when calculating modified apex coordinates. Default = 110.
    chunksize : int, optional
        the input arrays will be split in chunks in order to parallelize
        computations. Larger chunks consumes more memory, but might be faster. Default is 15000.
    coeff_fn: str, optional
        file name of model coefficients - must be in format produced by model_vector_to_txt.py
        (default is latest version)


    Returns
    -------
    Be : array_like
        array of model magnetic field (nT) in geodetic eastward direction 
        (same dimension as input)
    Bn : array_like
        array of model magnetic field (nT) in geodetic northward direction 
        (same dimension as input)
    Bu : array_like
        array of model magnetic field (nT) in geodetic upward direction 
        (same dimension as input)



    Note
    ----
    Array inputs should have the same dimensions.

    """

    # TODO: ADD CHECKS ON INPUT (?)

    m_matrix       = get_m_matrix(coeff_fn)
    NT, MT, NV, MV = get_truncation_levels(coeff_fn)

    # number of equations
    neq = m_matrix.shape[0]

    # turn coordinates/times into dask arrays
    glat   = da.from_array(glat  , chunks = chunksize)
    glon   = da.from_array(glon  , chunks = chunksize)
    time   = da.from_array(time  , chunks = chunksize)
    height = da.from_array(height, chunks = chunksize)

    # get G0 matrix - but first make a wrapper that only takes dask arrays as input
    _getG0 = lambda la, lo, t, h: getG0(la, lo, t, h, epoch = epoch, h_R = h_R, NT = NT, MT = MT, NV = NV, MV = MV)

    # use that wrapper to calculate G0 for each block
    G0 = da.map_blocks(_getG0, glat, glon, height, time, chunks = (3*chunksize, neq), new_axis = 1, dtype = np.float64)

    # get a matrix with columns that are 19 unscaled magnetic field terms at the given coords:
    B_matrix  = G0.dot( m_matrix ).compute()

    # the rows of B_matrix now correspond to (east, north, up, east, north, up, ...) and must be
    # reorganized so that we have only three large partitions: (east, north, up). Split and recombine:
    B_chunks = [B_matrix[i : (i + 3*chunksize)] for i in range(0, B_matrix.shape[0], 3 * chunksize)]
    B_e = np.vstack(tuple([B[                  :     B.shape[0]//3] for B in B_chunks]))
    B_n = np.vstack(tuple([B[    B.shape[0]//3 : 2 * B.shape[0]//3] for B in B_chunks]))
    B_r = np.vstack(tuple([B[2 * B.shape[0]//3 :                  ] for B in B_chunks]))
    Bs  = np.vstack((B_e, B_n, B_r)).T

    # prepare the scales (external parameters)
    By, Bz, v, tilt, f107 = map(lambda x: x.flatten(), [By, Bz, v, tilt, f107]) # flatten input
    ca = np.arctan2(By, Bz)
    epsilon = np.abs(v)**(4/3.) * np.sqrt(By**2 + Bz**2)**(2/3.) * (np.sin(ca/2)**(8))**(1/3.) / 1000 # Newell coupling           
    tau     = np.abs(v)**(4/3.) * np.sqrt(By**2 + Bz**2)**(2/3.) * (np.cos(ca/2)**(8))**(1/3.) / 1000 # Newell coupling - inverse 

    # make a dict of the 19 external parameters (flat arrays)
    external_params = {0  : np.ones_like(ca)           ,        # 'const'             
                       1  : 1              * np.sin(ca),        # 'sinca'             
                       2  : 1              * np.cos(ca),        # 'cosca'             
                       3  : epsilon                    ,        # 'epsilon'           
                       4  : epsilon        * np.sin(ca),        # 'epsilon_sinca'     
                       5  : epsilon        * np.cos(ca),        # 'epsilon_cosca'     
                       6  : tilt                       ,        # 'tilt'              
                       7  : tilt           * np.sin(ca),        # 'tilt_sinca'        
                       8  : tilt           * np.cos(ca),        # 'tilt_cosca'        
                       9  : tilt * epsilon             ,        # 'tilt_epsilon'      
                       10 : tilt * epsilon * np.sin(ca),        # 'tilt_epsilon_sinca'
                       11 : tilt * epsilon * np.cos(ca),        # 'tilt_epsilon_cosca'
                       12 : tau                        ,        # 'tau'               
                       13 : tau            * np.sin(ca),        # 'tau_sinca'         
                       14 : tau            * np.cos(ca),        # 'tau_cosca'         
                       15 : tilt * tau                 ,        # 'tilt_tau'          
                       16 : tilt * tau     * np.sin(ca),        # 'tilt_tau_sinca'    
                       17 : tilt * tau     * np.cos(ca),        # 'tilt_tau_cosca'    
                       18 : f107                        }       # 'f107'

    # scale the 19 magnetic field terms, and add (the scales are tiled once for each component)
    B = reduce(lambda x, y: x+y, [Bs[i] * np.tile(external_params[i], 3) for i in range(19)])


    # the resulting array will be stacked Be, Bn, Bu components. Return the partions
    return np.split(B, 3)


def get_B_ground(qdlat, mlt, height, v, By, Bz, tilt, f107, current_height = 110, epsilon_multiplier = 1., chunksize = 25000, coeff_fn = default_coeff_fn):
    """ Calculate model magnetic field on ground 
    
    This function uses dask to parallelize computations. That means that it is quite
    fast and that the memory consumption will not explode unless `chunksize` is too large.

    
    Parameters
    ----------
    qdlat : array_like or float
        quasi-dipole latitude, in degrees. Can be either a scalar (float), or
        an array with an equal number of elements as mlt
    mlt : array_like
        array of magnetic local times (hours)
    height : float
        geodetic height, in km (0 <= height <= current_height)
    v : array_like
        array of solar wind velocities in GSM/GSE x direction (km/s)
    By : array_like
        array of solar wind By values (nT)
    Bz : array_like
        array of solar wind Bz values (nT)
    tilt : array_like
        array of dipole tilt angles (degrees)
    f107 : array_like
        array of F10.7 index values (SFU)
    current_height : float, optional
        height (km) of the current sheet. Default is 110.
    epsilon_multiplier: float, optional
        multiplier for the epsilon parameter. Default is 1.
    chunksize : int
        the input arrays will be split in chunks in order to parallelize
        computations. Larger chunks consumes more memory, but might be faster. Default is 25000.
    coeff_fn: str, optional
        file name of model coefficients - must be in format produced by model_vector_to_txt.py
        (default is latest version)


    Returns
    -------
    Bqphi : array_like
        magnetic field in quasi-dipole eastward direction
    Bqlambda : array_like
        magnetic field in quasi-dipole northward direction
    Bqr : array_like
        magnetic field in upward direction. See notes

    Note
    ----
    We assume that there are no induced currents. The error in this assumption will be larger
    for the radial component than for the horizontal components

    Array inputs should have the same dimensions.
    """

    m_matrix_pol = get_m_matrix_pol(coeff_fn)
    _, _, N, M   = get_truncation_levels(coeff_fn)

    # number of equations
    neq = m_matrix_pol.shape[0]

    # convert input to dask arrays - qdlat is converted to np.float32 to make sure it has the flatten function
    qdlat = da.from_array(np.float32(qdlat).flatten(), chunks = chunksize)
    mlt   = da.from_array(mlt.flatten()  , chunks = chunksize)

    # get G0 matrix - but first make a wrapper that only takes dask arrays as input
    _getG0 = lambda x, y: get_ground_field_G0(x, y, height, current_height, N = N, M = M)

    # use that wrapper to calculate G0 for each block
    G0 = da.map_blocks(_getG0, qdlat, mlt, chunks = (3*chunksize, neq), new_axis = 1)

    # get a matrix with columns that are 19 unscaled magnetic field terms at the given coords:
    B_matrix  = G0.dot(  m_matrix_pol ).compute()

    # the rows of B_matrix now correspond to (east, north, up, east, north, up, ...) and must be
    # reorganized so that we have only three large partitions: (east, north, up). Split and recombine:
    B_chunks = [B_matrix[i : (i + 3*chunksize)] for i in range(0, B_matrix.shape[0], 3 * chunksize)]
    B_e = np.vstack(tuple([B[                  :     B.shape[0]//3] for B in B_chunks]))
    B_n = np.vstack(tuple([B[    B.shape[0]//3 : 2 * B.shape[0]//3] for B in B_chunks]))
    B_r = np.vstack(tuple([B[2 * B.shape[0]//3 :                  ] for B in B_chunks]))
    Bs  = np.vstack((B_e, B_n, B_r)).T

    # prepare the scales (external parameters)
    By, Bz, v, tilt, f107 = map(lambda x: x.flatten(), [By, Bz, v, tilt, f107]) # flatten input
    ca = np.arctan2(By, Bz)
    epsilon = np.abs(v)**(4/3.) * np.sqrt(By**2 + Bz**2)**(2/3.) * (np.sin(ca/2)**(8))**(1/3.) / 1000 * epsilon_multiplier # Newell coupling           
    tau     = np.abs(v)**(4/3.) * np.sqrt(By**2 + Bz**2)**(2/3.) * (np.cos(ca/2)**(8))**(1/3.) / 1000 # Newell coupling - inverse 

    # make a dict of the 19 external parameters (flat arrays)
    external_params = {0  : np.ones_like(ca)           ,        #'const'             
                       1  : 1              * np.sin(ca),        #'sinca'             
                       2  : 1              * np.cos(ca),        #'cosca'             
                       3  : epsilon                    ,        #'epsilon'           
                       4  : epsilon        * np.sin(ca),        #'epsilon_sinca'     
                       5  : epsilon        * np.cos(ca),        #'epsilon_cosca'     
                       6  : tilt                       ,        #'tilt'              
                       7  : tilt           * np.sin(ca),        #'tilt_sinca'        
                       8  : tilt           * np.cos(ca),        #'tilt_cosca'        
                       9  : tilt * epsilon             ,        #'tilt_epsilon'      
                       10 : tilt * epsilon * np.sin(ca),        #'tilt_epsilon_sinca'
                       11 : tilt * epsilon * np.cos(ca),        #'tilt_epsilon_cosca'
                       12 : tau                        ,        #'tau'               
                       13 : tau            * np.sin(ca),        #'tau_sinca'         
                       14 : tau            * np.cos(ca),        #'tau_cosca'         
                       15 : tilt * tau                 ,        #'tilt_tau'          
                       16 : tilt * tau     * np.sin(ca),        #'tilt_tau_sinca'    
                       17 : tilt * tau     * np.cos(ca),        #'tilt_tau_cosca'    
                       18 : f107                        }       #'f107'

    # scale the 19 magnetic field terms, and add (the scales are tiled once for each component)
    B = reduce(lambda x, y: x+y, [Bs[i] * np.tile(external_params[i], 3) for i in range(19)])


    # the resulting array will be stacked Be, Bn, Bu components. Return the partions
    return np.split(B, 3)

