""" Python interface for the Average Magnetic field and Polar current System (AMPS) model

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
||||||||  2) Calculate the model magnetic field in space, along a trajetory, 
||todo||     provided a time series of external parameters. This is done through
||||||||     the get_magnetic_field(...) function. The magnetic field will be provided in geographic coordinates
||||||||  3) Calculate the model magnetic field in space, along a trajetory, 
||todo||     provided a time series of external parameters. This is done through
||||||||     the get_ground_perturbation(...) function. 



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

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from plot_utils import equalAreaGrid, Polarsubplot
from sh_utils import get_legendre, SHkeys
from model_utils import get_model_vectors
from matplotlib import rc


rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

MU0   = 4*np.pi*1e-7 # Permeability constant
REFRE = 6371.2 # Reference radius used in geomagnetic modeling


class AMPS(object):
    """Calculate and plot maps of the model Average Magnetic field and Polar current System (AMPS)



    Attributes:
        tor_c      -- vector of cos term coefficents in the toroidal field expansion
        tor_s      -- vector of sin term coefficents in the toroidal field expansion
        pol_c      -- vector of cos term coefficents in the poloidal field expansion
        pol_s      -- vector of sin term coefficents in the poloidal field expansion
        keys_P     -- list of spherical harmonic wave number pairs (n,m) corresponding to elements of pol_c and pol_s 
        keys_T     -- list of spherical harmonic wave number pairs (n,m) corresponding to elements of tor_c and tor_s 
        vectorgrid -- grid used to calculate and plot vector fields
        scalargrid -- grid used to calculate and plot scalar fields
                       
                       The grid formats are as follows (see also example below):
                       (np.hstack((mlat_north, mlat_south)), np.hstack((mlt_north, mlt_south)))
                       
                       The grids can be changed directly, but member function calculate_matrices() 
                       must then be called for the change to take effect. Also the grid format
                       described above should be used.


    Example usage:
        m = AMPS(solar_wind_velocity_in_km_per_s, IMF_By_in_nT, IMF_Bz_in_nT, dipole_tilt_in_deg, F107_index)
        
        # make summary plot:
        m.plot_currents()
        
        # extract map of field-aligned currents in north and south:
        Jun, Jus = m.get_upward_current_function()

        # Jus.flatten() will be evaluated at the following coordinates:
        mlat, mlt = np.split(m.scalargrid[0], 2)[1], np.split(m.scalargrid[1], 2)[1]

        # extract map of total height-integrated horizontal currents:
        j_eastward_north, j_eastward_south, j_northward_north, j_northward_south = m.get_total_current()

        # j_eastward_north will be a flat array evalulated at the following coordinates:
        mlat, mlt = np.split(m.vectorgrid[0], 2)[0], np.split(m.vectorgrid[1], 2)[0]

        # update model vectors (tor_c, tor_s, etc.) without recalculating the other matrices:
        m.update_model(new_v, new_By, new_Bz, new_tilt, new_F107)

    """

    def __init__(self, v, By, Bz, tilt, F107, minlat = 60, maxlat = 89.99, height = 110., dr = 2, M0 = 4, resolution = 100):
        """ __init__ function for AMPS model class

            Args:
                v          -- solar wind velocity in km/s (scalar/float)
                By         -- IMF GSM y component in nT (scalar/float)
                Bz         -- IMF GSM z component in nT (scalar/float)
                tilt       -- dipole tilt angle in degrees (scalar/float)
                F107       -- F10.7 index in s.f.u. (scalar/float)
                
                minlat     -- optional: low latitude boundary of grids  (default 60)
                maxlat     -- optional: high latitude boundary of grids (default 89.99)
                height     -- optional: altitude of the ionospheric currents (default 110 km)
                dr         -- optional: latitudinal spacing between equal area grid points (default 2 degrees)
                M0         -- optional: number of grid points in the most poleward circle of equal area grid points (default 4)
                resolution -- optional: resolution in both directions of the scalar field grids (default 100)
        """

        self.tor_c, self.tor_s, self.pol_c, self.pol_s, self.pol_keys, self.tor_keys = get_model_vectors(v, By, Bz, tilt, F107)

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

        self.vectorgrid = self.vectorgrid()
        self.scalargrid = self.scalargrid(resolution = resolution)
        self.calculate_matrices()

    def update_model(self, v, By, Bz, tilt, F107):
        """update the model vectors without updating all the other matrices

           Note:
               If model currents shall be calculated on the same grid for a range of 
               external conditions, it is faster to do this:
                   m1 = AMPS(solar_wind_velocity_in_km_per_s, IMF_By_in_nT, IMF_Bz_in_nT, dipole_tilt_in_deg, F107_index)
                   # ... current calculations ...
                   m1.update_model(new_v, new_By, new_Bz, new_tilt, new_F107)
                   # ... new current calcuations ...
               than to make a new object:
                   m2 = AMPS(new_v, new_By, new_Bz, new_tilt, new_F107)
                   # ... new current calculations ...

                Also note that the inputs are scalars in both cases. It is possible to optimize the calculations significantly
                by allowing the inputs to be arrays. That is not yet implemented. 


           Args:
               v    -- solar wind velocity in km/s (scalar/float)
               By   -- IMF GSM y component in nT (scalar/float)
               Bz   -- IMF GSM z component in nT (scalar/float)
               tilt -- dipole tilt angle in degrees (scalar/float)
               F107 -- F10.7 index in s.f.u. (scalar/float)

        """
        
        self.tor_c, self.tor_s, self.pol_c, self.pol_s, self.pol_keys, self.tor_keys = get_model_vectors(v, By, Bz, tilt, F107)



    def vectorgrid(self, **kwargs):
        """ make grid for plotting vectors - using an equal area grid scheme for this

            kwargs are passed to equalAreaGrid
        """

        grid = equalAreaGrid(dr = self.dr, M0 = self.M0, **kwargs)
        mlt  = grid[1] + grid[2]/2. # shift to the center points of the bins
        mlat = grid[0] + (grid[0][1] - grid[0][0])/2  # shift to the center points of the bins

        mlt  = mlt[ (mlat >= self.minlat) & (mlat <= self.maxlat)]# & (mlat <=60 )]
        mlat = mlat[(mlat >= self.minlat) & (mlat <= self.maxlat)]# & (mlat <= 60)]

        mlat = np.hstack((mlat, -mlat)) # add southern hemisphere points
        mlt  = np.hstack((mlt ,  mlt)) # add southern hemisphere points


        return mlat[:, np.newaxis], mlt[:, np.newaxis] # reshape to column vectors and return


    def scalargrid(self, resolution = 100):
        """ make grid for calculations of scalar fields """

        mlat, mlt = map(np.ravel, np.meshgrid(np.linspace(self.minlat , self.maxlat, resolution), np.linspace(-179.9, 179.9, resolution)))
        mlat = np.hstack((mlat, -mlat)) # add southern hemisphere points
        mlt  = np.hstack((mlt ,   mlt)) * 12/180 # add points for southern hemisphere and scale to mlt
        self.scalar_resolution = resolution

        return mlat[:, np.newaxis], mlt[:, np.newaxis] + 12 # reshape to column vectors and return

    def calculate_matrices(self):
        """ calculated the matrices that are needed to calculate currents and potentials """

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

        # P and dP ( shape  NEQ, NED):
        vector_P, vector_dP = get_legendre(self.N, self.M, 90 - self.vectorgrid[0])
        scalar_P, scalar_dP = get_legendre(self.N, self.M, 90 - self.scalargrid[0])

        self.pol_P_vector  =  np.array([vector_P[ key] for key in self.keys_P ]).squeeze().T
        self.pol_dP_vector = -np.array([vector_dP[key] for key in self.keys_P ]).squeeze().T # change sign since we use lat - not colat
        self.pol_P_scalar  =  np.array([scalar_P[ key] for key in self.keys_P ]).squeeze().T
        self.pol_dP_scalar = -np.array([scalar_dP[key] for key in self.keys_P ]).squeeze().T
        self.tor_P_vector  =  np.array([vector_P[ key] for key in self.keys_T ]).squeeze().T
        self.tor_dP_vector = -np.array([vector_dP[key] for key in self.keys_T ]).squeeze().T
        self.tor_P_scalar  =  np.array([scalar_P[ key] for key in self.keys_T ]).squeeze().T
        self.tor_dP_scalar = -np.array([scalar_dP[key] for key in self.keys_T ]).squeeze().T



    def get_toroidal_potential(self):
        """ calculate toroidal potential on the scalar grid 

            returns tuple with values for north and south
        """

        T = (  np.dot(self.tor_P_scalar * self.tor_cosmphi_scalar, self.tor_c)
             + np.dot(self.tor_P_scalar * self.tor_sinmphi_scalar, self.tor_s) ) 

        _reshape = lambda x: np.reshape(x, (self.scalar_resolution, self.scalar_resolution))
        return map( _reshape, np.split(T, 2)) # north, south 


    def get_poloidal_potential(self):
        """ calculate poloidal potential on the scalar grid

            returns tuple with values for north and south
        """
        rtor = (REFRE / (REFRE + self.height)) ** (self.n_P + 1)
        P = REFRE * (  np.dot(rtor * self.pol_P_scalar * self.pol_cosmphi_scalar, self.pol_c ) 
                     + np.dot(rtor * self.pol_P_scalar * self.pol_sinmphi_scalar, self.pol_s ) )

        _reshape = lambda x: np.reshape(x, (self.scalar_resolution, self.scalar_resolution))
        return map( _reshape, np.split(P, 2)) # north, south 


    def get_equivalent_current_function(self):
        """ calculate equivalent current function from the poloidal potential -  unit kA

            returns tuple with values for north and south
        """

        rtor = (REFRE / (REFRE + self.height)) ** (self.n_P + 1.) * (2.*self.n_P + 1.)/self.n_P
        Psi = - REFRE / MU0 * (  np.dot(rtor * self.pol_P_scalar * self.pol_cosmphi_scalar, self.pol_c ) 
                               + np.dot(rtor * self.pol_P_scalar * self.pol_sinmphi_scalar, self.pol_s ) ) * 1e-9  # kA
        
        _reshape = lambda x: np.reshape(x, (self.scalar_resolution, self.scalar_resolution))
        return map( _reshape, np.split(Psi, 2)) # north, south 

    def get_equivalent_current_laplacian(self):
        """ calculate del^2(psi)

            returns tuple with values for north and south
        """
        
        rtor = (REFRE/(REFRE + self.height))**(self.n_P + 2)
        Ju = 1e-6/(MU0 * (REFRE + self.height) ) * (   np.dot((self.n_P + 1)* (2*self.n_P + 1) * rtor * self.pol_P_scalar * self.pol_cosmphi_scalar, self.pol_c) 
                                                     + np.dot((self.n_P + 1)* (2*self.n_P + 1) * rtor * self.pol_P_scalar * self.pol_sinmphi_scalar, self.pol_s) )

        _reshape = lambda x: np.reshape(x, (self.scalar_resolution, self.scalar_resolution))
        return map( _reshape, np.split(Ju, 2)) # north, south 

    def get_upward_current_function(self):
        """ calculate upward current function from toroidal  potential, in uA/m^2

            returns tuple with values for north and south
        """
        
        Ju = -1e-6/(MU0 * (REFRE + self.height) ) * (   np.dot(self.n_T * (self.n_T + 1) * self.tor_P_scalar * self.tor_cosmphi_scalar, self.tor_c) 
                                                      + np.dot(self.n_T * (self.n_T + 1) * self.tor_P_scalar * self.tor_sinmphi_scalar, self.tor_s) )

        _reshape = lambda x: np.reshape(x, (self.scalar_resolution, self.scalar_resolution))
        return map( _reshape, np.split(Ju, 2)) # north, south 


    def get_curl_free_current_potential(self):
        """ get the curl-free current scalar potential from toroidal potential             

            returns tuple with values for north and south

            in kA
        """
        alpha = (REFRE + self.height) / MU0 * (   np.dot(self.tor_P_scalar * self.tor_cosmphi_scalar, self.tor_c) 
                                                + np.dot(self.tor_P_scalar * self.tor_sinmphi_scalar, self.tor_s) ) * 1e-9

        _reshape = lambda x: np.reshape(x, (self.scalar_resolution, self.scalar_resolution))
        return map( _reshape, np.split(alpha, 2)) # north, south 



    def get_divergence_free_current(self):
        """ get divergence-free current vectors on vector grid 


            This is calculated as k cross grad(equivalent current function)

            Return values are east_n, east_s, north_n, north_s, corresponding to the 
            east and north vector components in the northern and southern hemispheres 

            Calculated at REFRE + height

            unit in mA/m

        """
        
        rtor = (REFRE / (REFRE + self.height)) ** (self.n_P + 2.) * (2.*self.n_P + 1.)/self.n_P /MU0 * 1e-6

        east  =    (  np.dot(rtor * self.pol_dP_vector * self.pol_cosmphi_vector, self.pol_c) 
                    + np.dot(rtor * self.pol_dP_vector * self.pol_sinmphi_vector, self.pol_s) )

        north =  - (  np.dot(rtor * self.pol_P_vector * self.m_P * self.pol_cosmphi_vector, self.pol_s)
                    - np.dot(rtor * self.pol_P_vector * self.m_P * self.pol_sinmphi_vector, self.pol_c) ) / self.coslambda_vector

        e_n, e_s = map(np.ravel, np.split(east, 2))
        n_n, n_s = map(np.ravel, np.split(north, 2))

        return e_n, e_s, n_n, n_s


    def get_curl_free_current(self):
        """ get curl-free current vectors from toroidal potential

            unit is mA/m

        """
        rtor = -1.e-6/MU0

        east = rtor * (    np.dot(self.tor_P_vector * self.m_T * self.tor_cosmphi_vector, self.tor_s )
                         - np.dot(self.tor_P_vector * self.m_T * self.tor_sinmphi_vector, self.tor_c )) / self.coslambda_vector

        north = rtor * (   np.dot(self.tor_dP_vector * self.tor_cosmphi_vector, self.tor_c)
                         + np.dot(self.tor_dP_vector * self.tor_sinmphi_vector, self.tor_s))


        e_n, e_s = map(np.ravel, np.split(east , 2))
        n_n, n_s = map(np.ravel, np.split(north, 2))

        return e_n, e_s, n_n, n_s


    def get_total_current(self):
        """ get total current vectors by summing curl-free and divergence-free parts 

            return sum of curl free and divergence free currents

            unit in mA/m
        """
        
        return [x + y for x, y in zip(self.get_curl_free_current(), self.get_divergence_free_current())]

    def get_integrated_bc(self):
        """ integrate birkeland current poleward of minlat
            return (in MA), J_up_north, J_down_north, J_up_south, J_down_south
        """

        jun, jus = self.get_upward_current_function()
        jun, jus = jun * 1e-6, jus * 1e-6 # convert to A/m^2

        # get surface area element in each cell:
        mlat, mlt = np.split(self.scalargrid[0], 2)[0], np.split(self.scalargrid[1], 2)[0]
        mlat, mlt = mlat.reshape((self.scalar_resolution, self.scalar_resolution)), mlt.reshape((self.scalar_resolution, self.scalar_resolution))
        mltres  = (mlt[1] - mlt[0])[0] * np.pi/12
        mlatres = (mlat[:, 1] - mlat[:, 0])[0] * np.pi/180
        R = (REFRE + self.height) * 1e3  # radius in meters
        dS = R**2 * np.cos(mlat * np.pi/180) * mlatres * mltres


        J_n = dS * jun * 1e-6 # convert to MA
        J_s = dS * jus * 1e-6 # 

        #      J_up_north            J_down_north          J_up_south            J_down_south
        return np.sum(J_n[J_n > 0]), np.sum(J_n[J_n < 0]), np.sum(J_s[J_s > 0]), np.sum(J_s[J_s < 0])

    def get_ground_perturbation(self, mlat, mlt):
        """ return ground perturbation at mlat, mlt

            return values are east, north, with same shape as mlat and mlt
            The assumption is that the equivalent current function corresponds to an 
            external magnetic potential as in Chapman and Bartels (and Laundal et al. 2016)

            (to calculate long time series of ground perturbations, there are faster ways than this function)
        """

        mlt  = mlt. flatten()[:, np.newaxis]
        mlat = mlat.flatten()[:, np.newaxis]
        rr   = (REFRE + self.height) / REFRE # ratio of current radius to earth radius

        m = self.m_P
        n = self.n_P


        P, dP = get_legendre(self.N, self.M, 90 - mlat)
        P  = np.array([ P[ key] for key in self.keys_P]).T.squeeze()
        dP = np.array([dP[ key] for key in self.keys_P]).T.squeeze()
        cosmphi = np.cos(m * mlt * np.pi/12)
        sinmphi = np.sin(m * mlt * np.pi/12)

        G_cn   =  - rr ** (2 * n + 1) * (n + 1.)/n * dP
        Gn     =  np.hstack(( G_cn * cosmphi, G_cn * sinmphi))
        
        # G matrix for east component
        G_ce   =  rr ** (2 * n + 1) * (n + 1.)/n * P * m / np.cos(mlat * np.pi / 180)
        Ge     =  np.hstack((-G_ce * sinmphi, G_ce * cosmphi))

        model = np.vstack((self.pol_c, self.pol_s))

        return Ge.dot(model), Gn.dot(model)


    def get_AE_indices(self):
        """ calculate model AE indices, AL and AU by calculating the (QD) northward component on a uniform grid, and
            returning the minima (AL) and maxima (AU) in both hemispheres

            return: AL_n, AL_s, AU_n, AU_s
        """

        rr   = (REFRE + self.height) / REFRE # ratio of current radius to earth radius
        m = self.m_P
        n = self.n_P

        dP = self.pol_dP_scalar

        G_cn   =  rr ** (2 * n + 1) * (n + 1.)/n * dP
        Gn     =  np.hstack(( G_cn * self.pol_cosmphi_scalar, G_cn * self.pol_sinmphi_scalar))

        Bn     = Gn.dot(np.vstack((self.pol_c, self.pol_s)))
        Bn_n, Bn_s = np.split(Bn, 2)

        return Bn_n.min(), Bn_s.min(), Bn_n.max(), Bn_s.max()



    def plot_currents(self, VECTOR_SCALE = 200):
        """ plot all the current fields 
        """

        mlats = np.split(self.scalargrid[0], 2)[0].reshape((self.scalar_resolution, self.scalar_resolution))
        mlts  = np.split(self.scalargrid[1], 2)[0].reshape((self.scalar_resolution, self.scalar_resolution))
        mlatv = np.split(self.vectorgrid[0], 2)[0]
        mltv  = np.split(self.vectorgrid[1], 2)[0]

        fig = plt.figure(figsize = (24, 12), facecolor = 'white')
        axes = [plt.subplot2grid((101, 4), (50*(i//4), i % 4), colspan = 1, rowspan = 50) for i in range(8)]


        # get Polarsubplot objects:
        paxes = map(lambda x: Polarsubplot(x, minlat = self.minlat, linestyle = ':', color = 'grey'), axes)

        # FAC
        Jun, Jus = self.get_upward_current_function()
        faclevels = np.linspace(-.55, .55, 12)
        paxes[0].contourf(mlats, mlts, Jun, levels = faclevels, cmap = plt.cm.bwr_r, extend = 'both')
        paxes[4].contourf(mlats, mlts, Jus, levels = faclevels, cmap = plt.cm.bwr_r, extend = 'both')

        paxes[0].writeMLTlabels(mlat = self.minlat, size = 16)
        paxes[4].writeMLTlabels(mlat = self.minlat, size = 16)
        paxes[0].write(self.minlat, 3,    str(self.minlat) + r'$^\circ$' , ha = 'left', va = 'top', size = 18)
        paxes[4].write(self.minlat, 3,    r'$-$' + str(self.minlat) + '$^\circ$', ha = 'left', va = 'top', size = 18)
        paxes[0].write(self.minlat-5, 18, r'North' , ha = 'right', va = 'center', rotation = 90, size = 18)
        paxes[4].write(self.minlat-5, 18, r'South' , ha = 'right', va = 'center', rotation = 90, size = 18)
        paxes[0].write(self.minlat-5, 12, r'$J_u$ [$\mu$A/m$^2$]' , ha = 'center', va = 'bottom', size = 18)


        # Curl-free horizontal
        alphan, alphas = self.get_curl_free_current_potential()
        alphan -= (alphan.min() + (alphan.max() - alphan.min())/2)
        alphas -= (alphas.min() + (alphas.max() - alphas.min())/2)
        paxes[1].contour(mlats, mlts, alphan, levels = np.r_[alphan.min():alphan.max():30], colors = 'black', linewidths = .5)
        paxes[5].contour(mlats, mlts, alphas, levels = np.r_[alphas.min():alphas.max():30], colors = 'black', linewidths = .5)

        en, es, nn, ns = self.get_curl_free_current()
        paxes[1].featherplot(mlatv, mltv, nn , en, SCALE = VECTOR_SCALE, markersize = 2, unit = 'mA/m')
        paxes[5].featherplot(mlatv, mltv, -ns, es, SCALE = VECTOR_SCALE, markersize = 2, unit = 'mA/m')

        paxes[1].write(self.minlat-5, 12, r'$\alpha$ and $\mathbf{j}_{cf} = \nabla\alpha$, where $\nabla^2\alpha = - J_u$' , ha = 'center', va = 'bottom', size = 18)

        # Divergence-free horizontal
        Psin, Psis = self.get_equivalent_current_function()
        paxes[2].contour(mlats, mlts, Psin, levels = np.r_[Psin.min():Psin.max():30], colors = 'black', linewidths = .5)
        paxes[6].contour(mlats, mlts, Psis, levels = np.r_[Psis.min():Psis.max():30], colors = 'black', linewidths = .5)

        en, es, nn, ns = self.get_divergence_free_current()
        paxes[2].featherplot(mlatv, mltv, nn , en, SCALE = VECTOR_SCALE, markersize = 2, unit = 'mA/m')
        paxes[6].featherplot(mlatv, mltv, -ns, es, SCALE = VECTOR_SCALE, markersize = 2, unit = 'mA/m')
        paxes[2].write(self.minlat-5, 12, r'$\Psi$ and $\mathbf{j}_{df} = \mathbf{k}\times\nabla\Psi$' , ha = 'center', va = 'bottom', size = 18)

        # Total horizontal
        en, es, nn, ns = self.get_total_current()
        paxes[3].featherplot(mlatv, mltv, nn , en, SCALE = VECTOR_SCALE, markersize = 2, unit = 'mA/m')
        paxes[7].featherplot(mlatv, mltv, -ns, es, SCALE = VECTOR_SCALE, markersize = 2, unit = 'mA/m')

        paxes[3].write(self.minlat-5, 12, r'$\mathbf{j} = \mathbf{j}_{df} + \mathbf{j}_{cf}$' , ha = 'center', va = 'bottom', size = 18)


        # colorbar
        cbar = plt.subplot2grid((101, 4), (100, 0))
        cbar.contourf(np.vstack((faclevels, faclevels)), np.vstack((np.zeros_like(faclevels), np.ones_like(faclevels))), np.vstack((faclevels, faclevels)), levels = faclevels, cmap = plt.cm.bwr)
        cbar.set_yticks([])
        cbar.set_xlabel('downward    $\hspace{3cm}\mu$A/m$^2\hspace{3cm}$      upward', size = 18)

        plt.subplots_adjust(hspace = 0, wspace = 0, left = .05, right = .95, bottom = .05, top = .95)
        plt.show()
