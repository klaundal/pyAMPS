from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from plot_utils import grids, polarsubplot
from sh_utils import get_legendre, SHkeys

MU0   = 4*pi*1e-7 # Permeability constant
REFRE = 6371.2 # Reference radius used in geomagnetic modeling


class AMPS(object):
    def __init__(self, tor_c, tor_s, iota_c, iota_s, minlat = 60, maxlat = 90.-.01, height = 110., dr = 2, M0 = 4, resolution = 100, global_map = False):
        """ initialize the model with SH coefficients in the form of pandas dataseries, 
            indexed by (n, m)

            tor_c and tor_s are the toroidal coefficients for the cos and sin terms, respectively
            iota_c and iota_s are the poloidal coefficients for the cos and sin terms, respectively
        """
        self.height = height

        self.tor_c = tor_c.dropna().values.flatten()[:, np.newaxis]
        self.tor_s = tor_s.dropna().values.flatten()[:, np.newaxis]
        self.pol_c = iota_c.dropna().values.flatten()[:, np.newaxis]
        self.pol_s = iota_s.dropna().values.flatten()[:, np.newaxis]

        self.dr = dr
        self.M0 = M0


        assert (len(self.pol_s) == len(self.pol_c)) and (len(self.pol_s) == len(self.pol_c))

        self.minlat = minlat
        self.maxlat = maxlat

        self.global_map = global_map

        self.keys_P = [c for c in iota_c.dropna().index]
        self.keys_T = [c for c in tor_c.dropna().index]
        self.m_P = np.array(self.keys_P).T[1][np.newaxis, :]
        self.m_T = np.array(self.keys_T).T[1][np.newaxis, :]
        self.n_P = np.array(self.keys_P).T[0][np.newaxis, :]
        self.n_T = np.array(self.keys_T).T[0][np.newaxis, :]

        self.pol_s[self.m_P.flatten() == 0] = 0
        self.tor_s[self.m_T.flatten() == 0] = 0


        # find highest degree and order:
        self.N, self.M = np.max( np.hstack((np.array([c for c in tor_c.index]).T, np.array([c for c in tor_c.index]).T)), axis = 1)

        self.vectorgrid = self.vectorgrid()
        self.scalargrid = self.scalargrid(resolution = resolution)
        self.calculate_matrices()

    def update_model(self, tor_c, tor_s, iota_c, iota_s):
        """ update the model vector without doing the rest of the calculations """
        self.tor_c = tor_c.dropna().values.flatten()[:, np.newaxis]
        self.tor_s = tor_s.dropna().values.flatten()[:, np.newaxis]
        self.pol_c = iota_c.dropna().values.flatten()[:, np.newaxis]
        self.pol_s = iota_s.dropna().values.flatten()[:, np.newaxis]



    def vectorgrid(self, **kwargs):
        """ make grid for plotting vectors - using the equal_area_grid function in pyrkeland.plotting.grids

            kwargs are passed to equal_area_grid
        """

        grid = grids.equalAreaGrid(dr = self.dr, M0 = self.M0, returnarrays = True, **kwargs)
        mlt  = grid[1] + grid[2]/2. # shift to the center points of the bins
        mlat = grid[0] + (grid[0][1] - grid[0][0])/2  # shift to the center points of the bins

        mlt  = mlt[ (mlat >= self.minlat) & (mlat <= self.maxlat)]# & (mlat <=60 )]
        mlat = mlat[(mlat >= self.minlat) & (mlat <= self.maxlat)]# & (mlat <= 60)]

        mlat = np.hstack((mlat, -mlat)) # add southern hemisphere points
        mlt  = np.hstack((mlt ,  mlt)) # add southern hemisphere points


        return mlat[:, np.newaxis], mlt[:, np.newaxis] # reshape to column vectors and return


    def scalargrid(self, resolution = 100):
        """ make grid for calculations of scalar fields """

        if self.global_map:
            mlat, mlt = map(np.ravel, np.meshgrid(np.linspace(-self.maxlat, self.maxlat, resolution), np.linspace(-179.9, 179.9, resolution)))
        else:
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
        P = REFRE * (  np.dot(rtor * self.pol_P_scalar * self.pol_cosmphi_scalar, self.iota_c ) 
                     + np.dot(rtor * self.pol_P_scalar * self.pol_sinmphi_scalar, self.iota_s ) )

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


        # get polarsubplot objects:
        paxes = map(lambda x: polarsubplot.Polarsubplot(x, minlat = self.minlat, linestyle = ':', color = 'grey'), axes)

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
        paxes[1].plotpins(mlatv, mltv, nn , en, SCALE = VECTOR_SCALE, markersize = 2, unit = 'mA/m')
        paxes[5].plotpins(mlatv, mltv, -ns, es, SCALE = VECTOR_SCALE, markersize = 2, unit = 'mA/m')

        paxes[1].write(self.minlat-5, 12, r'$\alpha$ and $\mathbf{j}_{cf} = \nabla\alpha$, where $\nabla^2\alpha = - J_u$' , ha = 'center', va = 'bottom', size = 18)

        # Divergence-free horizontal
        Psin, Psis = self.get_equivalent_current_function()
        paxes[2].contour(mlats, mlts, Psin, levels = np.r_[Psin.min():Psin.max():30], colors = 'black', linewidths = .5)
        paxes[6].contour(mlats, mlts, Psis, levels = np.r_[Psis.min():Psis.max():30], colors = 'black', linewidths = .5)

        en, es, nn, ns = self.get_divergence_free_current()
        paxes[2].plotpins(mlatv, mltv, nn , en, SCALE = VECTOR_SCALE, markersize = 2, unit = 'mA/m')
        paxes[6].plotpins(mlatv, mltv, -ns, es, SCALE = VECTOR_SCALE, markersize = 2, unit = 'mA/m')
        paxes[2].write(self.minlat-5, 12, r'$\Psi$ and $\mathbf{j}_{df} = \mathbf{k}\times\nabla\Psi$' , ha = 'center', va = 'bottom', size = 18)

        # Total horizontal
        en, es, nn, ns = self.get_total_current()
        paxes[3].plotpins(mlatv, mltv, nn , en, SCALE = VECTOR_SCALE, markersize = 2, unit = 'mA/m')
        paxes[7].plotpins(mlatv, mltv, -ns, es, SCALE = VECTOR_SCALE, markersize = 2, unit = 'mA/m')

        paxes[3].write(self.minlat-5, 12, r'$\mathbf{j} = \mathbf{j}_{df} + \mathbf{j}_{cf}$' , ha = 'center', va = 'bottom', size = 18)


        # colorbar
        cbar = plt.subplot2grid((101, 4), (100, 0))
        cbar.contourf(np.vstack((faclevels, faclevels)), np.vstack((np.zeros_like(faclevels), np.ones_like(faclevels))), np.vstack((faclevels, faclevels)), levels = faclevels, cmap = plt.cm.bwr)
        cbar.set_yticks([0, 1])
        cbar.set_yticklabels(['', ''])
        cbar.set_xlabel('downward    $\hspace{3cm}\mu$A/m$^2\hspace{3cm}$      upward', size = 18)


        plt.subplots_adjust(hspace = 0, wspace = 0, left = .05, right = .95, bottom = .05, top = .95)
        plt.show()
