from __future__ import division
import numpy as np
from scipy.interpolate import griddata


class Polarsubplot(object):
    def __init__(self, ax, minlat = 60, plotgrid = True, **kwargs):
        """ pax = Polarsubplot(axis, minlat = 50, plotgrid = True, **kwargs)
            
            **kwargs are the plot parameters for the grid 

            this is a class which handles plotting in polar coordinates, specifically
            an MLT/MLAT grid or similar

            Example:
            --------
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(111)   
            pax = Polarsubplot(ax)
            pax.MEMBERFUNCTION()
            plt.show()


            where memberfunctions include:
            plotgrid()                                   - called by __init__
            plot(mlat, mlt, **kwargs)                    - works like plt.plot
            write(mlat, mlt, text, **kwargs)             - works like plt.text
            scatter(mlat, mlt, **kwargs)                 - works like plt.scatter
            writeMLTlabels(mlat = self.minlat, **kwargs) - writes MLT at given mlat - **kwargs to plt.text
            plotarrows(mlats, mlts, north, east)         - works like plt.arrow (accepts **kwargs too)
            contour(mlat, mlt, f)                        - works like plt.contour
            contourf(mlat, mlt, f)                       - works like plt.contourf

        """
        self.minlat = minlat # the lower latitude boundary of the plot
        self.ax = ax
        self.ax.axis('equal')
        self.minlat = minlat

        self.ax.set_xlim(-1.1, 1.1)
        self.ax.set_ylim(-1.1, 1.1)
        self.ax.set_axis_off()


        if plotgrid:
            self.plotgrid(**kwargs)

    def plot(self, mlat, mlt, **kwargs):
        """ plot curve based on mlat, mlt. Calls matplotlib.plot, so any keywords accepted by this is also accepted here """

        x, y = self._mltMlatToXY(mlt, mlat)
        return self.ax.plot(x, y, **kwargs)

    def write(self, mlat, mlt, text, **kwargs):
        """ write text on specified mlat, mlt. **kwargs go to matplotlib.pyplot.text"""
        x, y = self._mltMlatToXY(mlt, mlat) 

        self.ax.text(x, y, text, **kwargs)

    def scatter(self, mlat, mlt, **kwargs):
        """ scatterplot on the polar grid. **kwargs go to matplotlib.pyplot.scatter """

        x, y = self._mltMlatToXY(mlt, mlat)
        c = self.ax.scatter(x, y, **kwargs)
        return c

    def plotgrid(self, **kwargs):
        """ plot mlt, mlat-grid on self.ax """

        self.ax.plot([-1, 1], [0 , 0], 'k--', **kwargs)
        self.ax.plot([0, 0], [-1, 1] , 'k--', **kwargs)

        latgrid = (90 - np.r_[self.minlat:90:10])/(90. - self.minlat)

        angles = np.linspace(0, 2*np.pi, 360)

        for lat in latgrid:
            self.ax.plot(lat*np.cos(angles), lat*np.sin(angles), 'k--', **kwargs)

    def writeMLTlabels(self, mlat = 60, degrees = False, **kwargs):
        """ write MLT labels at given latitude (default self.minlat) 
            if degrees is true, the longitude will be written instead of hour (with 0 at midnight)
        """

        if degrees:
            self.write(mlat, 0,    '0$^\circ$', verticalalignment = 'top'    , horizontalalignment = 'center', **kwargs)
            self.write(mlat, 6,   '90$^\circ$', verticalalignment = 'center' , horizontalalignment = 'left'  , **kwargs) 
            self.write(mlat, 12, '180$^\circ$', verticalalignment = 'bottom', horizontalalignment = 'center', **kwargs)
            self.write(mlat, 18, '-90$^\circ$', verticalalignment = 'center', horizontalalignment = 'right' , **kwargs)            
        else:
            self.write(mlat, 0,  '00', verticalalignment = 'top'    , horizontalalignment = 'center', **kwargs)
            self.write(mlat, 6,  '06', verticalalignment = 'center' , horizontalalignment = 'left'  , **kwargs) 
            self.write(mlat, 12, '12', verticalalignment = 'bottom' , horizontalalignment = 'center', **kwargs)
            self.write(mlat, 18, '18', verticalalignment = 'center' , horizontalalignment = 'right' , **kwargs)

    def featherplot(self, mlats, mlts, north, east, rotation = 0, SCALE = None, size = 10, unit = '', color = 'black', markercolor = 'black', marker = 'o', markersize = 20, **kwargs):
        """ like plotarrows, only it's not arrows but a dot with a line pointing in the arrow direction 
            
            kwargs go to ax.plot
            
            the markers at each pin can be modified by the following keywords, that go to ax.scatter:
            marker (default 'o')
            markersize (defult 20 - size in points^2)
            markercolor (default black)

        """

        mlts = mlts.flatten()
        mlats = mlats.flatten()
        north = north.flatten()
        east = east.flatten()
        R = np.array(([[np.cos(rotation), -np.sin(rotation)], [np.sin(rotation), np.cos(rotation)]]))

        if SCALE is None:
            scale = 1.
        else:

            if unit is not None:
                self.ax.plot([0.9, 1], [0.95, 0.95], color = color, linestyle = '-', linewidth = 2)
                self.ax.text(0.9, 0.95, ('{} '.format(SCALE) + unit), horizontalalignment = 'right', verticalalignment = 'center', size = size)

            scale = 0.1/SCALE

        for i in range(len(mlats)):

            mlt = mlts[i]
            mlat = mlats[i]

            x, y = self._mltMlatToXY(mlt, mlat)
            dx, dy = R.dot(self._northEastToCartesian(north[i], east[i], mlt).reshape((2, 1))).flatten()

            self.ax.plot([x, x + dx*scale], [y, y + dy*scale], color = color, **kwargs)
            if markersize != 0:
                self.ax.scatter(x, y, marker = marker, c = markercolor, s = markersize, edgecolors = markercolor)


    def contour(self, mlat, mlt, f, **kwargs):
        """ plot contour on grid, **kwargs are given to self.ax.contour. MLT in hours - no rotation
        """

        xea, yea = self._mltMlatToXY(mlt.flatten(), mlat.flatten())

        # convert to cartesian uniform grid
        xx, yy = np.meshgrid(np.linspace(-1, 1, 150), np.linspace(-1, 1, 150))
        points = np.vstack( tuple((xea, yea)) ).T
        gridf = griddata(points, f.flatten(), (xx, yy))

        # ... and plot
        return self.ax.contour(xx, yy, gridf, **kwargs)


    def contourf(self, mlat, mlt, f, **kwargs):
        """ plot contour on grid, **kwargs are given to self.ax.contour. MLT in hours - no rotation
        """

        xea, yea = self._mltMlatToXY(mlt.flatten(), mlat.flatten())

        # convert to cartesian uniform grid
        xx, yy = np.meshgrid(np.linspace(-1, 1, 150), np.linspace(-1, 1, 150))
        points = np.vstack( tuple((xea, yea)) ).T
        gridf = griddata(points, f.flatten(), (xx, yy))

        # ... and plot
        return self.ax.contourf(xx, yy, gridf, **kwargs)



    def _mltMlatToXY(self, mlt, mlat):
        r = (90. - np.abs(mlat))/(90. - self.minlat)
        a = (mlt - 6.)/12.*np.pi

        return r*np.cos(a), r*np.sin(a)

    def _XYtomltMlat(self, x, y):
        """ convert x, y to mlt, mlat, where x**2 + y**2 = 1 corresponds to self.minlat """
        x, y = np.array(x, ndmin = 1), np.array(y, ndmin = 1) # conver to array to allow item assignment

        lat = 90 - np.sqrt(x**2 + y**2)*(90. - self.minlat)
        mlt = np.arctan2(y, x)*12/np.pi + 6
        mlt[mlt < 0] += 24
        mlt[mlt > 24] -= 24

        return lat, mlt


    def _northEastToCartesian(self, north, east, mlt):
        a = (mlt - 6)/12*np.pi # convert MLT to angle with x axis (pointing from pole towards dawn)
        
        x1 = np.array([-north*np.cos(a), -north*np.sin(a)]) # arrow direction towards origin (northward)
        x2 = np.array([-east*np.sin(a),  east*np.cos(a)])   # arrow direction eastward

        return x1 + x2



def equalAreaGrid(dr = 2, K = 0, M0 = 8, N = 20):
    """ 
    mlat, mlt, mltres = equalAreaGrid(dr = 2, K = 0, M0 = 8, N = 20)

    mlat and mlt are the coordinates of the equatorward west ("lower left") corner
    of an equal area grid. mltres is the longitudinal width


    dr is the latitudinal resolution
    K gives the starting latitude r0 (grid not valid above this) from r0/dr = (2K + 1)/2 => K = (2r0/dr - 1)/2
    M0 is the number of sectors in the first circle
    N is the number of bins (lower boundary = r0 + dr*N)

    """

    r0 = dr * (2*K + 1)/2.

    assert M0 % (K + 1) == 0 # this must be fulfilled

    grid = {}

    M = M0
    grid[90 - r0 - dr] = np.linspace(0, 24 - 24./M, M) # these are the lower limits in MLT

    for i in range(1, N):

        M = M *  (1 + 1./(K + i + 1.)) # this is the partion for i + 1

        grid[90 - (r0 + i*dr) - dr] = np.linspace(0, 24 - 24./M, M) # these are the lower limits in MLT

    mlats = []
    mlts = []
    mltres = []
    for key in sorted(grid.keys()):
        mltres_ = sorted(grid[key])[1] - sorted(grid[key])[0]
        for mlt in sorted(grid[key]):
            mlats.append(key)
            mlts.append(mlt)
            mltres.append(mltres_)

    return np.array(mlats), np.array(mlts), np.array(mltres)

