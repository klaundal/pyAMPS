"""
This module contains functions which are used to calculate magnetic local time accurately and efficiently.

Two main reasons for using this module to calculate MLT:
1) It is fast, since it handles lists of datetimes as input
2) It is in line with the definition of MLT in Laundal & Richmond (2017)



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

import pandas as pd # because of the DatetimeIndex class and IGRF coefficient interpolation
import numpy as np

d2r = np.pi/180
r2d = 180/np.pi

# first make arrays of IGRF dipole coefficients. This is used to make rotation matrix from geo to cd coords
# these values are from https://www.ngdc.noaa.gov/IAGA/vmod/igrf12coeffs.txt
time =[1900.0, 1905.0, 1910.0, 1915.0, 1920.0, 1925.0, 1930.0, 1935.0, 1940.0, 1945.0, 1950.0, 1955.0, 1960.0, 1965.0, 1970.0, 1975.0, 1980.0, 1985.0, 1990.0, 1995.0,   2000.0,    2005.0,    2010.0,   2015.0, 2020.0, 2025.0]
g10 = [-31543, -31464, -31354, -31212, -31060, -30926, -30805, -30715, -30654, -30594, -30554, -30500, -30421, -30334, -30220, -30100, -29992, -29873, -29775, -29692, -29619.4, -29554.63, -29496.57, -29441.46,  -29404.8]
g11 = [ -2298,  -2298,  -2297,  -2306,  -2317,  -2318,  -2316,  -2306,  -2292,  -2285,  -2250,  -2215,  -2169,  -2119,  -2068,  -2013,  -1956,  -1905,  -1848,  -1784,  -1728.2,  -1669.05,  -1586.42,  -1501.77,   -1450.9]
h11 = [  5922,   5909,   5898,   5875,   5845,   5817,   5808,   5812,   5821,   5810,   5815,   5820,   5791,   5776,   5737,   5675,   5604,   5500,   5406,   5306,   5186.1,   5077.99,   4944.26,   4795.99,    4652.5]
g10sv =  5.7 # secular variations
g11sv =  7.4
h11sv = -25.9
g10.append(g10[-1] + g10sv * 5) # append 2025 values using secular variation
g11.append(g11[-1] + g11sv * 5)
h11.append(h11[-1] + h11sv * 5)
igrf_dipole = pd.DataFrame({'g10':g10, 'g11':g11, 'h11':h11}, index = time)
igrf_dipole['B0'] = np.sqrt(igrf_dipole.g10**2 + igrf_dipole.g11**2 + igrf_dipole.h11**2)



def mlon_to_mlt(mlon, times, epoch):
    """ Calculate magnetic local time from magnetic longitude and time(s). 

    This is an implementation of the formula recommended in Laundal & Richmond, 2017 [4]_. 
    It uses the subsolar point geomagnetic (centered dipole) longitude to define
    the noon meridian. 

    Parameters
    ----------
    mlon : array_like
        array of magnetic longitudes
    times : datetime or list of datetimes
        datetime object, or list of datetimes with equal number of elements
        as mlon
    epoch : float
        the epoch (year, ) used for geo->mag conversion

    References
    ----------
    .. [4] Laundal, K.M. & Richmond, A.D. Space Sci Rev (2017) 206: 27. 
           https://doi.org/10.1007/s11214-016-0275-y

    See Also
    --------
    mlt_to_mlon: Magnetic longitude from magnetic local time and universal time

    """
    # flatten the input
    mlon = np.asarray(mlon).flatten() 

    ssglat, ssglon = map(np.array, subsol(times))
    sqlat, ssqlon = geo2mag(ssglat, ssglon, epoch)


    londiff = mlon - ssqlon
    londiff = (londiff + 180) % 360 - 180 # signed difference in longitude

    mlt = (180. + londiff)/15. # convert to mlt with ssqlon at noon

    return mlt


def mlt_to_mlon(mlt, times, epoch):
    """ Calculate quasi-dipole magnetic longitude from magnetic local time and universal time(s). 

    This is an implementation of the formula recommended in Laundal & Richmond, 2017 [4]_. 
    It uses the subsolar point geomagnetic (centered dipole) longitude to define
    the noon meridian. 

    Parameters
    ----------
    mlt : array_like
        array of magnetic local times
    times : datetime or list of datetimes
        datetime object, or list of datetimes with equal number of elements
        as mlon
    epoch : float
        the epoch (year, ) used for geo->mag conversion

    References
    ----------
    .. [4] Laundal, K.M. & Richmond, A.D. Space Sci Rev (2017) 206: 27. 
           https://doi.org/10.1007/s11214-016-0275-y

    See Also
    --------
    mlon_to_mlt: Magnetic local time from magnetic longitude and universal time

    """
    # flatten the input
    mlt = np.asarray(mlt).flatten() 

    ssglat, ssglon = map(np.array, subsol(times))
    sqlat, ssqlon = geo2mag(ssglat, ssglon, epoch)

    return (15 * mlt - 180 + ssqlon + 360) % 360



def sph_to_car(sph, deg = True):
    """ Convert from spherical to cartesian coordinates

    Parameters
    ----------
    sph : 3 x N array
        3 x N array, where the rows are, from top to bottom:
        radius, colatitude, and longitude
    deg : bool, optional
        set to True if input is given in degrees. False if radians

    Returns
    -------
    car : 3 x N array
        3 x N array, where the rows are, from top to bottom:
        x, y, z, in ECEF coordinates
    """

    r, theta, phi = sph

    if deg == False:
        conv = 1.
    else:
        conv = d2r


    return np.vstack((r * np.sin(theta * conv) * np.cos(phi * conv), 
                      r * np.sin(theta * conv) * np.sin(phi * conv), 
                      r * np.cos(theta * conv)))


def car_to_sph(car, deg = True):
    """ Convert from spherical to cartesian coordinates

    Parameters
    ----------
    car : 3 x N array
        3 x N array, where the rows are, from top to bottom:
        x, y, z, in ECEF coordinates
    deg : bool, optional
        set to True if output is wanted in degrees. False if radians
    
    Returns
    -------
    sph : 3 x N array
        3 x N array, where the rows are, from top to bottom:
        radius, colatitude, and longitude
    """

    x, y, z = car

    if deg == False:
        conv = 1.
    else:
        conv = r2d

    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z/r)*conv
    phi = ((np.arctan2(y, x)*180/np.pi) % 360)/180*np.pi * conv

    return np.vstack((r, theta, phi))


def subsol(datetimes):
    """ 
    calculate subsolar point at given datetime(s)

    Parameters
    ----------
    datetimes : datetime or list of datetimes
        datetime or list (or other iterable) of datetimes

    Returns
    -------
    subsol_lat : ndarray
        latitude(s) of the subsolar point
    subsol_lon : ndarray
        longiutde(s) of the subsolar point
    
    Raises
    ------
        ValueError
            if any datetime.year value provided is not within (1600,2100) 

    Note
    ----
    The code is vectorized, so it should be fast.

    After Fortran code by: 961026 A. D. Richmond, NCAR

    Documentation from original code:
    Find subsolar geographic latitude and longitude from date and time.
    Based on formulas in Astronomical Almanac for the year 1996, p. C24.
    (U.S. Government Printing Office, 1994).
    Usable for years 1601-2100, inclusive.  According to the Almanac, 
    results are good to at least 0.01 degree latitude and 0.025 degree 
    longitude between years 1950 and 2050.  Accuracy for other years 
    has not been tested.  Every day is assumed to have exactly
    86400 seconds; thus leap seconds that sometimes occur on December
    31 are ignored:  their effect is below the accuracy threshold of
    the algorithm.
    """

    # use pandas DatetimeIndex for fast access to year, month day etc...
    if hasattr(datetimes, '__iter__'): 
        datetimes = pd.DatetimeIndex(datetimes)
    else:
        datetimes = pd.DatetimeIndex([datetimes])

    year = datetimes.year
    # day of year:
    doy  = datetimes.dayofyear
    # seconds since start of day:
    ut   = datetimes.hour * 60.**2 + datetimes.minute*60. + datetimes.second 
 
    yr = year - 2000

    if year.max() >= 2100 or year.min() <= 1600:
        raise ValueError('subsol.py: subsol invalid after 2100 and before 1600')

    nleap = np.floor((year-1601)/4.)
    nleap = np.array(nleap) - 99

    # exception for years <= 1900:
    ncent = np.floor((year-1601)/100.)
    ncent = 3 - ncent
    nleap[year <= 1900] = nleap[year <= 1900] + ncent[year <= 1900]

    l0 = -79.549 + (-.238699*(yr-4*nleap) + 3.08514e-2*nleap)

    g0 = -2.472 + (-.2558905*(yr-4*nleap) - 3.79617e-2*nleap)

    # Days (including fraction) since 12 UT on January 1 of IYR:
    df = (ut/86400. - 1.5) + doy

    # Addition to Mean longitude of Sun since January 1 of IYR:
    lf = .9856474*df

    # Addition to Mean anomaly since January 1 of IYR:
    gf = .9856003*df

    # Mean longitude of Sun:
    l = l0 + lf

    # Mean anomaly:
    g = g0 + gf
    grad = g*np.pi/180.

    # Ecliptic longitude:
    lmbda = l + 1.915*np.sin(grad) + .020*np.sin(2.*grad)
    lmrad = lmbda*np.pi/180.
    sinlm = np.sin(lmrad)

    # Days (including fraction) since 12 UT on January 1 of 2000:
    n = df + 365.*yr + nleap

    # Obliquity of ecliptic:
    epsilon = 23.439 - 4.e-7*n
    epsrad  = epsilon*np.pi/180.

    # Right ascension:
    alpha = np.arctan2(np.cos(epsrad)*sinlm, np.cos(lmrad)) * 180./np.pi

    # Declination:
    delta = np.arcsin(np.sin(epsrad)*sinlm) * 180./np.pi

    # Subsolar latitude:
    sbsllat = delta

    # Equation of time (degrees):
    etdeg = l - alpha
    nrot = np.round(etdeg/360.)
    etdeg = etdeg - 360.*nrot

    # Apparent time (degrees):
    aptime = ut/240. + etdeg    # Earth rotates one degree every 240 s.

    # Subsolar longitude:
    sbsllon = 180. - aptime
    nrot = np.round(sbsllon/360.)
    sbsllon = sbsllon - 360.*nrot

    return sbsllat.values, sbsllon.values


def is_leapyear(year):
    """ Return True if leapyear else False

        Handles arrays and preserves shape
    
        Parameters
        ----------
        year : array_like
            array of years

        Returns
        -------
        is_leapyear : ndarray of bools
            True where input is leapyear, False elsewhere
    """

    # if array:
    if type(year) is np.ndarray:
        out = np.full_like(year, False, dtype = bool)

        out[ year % 4   == 0] = True
        out[ year % 100 == 0] = False
        out[ year % 400 == 0] = True

        return out

    # if scalar:
    if year % 400 == 0:
        return True

    if year % 100 == 0:
        return False

    if year % 4 == 0:
        return True

    else:
        return False


def geo2mag(glat, glon, epoch, deg = True, inverse = False):
    """ Convert geographic (geocentric) to centered dipole coordinates

    The conversion uses IGRF coefficients directly, interpolated
    to the provided epoch. The construction of the rotation matrix
    follows Laundal & Richmond (2017) [4]_ quite directly. 

    Parameters
    ----------
    glat : array_like
        array of geographic latitudes
    glon : array_like
        array of geographic longitudes
    epoch : float
        epoch (year) for the dipole used in the conversion
    deg : bool, optional
        True if input is in degrees, False otherwise
    inverse: bool, optional
        set to True to convert from magnetic to geographic. 
        Default is False

    Returns
    -------
    cdlat : ndarray
        array of centered dipole latitudes [degrees]
    cdlon : ndarray
        array of centered dipole longitudes [degrees]

    """
    glat = np.asarray(glat)
    glon = np.asarray(glon)

    # Find IGRF parameters for given epoch:
    dipole = igrf_dipole.reindex(list(igrf_dipole.index) + [epoch]).sort_index().interpolate().drop_duplicates() 
    dipole = dipole.loc[epoch, :]

    # make rotation matrix from geo to cd
    Zcd = -np.array([dipole.g11, dipole.h11, dipole.g10])/dipole.B0
    Zgeo_x_Zcd = np.cross(np.array([0, 0, 1]), Zcd)
    Ycd = Zgeo_x_Zcd / np.linalg.norm(Zgeo_x_Zcd)
    Xcd = np.cross(Ycd, Zcd)

    Rgeo_to_cd = np.vstack((Xcd, Ycd, Zcd))

    if inverse: # transpose rotation matrix to get inverse operation
        Rgeo_to_cd = Rgeo_to_cd.T

    # convert input to ECEF:
    colat = 90 - glat.flatten() if deg else np.pi/2 - glat.flatten()
    glon  = glon.flatten()
    r_geo = sph_to_car(np.vstack((np.ones_like(colat), colat, glon)), deg = deg)

    # rotate:
    r_cd = Rgeo_to_cd.dot(r_geo)

    # convert result back to spherical:
    _, colat_cd, lon_cd = car_to_sph(r_cd, deg = True)

    # return, converting colat to lat
    return 90 - colat_cd, lon_cd
