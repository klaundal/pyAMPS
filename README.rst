Overview
========

Python interface for the Average Magnetic field and Polar current System (AMPS) model

This module can be used to calculate and plot average magnetic field and current parameters on a grid. The parameters that are available for calculation/plotting are:

- field aligned current (scalar)
- equivalent current function (scalar)
- divergence-free part of horizontal current (vector)
- curl-free part of horizontal current (vector)
- total horizontal current (vector)
- eastward or northward ground perturbation corresponding to equivalent current (scalars)

Installation
------------

Using pip::

    pip install pyamps


Dependencies:

- numpy
- dask
- matplotlib
- scipy.interpolate (for plotting purposes)
- pandas (for reading csv file containing the coefficients)


References
----------
Laundal, K. M., Finlay, C. C. & Olsen, N. (2016), Sunlight effects on the 3D polar current system determined from low Earth orbit measurements. Earth Planets Space. `doi:10.1186/s40623-016-0518-x <https://earth-planets-space.springeropen.com/articles/10.1186/s40623-016-0518-x>`_ 
