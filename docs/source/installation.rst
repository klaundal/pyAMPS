Installation
============



The easiest way to install pyAMPS is using ``pip``::

    pip install pyamps

pyAMPS has the following dependencies:

- numpy
- dask
- matplotlib
- scipy (scipy.interpolate required for plotting purposes)
- pandas (for reading csv file containing model coefficients)
- apexpy (magnetic coordinate conversion)



pyAMPS can also be installed directly from source. You will then manually have to install the relevant dependencies. The source code can then be downloaded from Github and installed::

    git clone https://github.com/klaundal/pyAMPS
    cd pyAMPS 
    python setup.py install
