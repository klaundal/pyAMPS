# pytest conftest.py functions from https://docs.pytest.org/en/latest/example/simple.html

import pytest
import sys
import os

import pandas as pd

import pyamps


def pytest_runtest_makereport(item, call):
    """determine if previous function under fixture pytest.mark.incremental failed"""
    if "incremental" in item.keywords:
        if call.excinfo is not None:
            parent = item.parent
            parent._previousfailed = item


def pytest_runtest_setup(item):
    """set expected failure when prevous function failed"""
    if "incremental" in item.keywords:
        previousfailed = getattr(item.parent, "_previousfailed", None)
        if previousfailed is not None:
            pytest.xfail("previous test failed (%s)" % previousfailed.name)

def pytest_addoption(parser):
    parser.addoption("--skip_apex", action="store_true",
                     default=False, help="run tests dependent on apexpy")
def pytest_collection_modifyitems(config, items):
    if config.getoption("--skip_apex"):
        skip_apex = pytest.mark.skip(reason="need --run_apex option to run")
        for item in items:
            if "apex_dep" in item.keywords:
                item.add_marker(skip_apex)


# Fixtures for test functions
@pytest.fixture(scope="session")
def mpl_backend():
    import matplotlib as mpl
    mpl.use('Agg')
    mpl.rcParams['backend'] = 'Agg'

@pytest.fixture(scope="function")
def model_coeff():
    """Generate test data similar in form to AMPS model coefficients"""

    true_name = pyamps.model_utils.default_coeff_fn
    fake_name = os.path.abspath(os.path.join(pyamps.model_utils.basepath,'coefficients','test_model.txt'))
    pyamps.model_utils.default_coeff_fn = fake_name


    yield fake_name
    #yield true_name, fake_name
    pyamps.model_utils.default_coeff_fn = true_name
