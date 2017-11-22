#pytest conftest.py functions from https://docs.pytest.org/en/latest/example/simple.html

import pytest
import sys

# def pytest_cmdline_preparse(args):
#     """use #cpu's-1 for testing if pytest-xdist installed"""
#     if 'xdist' in sys.modules: # pytest-xdist plugin
#         import multiprocessing
#         num = max(multiprocessing.cpu_count()-1, 1)
#         args[:] = ["-n", str(num)] + args

# def pytest_addoption(parser):
#     """add --runslow commandline argument"""
#     parser.addoption("--runslow", action="store_true",
#                      default=False, help="run slow tests")

# def pytest_collection_modifyitems(config, items):
#     """skip test if test function has fixture pytest.mark.slow"""
#     if not config.getoption("--runslow"):
#         skip_slow = pytest.mark.skip(reason="need --runslow option to run")
#         for item in items:
#             if "slow" in item.keywords:
#                 item.add_marker(skip_slow)

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
            pytest.xfail("previous test failed (%s)" %previousfailed.name)

