import pyamps
import pytest


def test_success():
    assert True


def test_failure():
    with pytest.raises(AssertionError):
        assert False


def test_content():
    assert pyamps.AMPS
    assert pyamps.get_B_ground
    assert pyamps.get_B_space
    assert pyamps.mlon_to_mlt

    assert pyamps.amps
    assert pyamps.plot_utils
    assert pyamps.sh_utils
    assert pyamps.model_utils


def test_imports():
    try:
        import pyamps.model_vector_to_txt
    except ImportError:
        raise AssertionError("Module not found")
